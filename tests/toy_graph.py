"""A package-shaped toy graph: recordstream's own ops plus three ops of this module, each with its lowering.

The graph has every part a real model package has — a preprocessing subgraph, a step whose parameter
is the weights file (the predictor's slot), training-only sidecars and a trailing drop — so the tests
can measure what the compiler prunes, what it declares as the artifact's inputs and outputs, and
that the artifact answers as the Python graph does.

    source record: image (uint8 HWC Image), gain (float), tag (str)

    pre:      Subgraph  scale (Scale: uint8 -> 0..1 float32)  ->  bright (Brighten: image * gain)
    level:    Mean      image -> level (a float entry)            # a sidecar: nothing downstream reads it
    stamp:    PutField  note = 7                                  # a sidecar
    classify: Classify  image -> scores (ndarray [2]), label (Label)   through a 1x1 conv read from weights_path
    done:     DropField note

The module is importable by the test process (pytest puts ``tests/`` on ``sys.path``), so the package
graph a test writes names these ops as ``!class:toy_graph.Brighten`` and confluid builds them.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from recordstream import Image, Label, Record, Transform
from recordstream.algorithm import Algorithm, Input, Output
from torch import nn

from torpaido.lowering import Context, ItemValue, lowers

__all__ = ["Brighten", "Classify", "Mean", "GRAPH", "HALF", "WEIGHTS_SLOT", "write_weights", "sample_record"]


class Brighten(Transform):
    """``image`` (an ``Image``) times the record's ``gain`` (a float entry) -> ``image``, float32.

    Args:
        field: The image entry.
        gain_field: The float entry holding the gain.
    """

    handles = (Image,)
    consumes = (Image,)
    produces = (Image,)

    def __init__(self, image_field: str = "image", gain_field: str = "gain") -> None:
        super().__init__()
        self.image_field = image_field
        self.gain_field = gain_field

    def __call__(self, record: Record) -> Record:
        image = record[self.image_field]
        gain = float(record[self.gain_field])
        out = Image(np.asarray(image, dtype=np.float32) * gain, layout=image.layout)
        return {**record, self.image_field: out}


class Mean(Algorithm):
    """The mean pixel of ``image`` as the float entry ``level``."""

    image: Image = Input(doc="The picture to average.")
    level: float = Output(doc="Its mean pixel value.")

    def compute(self) -> Dict[str, Any]:
        return {"level": float(np.asarray(self.image).mean())}


class Classify(Transform):
    """A 1x1 convolution over the picture, averaged -> ``scores`` (a ``[2]`` array) and ``label`` (its argmax).

    The weights are read from ``weights_path`` on the first call, the way a predictor reads its checkpoint.

    Args:
        weights_path: A ``torch.save``d state dict of a ``Conv2d(3, 2, 1)``; the package fills it.
        field: The image entry (float HWC).
        output: The entry the scores are written to.
        label_output: The entry the label is written to.
    """

    handles = (Image,)
    consumes = (Image,)
    produces = (np.ndarray, Label)

    def __init__(
        self, weights_path: str = "", image_field: str = "image", output: str = "scores", label_output: str = "label"
    ) -> None:
        super().__init__()
        self.weights_path = weights_path
        self.image_field = image_field
        self.output = output
        self.label_output = label_output
        self._net: Optional[nn.Module] = None

    def model(self) -> nn.Module:
        """The network, built and loaded once."""
        if self._net is None:
            if not self.weights_path:
                raise ValueError("Classify: weights_path is not set")
            net = nn.Conv2d(3, 2, 1)
            net.load_state_dict(torch.load(self.weights_path))
            self._net = net.eval()
        return self._net

    def __call__(self, record: Record) -> Record:
        picture = torch.as_tensor(np.ascontiguousarray(np.asarray(record[self.image_field]).transpose(2, 0, 1)))
        with torch.no_grad():
            scores = self.model()(picture.float().unsqueeze(0)).mean(dim=(2, 3))[0]
        return {**record, self.output: scores.numpy(), self.label_output: Label(int(scores.argmax()))}


class Tile(Transform):
    """``image`` repeated ``count`` times along its width — ``count`` read from the record (an int entry).

    Its lowering needs ``count`` as a CONSTANT (a repeat count shapes the output), which is what a
    varying envelope entry must be refused for.
    """

    handles = (Image,)
    consumes = (Image,)
    produces = (Image,)

    def __init__(self, image_field: str = "image", count_field: str = "count") -> None:
        super().__init__()
        self.image_field = image_field
        self.count_field = count_field

    def __call__(self, record: Record) -> Record:
        image = record[self.image_field]
        tiled = Image(np.tile(np.asarray(image), (1, int(record[self.count_field]), 1)), layout=image.layout)
        return {**record, self.image_field: tiled}


class Fickle(Transform):
    """Reads ``tag`` only when ``level`` is high — a data-dependent read, which has no static graph."""

    def __call__(self, record: Record) -> Record:
        if float(record["level"]) > 100.0:
            return {**record, "flag": str(record["tag"])}
        return {**record, "flag": "low"}


class NoLower(Transform):
    """Writes ``extra``; deliberately has NO lowering."""

    def __call__(self, record: Record) -> Record:
        return {**record, "extra": 1.0}


class ToySamples:
    """The sample records as a config can name them: ``samples: !class:toy_graph.ToySamples {}``."""

    def __init__(self, count: int = 2) -> None:
        self.count = count

    def __iter__(self) -> Any:
        return iter([sample_record(4, 6, fill=200), sample_record(5, 3, fill=40)][: self.count])


def live_steps(weights_path: str, **extra_steps: Any) -> Tuple[List[Any], str]:
    """The toy graph built from LIVE ops (no package): ``(steps, result)`` as ``parse_flow`` returns them.

    ``extra_steps`` are appended before ``done`` (each reading the previous step) — a test adds a
    sidecar or an op without a lowering this way.
    """
    from recordstream.flow import Subgraph, parse_flow
    from recordstream.ops.numpy import Scale
    from recordstream.ops.structure import DropField, PutField

    flow: Dict[str, Any] = {
        "pre": {"op": Subgraph(steps={"scale": {"op": Scale(field="image")}, "bright": {"op": Brighten()}})},
        "level": {"op": Mean(), "from": "pre"},
        "stamp": {"op": PutField(key="note", value=7), "from": "level"},
        "classify": {"op": Classify(weights_path=weights_path), "from": "stamp"},
    }
    previous = "classify"
    for name, op in extra_steps.items():
        flow[name] = {"op": op, "from": previous}
        previous = name
    flow["done"] = {"op": DropField(key="note"), "from": previous}
    return parse_flow(flow, "done")


# --------------------------------------------------------------------------- #
# The lowerings — torch functions of the configured op and the values it reads
# --------------------------------------------------------------------------- #


@lowers(Tile)
def lower_tile(op: Tile, reads: Dict[str, ItemValue], ctx: Context) -> Dict[str, ItemValue]:
    image = reads[op.image_field]
    count = ctx.constant(reads[op.count_field].value, op.count_field)
    return {op.image_field: image.replace(data=image.data.repeat(1, int(count), 1))}


@lowers(Brighten)
def lower_brighten(op: Brighten, reads: Dict[str, ItemValue], ctx: Context) -> Dict[str, ItemValue]:
    image, gain = reads[op.image_field], reads[op.gain_field]
    return {op.image_field: image.replace(data=image.data.to(torch.float32) * gain.value)}


@lowers(Mean)
def lower_mean(op: Mean, reads: Dict[str, ItemValue], ctx: Context) -> Dict[str, ItemValue]:
    return {"level": ItemValue(float, {"value": reads["image"].data.to(torch.float32).mean()})}


@lowers(Classify)
def lower_classify(op: Classify, reads: Dict[str, ItemValue], ctx: Context) -> Dict[str, ItemValue]:
    net = ctx.module("net", op.model())
    picture = reads[op.image_field].data.to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    scores = net(picture).mean(dim=(2, 3))[0]
    return {
        op.output: ItemValue(np.ndarray, {"data": scores}),
        op.label_output: ItemValue(Label, {"value": scores.argmax()}, {"classes": None}),
    }


# --------------------------------------------------------------------------- #
# The package files
# --------------------------------------------------------------------------- #

HALF = """\
scale:
  op: !class:recordstream.ops.numpy.Scale {field: image}
bright:
  op: !class:toy_graph.Brighten {}
"""

GRAPH = """\
pre:
  include: ../graphs/half.steps.yaml
flow:
  pre:
    op: !class:recordstream.flow.subgraph.Subgraph
      steps: !ref:pre
  level:
    op: !class:toy_graph.Mean {}
    from: pre
  stamp:
    op: !class:recordstream.ops.structure.PutField {key: note, value: 7}
    from: level
  classify:
    op: !class:toy_graph.Classify
      weights_path: ""
    from: stamp
  done:
    op: !class:recordstream.ops.structure.DropField {key: note}
    from: classify
outputs: done
"""

WEIGHTS_SLOT = "flow.classify.op.weights_path"


def write_weights(path: Path) -> None:
    """A ``Conv2d(3, 2, 1)`` state dict with fixed weights, so every test classifies the same way."""
    net = nn.Conv2d(3, 2, 1)
    assert net.bias is not None
    with torch.no_grad():
        net.weight.copy_(torch.tensor([[[[1.0]], [[0.0]], [[0.0]]], [[[0.0]], [[1.0]], [[1.0]]]]))
        net.bias.copy_(torch.tensor([0.1, -0.1]))
    torch.save(net.state_dict(), path)


def sample_record(height: int = 4, width: int = 6, fill: int = 200, gain: float = 2.0) -> Record:
    """A source record: a ``uint8`` HWC picture whose red channel is ``fill`` and the rest dark, a gain, a tag."""
    picture = np.zeros((height, width, 3), dtype=np.uint8)
    picture[..., 0] = fill
    picture[..., 1] = fill // 4
    return {"image": Image(picture), "gain": gain, "tag": "a capture"}
