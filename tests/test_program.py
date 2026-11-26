"""The lowered graph is ONE torch program; the backends write it out and reproduce the Python graph."""

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
from conftest import needs_onnx
from recordstream import Label, Record, Transform
from recordstream.flow import parse_flow, run_steps_multi
from toy_graph import live_steps, sample_record

from torpaido.backends import Onnx, TorchExport, TorchScript, backend_named
from torpaido.dataflow import measure
from torpaido.lowering import Context, ItemValue, item_rule, lowers
from torpaido.program import build_program

#: What ``live_steps`` returns: the parsed steps and the yielded step.
Toy = Tuple[List[Any], str]


@pytest.fixture
def toy(project: Path) -> Toy:
    return live_steps(str(project / "runs" / "best.pt"))


def _python_answer(toy: Toy, record: Record) -> Record:
    return run_steps_multi(record, *toy)[0]


def _feed(program: Any, record: Record) -> Dict[str, np.ndarray]:
    """The artifact's input arrays for ``record``, by port name — what CompiledPackage does at run time."""
    arrays: Dict[str, np.ndarray] = {}
    for port in program.interface.inputs:
        parts, _ = item_rule(type(record[port.entry])).split(record[port.entry])
        arrays[port.name] = np.asarray(parts[port.part])
    return arrays


def test_the_interface_lists_the_tensor_ports_and_specializes_the_rest(toy: Toy, samples: List[Record]) -> None:
    graph = measure(*toy, samples)
    graph.prune_sidecars()

    program = build_program(graph, samples, envelope={"image.data": "varies"})

    inputs = {port.name: (port.dtype, port.shape, port.item) for port in program.interface.inputs}
    outputs = {port.name: (port.dtype, port.shape, port.item) for port in program.interface.outputs}
    assert inputs == {"image.data": ("uint8", [None, None, None], "recordstream.items.Image")}
    assert outputs == {
        "scores.data": ("float32", [2], "numpy.ndarray"),
        "label.value": ("int64", [], "recordstream.items.Label"),
    }
    assert program.interface.specialized == {"gain.value": 2.0, "image.layout": "HWC"}
    assert program.interface.constants == {"label.classes": None}
    assert program.interface.envelope == {"image.data": "varies"}


@pytest.mark.parametrize("backend", [TorchScript(), TorchExport(), pytest.param(Onnx(), marks=needs_onnx)])
def test_a_backend_reproduces_the_python_graph_on_a_picture_of_another_size(
    toy: Toy, samples: List[Record], tmp_path: Path, backend: Any
) -> None:
    graph = measure(*toy, samples)
    graph.prune_sidecars()
    program = build_program(graph, samples, envelope={"image.data": "varies"})

    files = backend.compile(program, tmp_path)
    runner = backend.runner(tmp_path, [path.name for path in files])
    record = sample_record(3, 7, fill=90)
    answer = runner(_feed(program, record))
    expected = _python_answer(toy, record)

    assert [path.name for path in files] == [backend.file_name]
    assert np.allclose(answer["scores.data"], expected["scores"], atol=1e-6)
    assert int(answer["label.value"]) == expected["label"].value
    assert backend_named(backend.name) is type(backend)


def test_a_varying_scalar_becomes_an_input_tensor(toy: Toy, samples: List[Record], tmp_path: Path) -> None:
    graph = measure(*toy, samples)
    graph.prune_sidecars()
    program = build_program(graph, samples, envelope={"image.data": "varies", "gain.value": "varies"})

    assert [port.name for port in program.interface.inputs] == ["image.data", "gain.value"]
    assert program.interface.inputs[1].dtype == "float64" and program.interface.inputs[1].shape == []
    assert "gain.value" not in program.interface.specialized
    backend = TorchScript()
    backend.compile(program, tmp_path)
    record = sample_record(4, 6, fill=200, gain=3.0)
    answer = backend.runner(tmp_path, [backend.file_name])(_feed(program, record))
    assert np.allclose(answer["scores.data"], _python_answer(toy, record)["scores"], atol=1e-6)


def test_a_read_scalar_the_samples_disagree_on_varies_without_being_named(toy: Toy, tmp_path: Path) -> None:
    """The toy's gain is read by the brighten step: two samples with different gains make it a 0-d input, as a
    tensor dimension the samples disagree on is dynamic — no envelope entry needed."""
    samples = [sample_record(gain=2.0), sample_record(gain=3.0)]
    graph = measure(*toy, samples)
    graph.prune_sidecars()

    program = build_program(graph, samples, envelope={"image.data": "varies"})

    assert [port.name for port in program.interface.inputs] == ["image.data", "gain.value"]
    assert program.interface.specialized == {"image.layout": "HWC"}
    backend = TorchScript()
    backend.compile(program, tmp_path)
    record = sample_record(4, 6, fill=200, gain=5.0)
    answer = backend.runner(tmp_path, [backend.file_name])(_feed(program, record))
    assert np.allclose(answer["scores.data"], _python_answer(toy, record)["scores"], atol=1e-6)


def test_an_envelope_value_still_refuses_a_sample_that_disagrees_with_it(toy: Toy) -> None:
    samples = [sample_record(gain=2.0), sample_record(gain=3.0)]
    graph = measure(*toy, samples)
    graph.prune_sidecars()

    with pytest.raises(ValueError, match="the envelope fixes gain.value to 2.0, but record 2 has 3.0"):
        build_program(graph, samples, envelope={"gain.value": 2.0})


def test_an_envelope_value_that_disagrees_with_the_samples_is_refused(toy: Toy, samples: List[Record]) -> None:
    graph = measure(*toy, samples)
    graph.prune_sidecars()

    with pytest.raises(ValueError, match="the envelope fixes gain.value to 5.0, but record 1 has 2.0"):
        build_program(graph, samples, envelope={"gain.value": 5.0})


def test_an_envelope_naming_no_input_part_is_refused_listing_them(toy: Toy, samples: List[Record]) -> None:
    graph = measure(*toy, samples)
    graph.prune_sidecars()

    with pytest.raises(
        ValueError,
        match=r"the envelope names 'picture.data', which is not an input part \(gain.value, image.data, image.layout\)",
    ):
        build_program(graph, samples, envelope={"picture.data": "varies"})


def test_the_program_keeps_each_node_s_eager_outputs_for_the_bisect(toy: Toy, samples: List[Record]) -> None:
    graph = measure(*toy, samples)
    graph.prune_sidecars()

    program = build_program(graph, samples)
    per_node = program.eager(samples[0])

    assert list(per_node) == ["pre/scale", "pre/bright", "classify"]
    assert isinstance(per_node["classify"]["label"], Label)
    assert np.allclose(per_node["classify"]["scores"], _python_answer(toy, samples[0])["scores"], atol=1e-6)


# --------------------------------------------------------------------------- #
# what the artifact reads is its interface
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Clip:
    """A toy capture: the samples, their rate, how the rate is applied — and two fields no step reads: a length
    derived from the samples and the source's name (the shape a domain's capture item has)."""

    data: Any = None
    rate: float = 1.0
    length_s: float = 0.0
    source: str = ""
    kind: str = "times"


class ScaleClip(Transform):
    """``clip`` -> ``scaled``: the samples times the rate (``kind: times``) or plus it (``kind: plus``)."""

    def __call__(self, record: Record) -> Record:
        clip = record["clip"]
        data = np.asarray(clip.data, dtype=np.float32)
        return {**record, "scaled": data * clip.rate if clip.kind == "times" else data + clip.rate}


@lowers(ScaleClip)
def lower_scale_clip(op: ScaleClip, reads: Dict[str, ItemValue], ctx: Context) -> Dict[str, ItemValue]:
    clip = reads["clip"]
    data, rate = clip.parts["data"].to(torch.float32), clip.parts["rate"]
    return {
        "scaled": ItemValue(np.ndarray, {"data": data * rate if clip.facts.get("kind") == "times" else data + rate})
    }


def _clip(
    values: List[float], rate: float = 2.0, length_s: float = 0.5, source: str = "a", kind: str = "times"
) -> Record:
    return {"clip": Clip(np.asarray(values, dtype=np.float32), rate, length_s, source, kind)}


def _clip_program(samples: List[Record], envelope: Optional[Dict[str, Any]] = None) -> Any:
    steps, result = parse_flow({"scale": {"op": ScaleClip()}}, "scale")
    graph = measure(steps, result, samples)
    graph.prune_sidecars()
    return build_program(graph, samples, envelope=envelope)


def test_an_input_field_no_step_reads_is_not_part_of_the_interface(tmp_path: Path) -> None:
    """Two samples disagreeing on the derived length and the source name compile: neither is an input nor fixed, and
    a record with yet other values runs — the artifact's interface is what its steps read."""
    samples = [_clip([1.0, 2.0, 3.0], length_s=0.5, source="a"), _clip([4.0, 5.0], length_s=0.7, source="b")]

    program = _clip_program(samples)

    assert [port.name for port in program.interface.inputs] == ["clip.data"]
    assert program.interface.specialized == {"clip.rate": 2.0, "clip.kind": "times"}
    backend = TorchScript()
    backend.compile(program, tmp_path)
    record = _clip([3.0, 4.0, 5.0], length_s=9.9, source="z")
    answer = backend.runner(tmp_path, [backend.file_name])(_feed(program, record))
    assert np.allclose(answer["scaled.data"], [6.0, 8.0, 10.0])


def test_a_read_scalar_the_samples_disagree_on_is_an_input_and_a_read_fact_is_refused(tmp_path: Path) -> None:
    program = _clip_program([_clip([1.0, 2.0], rate=2.0), _clip([3.0], rate=3.0)])
    assert [port.name for port in program.interface.inputs] == ["clip.data", "clip.rate"]
    assert program.interface.specialized == {"clip.kind": "times"}
    backend = TorchScript()
    backend.compile(program, tmp_path)
    answer = backend.runner(tmp_path, [backend.file_name])(_feed(program, _clip([1.0, 2.0], rate=5.0)))
    assert np.allclose(answer["scaled.data"], [5.0, 10.0])

    with pytest.raises(ValueError) as refused:
        _clip_program([_clip([1.0], kind="times"), _clip([1.0], kind="plus")])
    assert str(refused.value) == (
        "torpaido: the input clip.kind is a fact of the artifact, but the samples disagree: record 1 has 'times' "
        "and record 2 has 'plus'"
    )


def test_an_envelope_naming_a_field_no_step_reads_is_refused_listing_what_is_read() -> None:
    with pytest.raises(ValueError) as refused:
        _clip_program([_clip([1.0]), _clip([2.0])], envelope={"clip.length_s": "varies"})
    assert str(refused.value) == (
        "torpaido: the envelope names 'clip.length_s', which no step of the artifact reads "
        "(it reads clip.data, clip.kind, clip.rate)"
    )
