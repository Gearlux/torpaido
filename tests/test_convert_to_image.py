"""The ConvertToImage lowering renders a 2-D map as the op does: gray, quantized by truncation, sized by its rule."""

from typing import Any, Dict, List

import numpy as np
import pytest
from conftest import needs_onnx
from recordstream import Image, Record
from recordstream.flow import parse_flow, run_steps_multi
from recordstream.ops.image import ConvertToImage

from torpaido.backends import Onnx, TorchScript
from torpaido.dataflow import measure
from torpaido.parity import compare
from torpaido.program import build_program


def _records(seed: int = 0, shape: tuple = (40, 25)) -> List[Record]:
    rng = np.random.default_rng(seed)
    surface = rng.normal(size=shape).astype(np.float32) * 20.0 - 60.0
    surface[3, 4] = np.nan
    surface[5, 6] = np.inf
    return [{"map": surface, "tag": "x"}]


def _lowered(op: ConvertToImage, records: List[Record]) -> Dict[str, Any]:
    steps, result = parse_flow({"picture": {"op": op}}, "picture")
    graph = measure(steps, result, records)
    graph.prune_sidecars()
    return build_program(graph, records).eager(records[0])["picture"]


def _python(op: ConvertToImage, record: Record) -> Record:
    return run_steps_multi(record, *parse_flow({"picture": {"op": op}}, "picture"))[0]


@pytest.mark.parametrize(
    "settings",
    [
        {"field": "map", "width": 64, "height": 48, "vmin": -100.0, "vmax": -20.0},  # fixed scale, exact size
        {"field": "map", "width": None, "height": None},  # min-max scale, the pixels as they are
        {"field": "map", "width": 50, "height": None},  # the width set, the height kept
        {"field": "map", "width": 0, "height": 80},  # the width follows the height
        {"field": "map", "width": 0, "height": 0, "max_size": 20},  # the longest side bounded
        {"field": "map", "width": 32, "height": 32, "flip_vertical": True},
    ],
)
def test_the_rendered_picture_matches_the_op_within_one_of_255(settings: Dict[str, Any]) -> None:
    op = ConvertToImage(colormap="gray", **settings)
    records = _records()

    produced = _lowered(op, records)["image"]
    expected = _python(op, records[0])["image"]

    assert isinstance(produced, Image) and produced.layout == "HWC"
    assert produced.shape == expected.shape and produced.dtype == np.uint8
    assert compare({"image": expected}, {"image": produced}, ["image"], {"image": {"atol": 1.0, "rtol": 0.0}}) == []


def test_a_map_the_op_cannot_scale_renders_black_as_the_op_does() -> None:
    op = ConvertToImage(colormap="gray", field="map", width=None, height=None)
    records = [{"map": np.full((4, 5), 3.0, dtype=np.float32)}]

    produced = _lowered(op, records)["image"]

    assert not produced.any() and produced.shape == (4, 5, 3)


def test_another_colormap_is_refused_by_name() -> None:
    op = ConvertToImage(colormap="viridis", field="map", width=8, height=8)
    records = _records()
    steps, result = parse_flow({"picture": {"op": op}}, "picture")
    graph = measure(steps, result, records)
    graph.prune_sidecars()

    with pytest.raises(
        ValueError, match=r"step 'picture' \(ConvertToImage\) uses the colormap 'viridis', which is not lowered yet"
    ):
        build_program(graph, records)


@pytest.mark.parametrize("backend", [TorchScript(), pytest.param(Onnx(), marks=needs_onnx)])
def test_an_artifact_keeps_a_raster_following_height_dynamic(tmp_path: Any, backend: Any) -> None:
    """A picture as tall as its raster (``height=None``) resized to a fixed width: under ``torch.export`` the target
    height is a ``sym_size`` node, which a trace of the EXPORTED module bakes to the sample's (measured 2026-10-13: a
    40-row record came out 30 rows). The TorchScript backend traces the lowered module, and the ONNX export keeps the
    exported program's ``sym_size`` as a ``Shape`` node, so either artifact's picture is as tall as each record's
    raster."""
    from torpaido.program import interface_arrays

    op = ConvertToImage(colormap="gray", field="map", width=50, height=None, vmin=-100.0, vmax=-20.0)
    samples = _records(seed=1, shape=(30, 25)) + _records(seed=2, shape=(40, 25))
    steps, result = parse_flow({"picture": {"op": op}}, "picture")
    graph = measure(steps, result, samples)
    graph.prune_sidecars()
    program = build_program(graph, samples, envelope={"map.data": "varies"})
    files = backend.compile(program, tmp_path)
    runner = backend.runner(tmp_path, [path.name for path in files])

    for rows in (30, 34, 40, 7):
        (record,) = _records(seed=3, shape=(rows, 25))
        produced = runner(interface_arrays(program.interface, record))["image.data"]
        assert tuple(produced.shape) == (rows, 50, 3), f"{rows} rows in, {produced.shape} out"
