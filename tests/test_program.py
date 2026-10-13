"""The lowered graph is ONE torch program; the backends write it out and reproduce the Python graph."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from recordstream import Label, Record
from recordstream.flow import run_steps_multi
from toy_graph import live_steps, sample_record

from torpaido.backends import TorchExport, TorchScript, backend_named
from torpaido.dataflow import measure
from torpaido.lowering import item_rule
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


@pytest.mark.parametrize("backend", [TorchScript(), TorchExport()])
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


def test_a_fixed_input_must_agree_across_the_samples(toy: Toy) -> None:
    samples = [sample_record(gain=2.0), sample_record(gain=3.0)]
    graph = measure(*toy, samples)
    graph.prune_sidecars()

    with pytest.raises(ValueError) as refused:
        build_program(graph, samples)
    assert str(refused.value) == (
        "torpaido: the input gain.value is fixed, but the samples disagree: record 1 has 2.0 and record 2 has 3.0\n"
        "  Let it vary (envelope: gain.value: varies) or compile from samples that agree."
    )


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
