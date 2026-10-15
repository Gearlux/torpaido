"""The ONNX backend: ``model.onnx`` from the exported program, run by onnxruntime — the ports are the file's own."""

import importlib.util
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import confluid
import numpy as np
import pytest
import torch
from conftest import needs_onnx
from matrainer.artifacts import CLASS_NAMES_FILE
from matrainer.model_package import ModelPackage
from matrainer.tracking.provenance import file_digest
from recordstream import Record, Transform
from recordstream.flow import parse_flow, run_steps_multi
from toy_graph import live_steps, sample_record

from torpaido.backends import BACKENDS, Onnx, backend_named
from torpaido.compile import CompilePackage
from torpaido.compiled import CompiledPackage
from torpaido.dataflow import measure
from torpaido.lowering import Context, ItemValue, lowers
from torpaido.manifest import COMPILED_FILE, PARITY_FILE, read_compiled
from torpaido.program import Program, build_program, interface_arrays

#: What ``live_steps`` returns: the parsed steps and the yielded step.
Toy = Tuple[List[Any], str]


@pytest.fixture
def toy(project: Path) -> Toy:
    return live_steps(str(project / "runs" / "best.pt"))


def _program(toy: Toy, samples: List[Record], envelope: Optional[Dict[str, Any]] = None) -> Program:
    graph = measure(*toy, samples)
    graph.prune_sidecars()
    return build_program(graph, samples, envelope=envelope)


def _read(path: Path) -> Tuple[Dict[str, Tuple[List[Any], int]], Dict[str, int]]:
    """The file's own graph ports — ``{name: (dims, elem type)}``, a dynamic dimension as its name — and its opsets,
    after the checker's full check."""
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model, full_check=True)
    ports: Dict[str, Tuple[List[Any], int]] = {}
    for value in list(model.graph.input) + list(model.graph.output):
        dims = [dim.dim_param or dim.dim_value for dim in value.type.tensor_type.shape.dim]
        ports[value.name] = (dims, value.type.tensor_type.elem_type)
    return ports, {entry.domain or "ai.onnx": entry.version for entry in model.opset_import}


@needs_onnx
def test_the_file_carries_the_ports_and_the_dynamic_dimensions_in_its_own_graph(
    toy: Toy, samples: List[Record], tmp_path: Path
) -> None:
    """No extra file names the ports: the ONNX graph's inputs and outputs ARE ``<entry>.<part>``, a dynamic
    dimension (the samples disagree on height and width) a named one, a fixed one its size, and the file passes
    ``onnx.checker``'s full check."""
    import onnx

    program = _program(toy, samples)

    files = Onnx().compile(program, tmp_path)

    assert files == [tmp_path / "model.onnx"]
    ports, opsets = _read(files[0])
    assert list(ports) == ["image.data", "scores.data", "label.value"]
    assert [port.shape for port in program.interface.inputs] == [[None, None, 3]]
    height, width, channels = ports["image.data"][0]
    assert isinstance(height, str) and isinstance(width, str) and channels == 3
    assert ports["image.data"][1] == onnx.TensorProto.UINT8
    assert ports["scores.data"] == ([2], onnx.TensorProto.FLOAT)
    assert ports["label.value"] == ([], onnx.TensorProto.INT64)
    assert opsets["ai.onnx"] >= 18


@needs_onnx
def test_a_varying_scalar_is_a_0_d_double_input(toy: Toy, samples: List[Record], tmp_path: Path) -> None:
    import onnx

    program = _program(toy, samples, {"image.data": "varies", "gain.value": "varies"})
    (path,) = Onnx().compile(program, tmp_path)

    ports, _ = _read(path)
    assert ports["gain.value"] == ([], onnx.TensorProto.DOUBLE)
    runner = Onnx().runner(tmp_path, [path.name])
    record = sample_record(4, 6, fill=200, gain=3.0)
    answer = runner(interface_arrays(program.interface, record))
    assert np.allclose(answer["scores.data"], run_steps_multi(record, *toy)[0]["scores"], atol=1e-6)


@needs_onnx
def test_a_compile_with_the_onnx_backend_verifies_through_onnxruntime_and_the_folder_runs_alone(
    package: Path, samples: List[Record], tmp_path: Path
) -> None:
    out = Path(
        CompilePackage(
            package=str(package), name="onnx", backend=Onnx(), samples=samples, envelope={"image.data": "varies"}
        ).run()
    )

    assert out == package / "compiled" / "onnx"
    assert sorted(p.name for p in out.iterdir()) == [CLASS_NAMES_FILE, COMPILED_FILE, "model.onnx", PARITY_FILE]
    info = read_compiled(out)
    assert info["backend"] == "onnx"
    assert info["files"] == [{"path": "model.onnx", "sha256": file_digest(out / "model.onnx")}]
    assert info["parity"] == {"records": 2, "status": "passed", "report": PARITY_FILE}
    assert backend_named("onnx") is Onnx and BACKENDS["onnx"] is Onnx

    record = sample_record(9, 2, fill=120)
    expected = ModelPackage(path=str(package))(record)
    assert expected is not None
    moved = tmp_path / "shipped"
    shutil.move(str(out), str(moved))
    shutil.rmtree(package)

    answer = CompiledPackage(path=str(moved))(record)

    assert np.allclose(answer["scores"], expected["scores"], atol=1e-6)
    assert answer["label"] == expected["label"]


@needs_onnx
def test_the_opset_is_the_config_s_and_below_the_exporter_s_floor_is_refused(
    toy: Toy, samples: List[Record], tmp_path: Path
) -> None:
    backend = confluid.load("backend: !class:torpaido.backends.Onnx {opset_version: 18}")["backend"]
    program = _program(toy, samples, {"image.data": "varies"})

    (path,) = backend.compile(program, tmp_path)

    assert _read(path)[1]["ai.onnx"] == 18
    with pytest.raises(ValueError) as refused:
        Onnx(opset_version=17).compile(program, tmp_path)
    assert str(refused.value) == "torpaido: the onnx backend writes opset 18 and up; opset_version 17 cannot be met"


def test_a_missing_package_is_refused_naming_the_extra(
    monkeypatch: pytest.MonkeyPatch, toy: Toy, samples: List[Record], tmp_path: Path
) -> None:
    """Checked before any import, so the refusal names the extra instead of a ``ModuleNotFoundError`` from inside
    torch — and this test needs no extra itself."""
    real = importlib.util.find_spec

    def missing(*names: str) -> None:
        monkeypatch.setattr(
            importlib.util, "find_spec", lambda name, *rest: None if name in names else real(name, *rest)
        )

    program = _program(toy, samples, {"image.data": "varies"})

    missing("onnxscript")
    with pytest.raises(ValueError) as refused:
        Onnx().compile(program, tmp_path)
    assert str(refused.value) == (
        "torpaido: the onnx backend needs onnxscript to write model.onnx — pip install torpaido[onnx]"
    )
    missing("onnxscript", "onnx")
    with pytest.raises(ValueError, match=r"needs onnxscript and onnx to write model.onnx"):
        Onnx().compile(program, tmp_path)
    missing("onnxruntime")
    with pytest.raises(ValueError) as refused:
        Onnx().runner(tmp_path, ["model.onnx"])
    assert str(refused.value) == (
        "torpaido: the onnx backend needs onnxruntime to run model.onnx — pip install torpaido[onnx]"
    )
    assert not (tmp_path / "model.onnx").exists()


# --------------------------------------------------------------------------- #
# complex ports, a scalar derived from a scalar, a stable sort, an op the exporter cannot translate
# --------------------------------------------------------------------------- #


class Envelope(Transform):
    """``iq`` (complex64 samples) -> ``envelope`` (|iq|, float32) and ``rotated`` (iq times 1j, complex64)."""

    def __call__(self, record: Record) -> Record:
        iq = np.asarray(record["iq"], dtype=np.complex64)
        return {**record, "envelope": np.abs(iq).astype(np.float32), "rotated": (iq * 1j).astype(np.complex64)}


@lowers(Envelope)
def lower_envelope(op: Envelope, reads: Dict[str, ItemValue], ctx: Context) -> Dict[str, ItemValue]:
    iq = reads["iq"].parts["data"]
    return {
        "envelope": ItemValue(np.ndarray, {"data": iq.abs().to(torch.float32)}),
        "rotated": ItemValue(np.ndarray, {"data": iq * 1j}),
    }


class Doubled(Transform):
    """``gain`` (a float entry) -> ``twice`` (a float entry): a scalar output derived from a scalar input."""

    def __call__(self, record: Record) -> Record:
        return {**record, "twice": float(record["gain"]) * 2.0}


@lowers(Doubled)
def lower_doubled(op: Doubled, reads: Dict[str, ItemValue], ctx: Context) -> Dict[str, ItemValue]:
    return {"twice": ItemValue(float, {"value": reads["gain"].value * 2.0})}


class Ranked(Transform):
    """``scores`` (float32 [N]) -> ``order`` (int64 [N]: descending, ties by index) and ``top`` (the scores so
    ordered) — the stable sort a detector library ranks its rows with."""

    def __call__(self, record: Record) -> Record:
        scores = np.asarray(record["scores"], dtype=np.float32)
        order = np.argsort(-scores, kind="stable").astype(np.int64)
        return {**record, "order": order, "top": scores[order]}


@lowers(Ranked)
def lower_ranked(op: Ranked, reads: Dict[str, ItemValue], ctx: Context) -> Dict[str, ItemValue]:
    scores = reads["scores"].parts["data"]
    order = torch.argsort(scores, descending=True, stable=True)
    return {"order": ItemValue(np.ndarray, {"data": order}), "top": ItemValue(np.ndarray, {"data": scores[order]})}


class Framed(Transform):
    """``iq`` -> ``frames`` (complex64 [M, 4]: windows of 4 samples every 2) — framed on the COMPLEX value, which
    the ONNX exporter has no translation for."""

    def __call__(self, record: Record) -> Record:
        iq = np.asarray(record["iq"], dtype=np.complex64)
        return {**record, "frames": np.stack([iq[start : start + 4] for start in range(0, len(iq) - 3, 2)])}


@lowers(Framed)
def lower_framed(op: Framed, reads: Dict[str, ItemValue], ctx: Context) -> Dict[str, ItemValue]:
    return {"frames": ItemValue(np.ndarray, {"data": reads["iq"].parts["data"].unfold(0, 4, 2)})}


def _samples_of(seed: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=length) + 1j * rng.normal(size=length)).astype(np.complex64)


def _lowered(flow: Dict[str, Any], samples: List[Record], envelope: Optional[Dict[str, Any]] = None) -> Any:
    """``(steps, result, program)`` of a one-step flow measured on ``samples``."""
    steps, result = parse_flow(flow, next(iter(flow)))
    graph = measure(steps, result, samples)
    graph.prune_sidecars()
    return steps, result, build_program(graph, samples, envelope=envelope)


@needs_onnx
def test_a_complex_port_is_carried_as_real_pairs_and_comes_back_complex(tmp_path: Path) -> None:
    """ONNX has no complex tensors: a complex port is the file's ``float32 [..., 2]`` (I and Q last), the manifest
    keeps ``complex64``, and the runner converts both ways — a flow sees the record's complex arrays only."""
    import onnx

    samples = [{"iq": _samples_of(0, 16)}, {"iq": _samples_of(1, 24)}]
    steps, result, program = _lowered({"env": {"op": Envelope()}}, samples)
    assert [(p.name, p.dtype, p.shape) for p in program.interface.inputs] == [("iq.data", "complex64", [None])]
    assert [(p.name, p.dtype) for p in program.interface.outputs] == [
        ("envelope.data", "float32"),
        ("rotated.data", "complex64"),
    ]

    (path,) = Onnx().compile(program, tmp_path)

    ports, _ = _read(path)
    length, pair = ports["iq.data"][0]
    assert isinstance(length, str) and pair == 2 and ports["iq.data"][1] == onnx.TensorProto.FLOAT
    assert ports["rotated.data"][0][-1] == 2 and ports["rotated.data"][1] == onnx.TensorProto.FLOAT
    assert len(ports["envelope.data"][0]) == 1 and ports["envelope.data"][1] == onnx.TensorProto.FLOAT
    record = {"iq": _samples_of(2, 20)}
    answer = Onnx().runner(tmp_path, [path.name])(interface_arrays(program.interface, record))
    expected = run_steps_multi(dict(record), steps, result)[0]
    assert answer["rotated.data"].dtype == np.complex64 and np.allclose(answer["rotated.data"], expected["rotated"])
    assert answer["envelope.data"].dtype == np.float32
    assert np.allclose(answer["envelope.data"], expected["envelope"], atol=1e-6)


@needs_onnx
def test_a_scalar_output_derived_from_a_scalar_input_keeps_its_shape(tmp_path: Path) -> None:
    """A 0-d input fed as a one-element array makes every scalar derived from it a one-element array, and the
    record's number comes back as a list (measured 2026-10-15 on a real package's ``time_end_s``)."""
    samples = [{"gain": 2.0}, {"gain": 3.0}]  # the samples disagree: the scalar is a 0-d input
    _, _, program = _lowered({"twice": {"op": Doubled()}}, samples)
    assert [p.shape for p in program.interface.inputs] == [[]]
    assert [(p.name, p.shape) for p in program.interface.outputs] == [("twice.value", [])]

    (path,) = Onnx().compile(program, tmp_path)
    answer = Onnx().runner(tmp_path, [path.name])(interface_arrays(program.interface, {"gain": 5.0}))

    assert answer["twice.value"].shape == () and float(answer["twice.value"]) == 10.0


@needs_onnx
def test_a_stable_sort_exports_as_onnx_s_sort(tmp_path: Path) -> None:
    """The exporter has no translation for ``aten.sort.stable``; the backend writes it as ONNX's sort (TopK), whose
    tie order is the runtime's — distinct scores rank exactly as the op ranks them."""
    samples = [{"scores": np.asarray([0.2, 0.9, 0.5, 0.1], dtype=np.float32)}]
    steps, result, program = _lowered({"rank": {"op": Ranked()}}, samples)

    (path,) = Onnx().compile(program, tmp_path)

    record = {"scores": np.asarray([0.3, 0.8, 0.7, 0.05], dtype=np.float32)}
    answer = Onnx().runner(tmp_path, [path.name])(interface_arrays(program.interface, record))
    expected = run_steps_multi(dict(record), steps, result)[0]
    assert np.array_equal(answer["order.data"], expected["order"]) and np.array_equal(
        answer["top.data"], expected["top"]
    )


@needs_onnx
def test_an_op_the_exporter_cannot_translate_is_refused_naming_the_step_and_the_lowering_s_line(
    tmp_path: Path,
) -> None:
    _, _, program = _lowered({"frames": {"op": Framed()}}, [{"iq": _samples_of(0, 16)}])

    with pytest.raises(ValueError) as refused:
        Onnx().compile(program, tmp_path)

    message = str(refused.value)
    assert message.startswith("torpaido: the onnx backend cannot translate aten.unfold.default on a complex value")
    assert "lower_framed" in message and "test_onnx.py:" in message and "step 'frames'" in message
    assert not (tmp_path / "model.onnx").exists()
