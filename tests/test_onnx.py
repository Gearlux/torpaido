"""The ONNX backend: ``model.onnx`` from the exported program, run by onnxruntime — the ports are the file's own."""

import importlib.util
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import confluid
import numpy as np
import pytest
from conftest import needs_onnx
from matrainer.artifacts import CLASS_NAMES_FILE
from matrainer.model_package import ModelPackage
from matrainer.tracking.provenance import file_digest
from recordstream import Record
from recordstream.flow import run_steps_multi
from toy_graph import live_steps, sample_record

from torpaido.backends import BACKENDS, Onnx, backend_named
from torpaido.compile import CompilePackage
from torpaido.compiled import CompiledPackage
from torpaido.dataflow import measure
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
