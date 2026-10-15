"""Backends: each takes the exported program and writes its files; each gives a runner that runs them on arrays.

``torch.export`` is the hub — a backend never sees a lowering or a record, only the
:class:`~torpaido.program.Program` and the folder to write into. The port names travel INSIDE the
artifact (an extra file of the ``.pt`` / ``.pt2``; the ONNX graph's own input and output names), so a
runner built from the folder alone knows the input order and the output names: the compiled folder
stands on its own.

A backend is ``@configurable(category="compiler_backend")``, so a compile config picks it with a
``!class:`` marker and a discovery service lists them; :func:`backend_named` finds a backend class by
the name its manifest carries, for the op that runs a compiled folder.
"""

import importlib.util
import json
import logging
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Sequence, Type, runtime_checkable

import numpy as np
from confluid import configurable

from torpaido.program import Program

__all__ = ["BACKENDS", "Backend", "Onnx", "Runner", "TorchExport", "TorchScript", "backend_named"]

#: The extra file inside a torch artifact that names its ports.
INTERFACE_FILE = "torpaido_interface.json"


@runtime_checkable
class Runner(Protocol):
    def __call__(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]: ...


@runtime_checkable
class Backend(Protocol):
    """What every backend provides."""

    name: str
    file_name: str

    def compile(self, program: Program, out: Path) -> List[Path]: ...

    def runner(self, folder: Path, files: List[str]) -> Runner: ...


class _TorchRunner:
    """Runs a loaded torch module on arrays: tensors in port order in, arrays by port name out."""

    def __init__(self, module: Any, inputs: List[str], outputs: List[str]) -> None:
        import torch

        self._module = module
        self._inputs = inputs
        self._outputs = outputs
        self._torch = torch

    def __call__(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        tensors = [self._torch.as_tensor(arrays[name]) for name in self._inputs]
        with self._torch.no_grad():
            out = self._module(*tensors)
        if isinstance(out, self._torch.Tensor):
            out = (out,)
        return {name: tensor.detach().cpu().numpy() for name, tensor in zip(self._outputs, out)}


@contextmanager
def _torchscript_api() -> Any:
    """torch.jit is in maintenance and says so on every call; a user who picked this backend knows. The trace
    also warns that a tensor-to-bool conversion (the exported graph's own shape checks) "might not generalize" —
    parity on the sample records is what decides that, so the warning is noise here."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.jit.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\._jit_internal.*")
        try:
            import torch.jit

            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        except (ImportError, AttributeError):  # pragma: no cover - an older torch without the class
            pass
        yield


def _interface_json(program: Program) -> str:
    return json.dumps(
        {
            "inputs": [port.name for port in program.interface.inputs],
            "outputs": [port.name for port in program.interface.outputs],
        }
    )


@configurable(category="compiler_backend")
class TorchScript:
    """TorchScript: ``model.pt``, a trace of the LOWERED module — what a ``torch::jit::load`` runtime takes.

    The trace is of the eager lowered module (``program.module``), not of the exported program's module: under
    ``torch.export`` a size a lowering reads off a tensor (a picture as tall as its raster, resized to a fixed width)
    becomes a ``sym_size.int`` node, and ``torch.jit.trace`` turns that node's Python int into a CONSTANT — the
    artifact then resizes every record to the trace sample's size (measured 2026-10-13: a 40-row picture came out
    30 rows, the first sample's). Traced eagerly, the same size is an ``aten::size`` the trace keeps dynamic. The
    exported program still validates the lowering under the envelope and is what :class:`TorchExport` saves.

    TorchScript is in maintenance upstream; it stays a backend because the deployment runtimes that exist
    load it, and the manifest is the same whichever backend wrote the files.
    """

    name = "torchscript"
    file_name = "model.pt"

    def compile(self, program: Program, out: Path) -> List[Path]:
        import torch

        path = Path(out) / self.file_name
        with _torchscript_api():
            traced = torch.jit.trace(program.module, program.example_inputs, strict=False, check_trace=False)
            torch.jit.save(traced, str(path), _extra_files={INTERFACE_FILE: _interface_json(program)})
        return [path]

    def runner(self, folder: Path, files: List[str]) -> Runner:
        import torch

        extra = {INTERFACE_FILE: ""}
        with _torchscript_api():
            module = torch.jit.load(str(Path(folder) / self.file_name), map_location="cpu", _extra_files=extra)
        ports = json.loads(extra[INTERFACE_FILE])
        return _TorchRunner(module.eval(), ports["inputs"], ports["outputs"])


@configurable(category="compiler_backend")
class TorchExport:
    """The exported program itself: ``program.pt2`` (``torch.export.save``) — the modern torch deployment form."""

    name = "torch-export"
    file_name = "program.pt2"

    def compile(self, program: Program, out: Path) -> List[Path]:
        import torch

        path = Path(out) / self.file_name
        torch.export.save(program.exported, str(path), extra_files={INTERFACE_FILE: _interface_json(program)})
        return [path]

    def runner(self, folder: Path, files: List[str]) -> Runner:
        import torch

        extra = {INTERFACE_FILE: ""}
        exported = torch.export.load(str(Path(folder) / self.file_name), extra_files=extra)
        ports = json.loads(extra[INTERFACE_FILE])
        return _TorchRunner(exported.module(), ports["inputs"], ports["outputs"])


def _require(names: Sequence[str], purpose: str) -> None:
    """Refuse by name the packages of the ``[onnx]`` extra that are missing — checked before any import, so a bare
    ``ModuleNotFoundError`` from inside torch never reaches the user."""
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError(
            f"torpaido: the onnx backend needs {' and '.join(missing)} to {purpose} — pip install torpaido[onnx]"
        )


#: The exporter's own loggers, which narrate every rewrite pass at INFO ("Removed 4 unused nodes") — diagnostics,
#: held to WARNING while the backend exports so the compile's log keeps the operator's conditions only.
_EXPORTER_LOGGERS = ("onnxscript", "onnx_ir")


@contextmanager
def _onnx_export_api() -> Iterator[None]:
    """The dynamo exporter deep-copies a pytree spec, and torch warns about its own ``LeafSpec`` from inside itself
    (a FutureWarning raised in ``copyreg``) — nothing a user of this backend can act on; its passes narrate at
    INFO (see :data:`_EXPORTER_LOGGERS`)."""
    loggers = [logging.getLogger(name) for name in _EXPORTER_LOGGERS]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=r".*LeafSpec.*")
            yield
    finally:
        for logger, level in zip(loggers, levels):
            logger.setLevel(level)


class _OnnxRunner:
    """Runs an onnxruntime session on arrays: fed by port name, the outputs by port name — the file's own names."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._inputs = [port.name for port in session.get_inputs()]
        self._outputs = [port.name for port in session.get_outputs()]

    def __call__(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feed = {name: np.ascontiguousarray(arrays[name]) for name in self._inputs}
        return dict(zip(self._outputs, self._session.run(None, feed)))


#: The lowest opset the dynamo exporter writes (asked for 17, it wrote 18 — measured with torch 2.13).
ONNX_OPSET_FLOOR = 18


@configurable(category="compiler_backend")
class Onnx:
    """ONNX: ``model.onnx`` — the exported program through ``torch.onnx.export(dynamo=True)``, run by onnxruntime.

    One file, the weights inline, passed through ``onnx.checker``'s full check before it is kept. The port names
    need no extra file here: the ONNX graph's own inputs and outputs are ``<entry>.<part>``, a dynamic dimension a
    named one — what every ONNX runtime reads. The manifest stays the contract: a dimension ``varies`` freed and
    export then fixed (a channel count a convolution needs) is written under the exporter's name for it, not as a
    size (measured: ``[s95, s85, "3"]``), and the compiled op checks records against ``compiled.yaml``. Needs the
    ``[onnx]`` extra (``onnx``, ``onnxruntime``, ``onnxscript``); without it the compile and the runner are refused
    naming the extra. The runner is a CPU session; the compiled op's ``device`` does not reach it.

    Args:
        opset_version: The ONNX opset the file targets — what the deployment runtime supports. ``None`` = the
            exporter's recommended one (20 with torch 2.13). Below :data:`ONNX_OPSET_FLOOR` is refused rather than
            silently raised to it.
    """

    name = "onnx"
    file_name = "model.onnx"

    def __init__(self, opset_version: Optional[int] = None) -> None:
        self.opset_version = opset_version

    def compile(self, program: Program, out: Path) -> List[Path]:
        if self.opset_version is not None and self.opset_version < ONNX_OPSET_FLOOR:
            raise ValueError(
                f"torpaido: the onnx backend writes opset {ONNX_OPSET_FLOOR} and up; "
                f"opset_version {self.opset_version} cannot be met"
            )
        _require(("onnxscript", "onnx"), f"write {self.file_name}")
        import onnx
        import torch

        path = Path(out) / self.file_name
        with _onnx_export_api():
            exported = torch.onnx.export(
                program.exported,
                program.example_inputs,
                input_names=[port.name for port in program.interface.inputs],
                output_names=[port.name for port in program.interface.outputs],
                opset_version=self.opset_version,
                dynamo=True,
                verbose=False,
            )
        if (
            exported is None
        ):  # pragma: no cover - dynamo always returns the program; the hint is Optional for the legacy path
            raise ValueError(f"torpaido: torch.onnx.export gave no program to write to {path}")
        exported.save(str(path), external_data=False)
        onnx.checker.check_model(str(path), full_check=True)
        return [path]

    def runner(self, folder: Path, files: List[str]) -> Runner:
        _require(("onnxruntime",), f"run {self.file_name}")
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3  # errors only: a session's warnings are about its own graph rewrites
        session = onnxruntime.InferenceSession(
            str(Path(folder) / self.file_name), options, providers=["CPUExecutionProvider"]
        )
        return _OnnxRunner(session)


#: Backend name (as a manifest carries it) -> the backend class.
BACKENDS: Dict[str, Type[Any]] = {
    TorchScript.name: TorchScript,
    TorchExport.name: TorchExport,
    Onnx.name: Onnx,
}


def backend_named(name: str) -> Type[Any]:
    """The backend class a manifest's ``backend:`` names.

    Raises:
        ValueError: no backend of that name, listing the known ones.
    """
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"torpaido: no backend named {name!r} (the backends are: {', '.join(sorted(BACKENDS))})"
        ) from None
