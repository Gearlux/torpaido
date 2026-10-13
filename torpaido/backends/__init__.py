"""Backends: each takes the exported program and writes its files; each gives a runner that runs them on arrays.

``torch.export`` is the hub — a backend never sees a lowering or a record, only the
:class:`~torpaido.program.Program` and the folder to write into. The port names travel INSIDE the
artifact (an extra file of the ``.pt`` / ``.pt2``), so a runner built from the folder alone knows the
input order and the output names: the compiled folder stands on its own.

A backend is ``@configurable(category="compiler_backend")``, so a compile config picks it with a
``!class:`` marker and a discovery service lists them; :func:`backend_named` finds a backend class by
the name its manifest carries, for the op that runs a compiled folder.
"""

import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Protocol, Type, runtime_checkable

import numpy as np
from confluid import configurable

from torpaido.program import Program

__all__ = ["BACKENDS", "Backend", "Runner", "TorchExport", "TorchScript", "backend_named"]

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
    """TorchScript: ``model.pt``, a trace of the exported module — what a ``torch::jit::load`` runtime takes.

    TorchScript is in maintenance upstream; it stays a backend because the deployment runtimes that exist
    load it, and the manifest is the same whichever backend wrote the files.
    """

    name = "torchscript"
    file_name = "model.pt"

    def compile(self, program: Program, out: Path) -> List[Path]:
        import torch

        path = Path(out) / self.file_name
        with _torchscript_api():
            traced = torch.jit.trace(program.exported.module(), program.example_inputs, strict=False, check_trace=False)
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


#: Backend name (as a manifest carries it) -> the backend class.
BACKENDS: Dict[str, Type[Any]] = {TorchScript.name: TorchScript, TorchExport.name: TorchExport}


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
