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
import re
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple, Type, runtime_checkable

import numpy as np
from confluid import configurable

from torpaido.lowering.registry import lowering_for
from torpaido.program import STEP_ANNOTATION, Program, dotted

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


#: The exporter's own loggers, which narrate every rewrite pass at INFO ("Removed 4 unused nodes") and at WARNING
#: ("Skipping constant folding for op Split with multiple outputs", "Skipped deduplication of initializer") —
#: diagnostics of its passes, nothing an operator acts on — held to ERROR while the backend exports so the compile's
#: log keeps the operator's conditions only.
_EXPORTER_LOGGERS = ("onnxscript", "onnx_ir")


@contextmanager
def _onnx_export_api() -> Iterator[None]:
    """The dynamo exporter deep-copies a pytree spec, and torch warns about its own ``LeafSpec`` from inside itself
    (a FutureWarning raised in ``copyreg``) — nothing a user of this backend can act on; its passes narrate at
    INFO and WARNING (see :data:`_EXPORTER_LOGGERS`)."""
    loggers = [logging.getLogger(name) for name in _EXPORTER_LOGGERS]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=r".*LeafSpec.*")
            yield
    finally:
        for logger, level in zip(loggers, levels):
            logger.setLevel(level)


#: The ONNX model's metadata entry naming the ports that are complex in the record and real pairs in the file.
COMPLEX_PORTS_KEY = "torpaido.complex_ports"
#: The real dtype a complex port's pairs are written in.
_PAIR_DTYPE = {"complex64": np.float32, "complex128": np.float64}


def _pairs(array: np.ndarray) -> np.ndarray:
    """A complex array as its real pairs, ``[..., 2]`` (I then Q) — ONNX's only way to carry a complex tensor."""
    return np.ascontiguousarray(np.stack([array.real, array.imag], axis=-1).astype(_PAIR_DTYPE[array.dtype.name]))


def _complex(pairs: np.ndarray) -> np.ndarray:
    """The complex array a ``[..., 2]`` pair array stands for (float32 pairs -> complex64, float64 -> complex128)."""
    out = np.empty(pairs.shape[:-1], dtype=np.result_type(pairs.dtype, np.complex64))
    out.real, out.imag = pairs[..., 0], pairs[..., 1]
    return out


class _OnnxRunner:
    """Runs an onnxruntime session on arrays: fed by port name, the outputs by port name — the file's own names. A
    complex port (named in the model's metadata) is fed as real pairs and comes back complex; a 0-d input stays 0-d."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._inputs = [port.name for port in session.get_inputs()]
        self._outputs = [port.name for port in session.get_outputs()]
        self._complex = set(json.loads(session.get_modelmeta().custom_metadata_map.get(COMPLEX_PORTS_KEY, "[]")))

    def __call__(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feed = {
            name: _pairs(arrays[name]) if name in self._complex else np.asarray(arrays[name], order="C")
            for name in self._inputs
        }
        produced = self._session.run(None, feed)
        return {
            name: _complex(array) if name in self._complex else np.asarray(array)
            for name, array in zip(self._outputs, produced)
        }


#: The lowest opset the dynamo exporter writes (asked for 17, it wrote 18 — measured with torch 2.13).
ONNX_OPSET_FLOOR = 18

_TRANSLATING = re.compile(r"translating node %(\S+) :")


def _untranslatable(error: Exception, exported: Any, program: Program) -> Optional[ValueError]:
    """The exporter's conversion error as a LOCATED refusal — the op, whether its value was complex, the step whose
    lowering wrote it (the node's :data:`~torpaido.program.STEP_ANNOTATION`) and where that lowering is defined — or
    None when the error is something else."""
    text = str(error)
    named = _TRANSLATING.search(text)
    if named is None:
        return None
    node = next((n for n in exported.graph.nodes if n.name == named.group(1)), None)
    target = str(node.target) if node is not None else named.group(1)
    on = " on a complex value" if "complex-valued" in text else ""
    step = (node.meta.get("custom") or {}).get(STEP_ANNOTATION) if node is not None else None
    where = ""
    if step is not None:
        op_node = next((n for n in program.graph.op_nodes() if n.name == step), None)
        lowering = lowering_for(op_node.op) if op_node is not None else None
        kind = f" ({dotted(type(op_node.op))})" if op_node is not None else ""
        code = getattr(lowering, "__code__", None)
        by = (
            f" — written by {lowering.__name__} ({code.co_filename}:{code.co_firstlineno})" if lowering and code else ""
        )
        where = f" in step {step!r}{kind}{by}"
    return ValueError(f"torpaido: the onnx backend cannot translate {target}{on}{where}")


@configurable(category="compiler_backend")
class Onnx:
    """ONNX: ``model.onnx`` — the lowered program through ``torch.onnx.export(dynamo=True)``, run by onnxruntime.

    One file, the weights inline, passed through ``onnx.checker``'s full check before it is kept. The port names
    need no extra file here: the ONNX graph's own inputs and outputs are ``<entry>.<part>``, a dynamic dimension a
    named one — what every ONNX runtime reads. ONNX has no complex tensors, so a complex port is carried as REAL
    PAIRS: the lowered module is exported behind an adapter that takes a ``float32 [..., 2]`` tensor (I then Q) for
    a complex input and gives one for a complex output, the model's metadata names those ports
    (:data:`COMPLEX_PORTS_KEY`), and the runner converts both ways — the manifest keeps ``complex64``, and a flow
    sees the record's complex arrays only. A runtime feeding the file directly interleaves I and Q last. The manifest
    stays the contract: a dimension ``varies`` freed and export then fixed (a channel count a convolution needs) is
    written under the exporter's name for it, not as a size (measured: ``[s95, s85, "3"]``), and the compiled op
    checks records against ``compiled.yaml``. A stable sort (``aten.sort.stable``, a detector library's ranking) has
    no translation of the exporter's own and is written as ONNX's sort (TopK), whose tie order is the runtime's — a
    region set is compared as a set, so a tie's order never fails parity. An op the exporter cannot translate is
    refused naming the op, the lowering's line and the step (:func:`_untranslatable`). Needs the ``[onnx]`` extra
    (``onnx``, ``onnxruntime``, ``onnxscript``); without it the compile and the runner are refused naming the extra.
    The runner is a CPU session; the compiled op's ``device`` does not reach it.

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
        import torch.fx.traceback as fx_traceback
        from onnxscript.function_libs.torch_lib.ops.core import aten_sort
        from torch.export import Dim

        inputs, outputs = program.interface.inputs, program.interface.outputs
        complex_in = [port.dtype.startswith("complex") for port in inputs]
        module = program.module

        class RealPorts(torch.nn.Module):
            """The lowered module behind real pairs: a complex port is a ``[..., 2]`` real tensor at the boundary."""

            def forward(self, *reals: torch.Tensor) -> Tuple[torch.Tensor, ...]:
                tensors = [
                    torch.view_as_complex(t.contiguous()) if is_complex else t
                    for t, is_complex in zip(reals, complex_in)
                ]
                return tuple(torch.view_as_real(o) if o.is_complex() else o for o in module(*tensors))

        args = tuple(
            torch.view_as_real(t).contiguous() if is_complex else t
            for t, is_complex in zip(program.example_inputs, complex_in)
        )
        # One spec per input, nested once: the varargs tuple is the one argument export sees (program._dynamic_shapes).
        shapes = tuple(
            ({axis: Dim.AUTO for axis, size in enumerate(port.shape) if size is None} or None) for port in inputs
        )
        path = Path(out) / self.file_name
        with _onnx_export_api():
            with fx_traceback.preserve_node_meta():  # keeps each node's step (program.STEP_ANNOTATION)
                exported = torch.export.export(RealPorts().eval(), args, dynamic_shapes=(shapes,), strict=False)
            try:
                written = torch.onnx.export(
                    exported,
                    args,
                    input_names=[port.name for port in inputs],
                    output_names=[port.name for port in outputs],
                    opset_version=self.opset_version,
                    dynamo=True,
                    verbose=False,
                    custom_translation_table={torch.ops.aten.sort.stable: aten_sort},
                )
            except Exception as error:
                refusal = _untranslatable(error, exported, program)
                if refusal is None:
                    raise
                raise refusal from error
        if (
            written is None
        ):  # pragma: no cover - dynamo always returns the program; the hint is Optional for the legacy path
            raise ValueError(f"torpaido: torch.onnx.export gave no program to write to {path}")
        complex_ports = [port.name for port in (*inputs, *outputs) if port.dtype.startswith("complex")]
        written.model.metadata_props[COMPLEX_PORTS_KEY] = json.dumps(complex_ports)
        written.save(str(path), external_data=False)
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
