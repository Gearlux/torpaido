"""``CompiledPackage`` — the op that runs a compiled folder on a record: the twin of the package op, same slot.

The folder is read on first use (never in the constructor) and again when ``path`` changes. The record's
input entries are split into arrays by their item rules, checked against the manifest (dtype, rank, the
fixed sizes, the specialized values), run through the backend's runner, and the output arrays joined back
into items — the same rules the compiler split them with, so the two faces cannot disagree.
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from confluid import configurable
from loggair import get_logger
from recordstream import Record

from torpaido.lowering.items import item_rule
from torpaido.lowering.registry import discover
from torpaido.manifest import COMPILED_FORMAT, read_compiled
from torpaido.program import Interface, Port, interface_arrays, resolve_dotted

logger = get_logger(__name__)

__all__ = ["CompiledPackage"]


def _expand(path: str) -> Path:
    return Path(os.path.normpath(os.path.abspath(os.path.expanduser(os.path.expandvars(path)))))


@configurable(category="op", group="model")
class CompiledPackage:
    """Run a compiled package on a record — one op wherever a flow graph or a stream takes one.

    Args:
        path: The compiled folder (``$VAR`` and ``~`` expanded) — ``<package>/compiled/<name>``, or wherever
            it was moved to.
        device: The device the engine runs on (``"cpu"``; the torch backends map the artifact there).

    Example::

        flow:
          model:
            op: !class:torpaido.CompiledPackage
              path: $PROJECT_ROOT/runs/a/package/compiled/torchscript
    """

    def __init__(self, path: str = "", device: str = "cpu") -> None:
        self.path = path
        self.device = device
        self._runner: Any = None  # the backend's runner: any object taking arrays by port name
        self._interface: Optional[Interface] = None
        self._loaded_with: Optional[str] = None

    def _ensure(self) -> Tuple[Any, Interface]:
        """The runner and the interface, loaded on first use and again after ``path`` changes."""
        if self._runner is not None and self._interface is not None and self._loaded_with == self.path:
            return self._runner, self._interface
        if not self.path:
            raise ValueError("CompiledPackage: 'path' is blank — name the compiled package folder")
        try:
            info = read_compiled(self.path)
        except ValueError as refusal:
            raise ValueError(f"CompiledPackage: {refusal}") from refusal
        root = _expand(self.path)
        if info.get("format") != COMPILED_FORMAT:
            raise ValueError(
                f"CompiledPackage: {root} is a compiled package of format {info.get('format')}, "
                f"this torpaido reads format {COMPILED_FORMAT}"
            )
        from torpaido.backends import backend_named

        discover()  # a domain package's item rules register on import
        backend = backend_named(str(info["backend"]))()
        self._runner = backend.runner(root, [entry["path"] for entry in info.get("files") or []])
        self._interface = Interface(
            inputs=[Port.from_dict(name, port) for name, port in (info.get("inputs") or {}).items()],
            outputs=[Port.from_dict(name, port) for name, port in (info.get("outputs") or {}).items()],
            specialized=dict(info.get("specialized") or {}),
            constants=dict(info.get("constants") or {}),
            envelope=dict(info.get("envelope") or {}),
            consumes=dict(info.get("consumes") or {}),
            produces=dict(info.get("produces") or {}),
        )
        self._loaded_with = self.path
        logger.debug(f"CompiledPackage: loaded {root} ({info['backend']})")
        return self._runner, self._interface

    def __call__(self, record: Record) -> Record:
        """``record`` with the artifact's output entries added (the rest passes through untouched)."""
        runner, interface = self._ensure()
        root = _expand(self.path)
        try:
            arrays = interface_arrays(interface, record)
        except ValueError as refused:
            message = str(refused)
            if message.startswith("was compiled for"):
                part = message.split(" ", 3)[3].split(" =", 1)[0]
                raise ValueError(
                    f"CompiledPackage: {root} {message} — compile with the envelope letting it vary ({part}: varies) "
                    "to run such records"
                ) from refused
            raise ValueError(f"CompiledPackage: {root}: {message}") from refused
        outputs = runner(arrays)
        produced: Dict[str, Any] = {}
        for entry, item_name in interface.produces.items():
            parts = {port.part: outputs[port.name] for port in interface.outputs if port.entry == entry}
            facts = {
                key.split(".", 1)[1]: copy.deepcopy(value)
                for key, value in interface.constants.items()
                if key.startswith(entry + ".")
            }
            produced[entry] = item_rule(resolve_dotted(item_name)).join(parts, facts)
        return {**record, **produced}

    @property
    def consumes(self) -> Tuple[type, ...]:
        """The item types of the entries the artifact takes (read off the manifest)."""
        _, interface = self._ensure()
        return tuple(resolve_dotted(name) for name in interface.consumes.values())

    @property
    def produces(self) -> Tuple[type, ...]:
        """The item types of the entries the artifact adds."""
        _, interface = self._ensure()
        return tuple(resolve_dotted(name) for name in interface.produces.values())

    def __getstate__(self) -> Dict[str, Any]:
        """Pickled (a worker process) without the engine: the worker reads the folder itself."""
        state = dict(self.__dict__)
        state["_runner"] = None
        state["_interface"] = None
        state["_loaded_with"] = None
        return state
