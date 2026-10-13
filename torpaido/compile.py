"""``CompilePackage`` — the runnable that compiles a model package into a standalone compiled folder.

Run as a config's ``runnable:`` (``recordstream run compile.yaml``). The five passes, in order: the
package is loaded through its own loader; its steps become the IR; the dataflow is measured on the
sample records and pruned to the outputs; every surviving node is lowered and the program exported;
the backend writes its files, and the artifact is run on every sample record against the Python graph.
Only when parity holds is the folder moved into place — a failed compile leaves ``<name>.partial/``
with ``parity.yaml`` naming the first node whose lowering disagrees with its op.

The compiled folder stands on its own: the manifest, the backend's files, a copy of the package's
``class_names.json``, the parity report. Nothing in it reads the package again.
"""

import copy
import os
import shutil
from datetime import datetime, timezone
from importlib import metadata
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import confluid
from confluid import configurable, format_yaml_loc
from loggair import get_logger
from matrainer.artifacts import CLASS_NAMES_FILE
from matrainer.model_package import PACKAGE_FILE, LoadedPackage, load_package_graph
from matrainer.tracking.provenance import command_line, file_digest
from recordstream import Record
from recordstream.flow import FlowGraph, Tracer, parse_flow, run_steps_multi

from torpaido.dataflow import measure
from torpaido.lowering.items import item_rule
from torpaido.manifest import COMPILED_FILE, COMPILED_FORMAT, PARITY_FILE, write_compiled
from torpaido.parity import DEFAULT_TOLERANCES, Difference, compare
from torpaido.program import Program, build_program, interface_arrays

logger = get_logger(__name__)

__all__ = ["CompilePackage"]

#: Distributions every compiled folder names: the ones that read or run it.
_LOADERS = ("torpaido", "torch", "recordstream", "matrainer", "confluid", "numpy")


def _expand(path: str) -> Path:
    return Path(os.path.normpath(os.path.abspath(os.path.expanduser(os.path.expandvars(path)))))


def _locations(request: Mapping[str, Any]) -> Dict[str, str]:
    """``{step: file:line:col}`` of the package graph's steps, read off its markers at the document stage."""
    try:
        document = confluid.load(copy.deepcopy(dict(request)), until="document")
    except Exception as refused:  # the built graph already loaded: a location is a nicety, never a second refusal
        logger.debug(f"CompilePackage: no step locations ({refused})")
        return {}
    flow = document.get("flow") if isinstance(document, dict) else None
    if not isinstance(flow, dict):
        return {}
    where: Dict[str, str] = {}
    for name, value in flow.items():
        marker = value.get("op") if isinstance(value, dict) else value
        location = format_yaml_loc(marker)
        if location:
            where[str(name)] = location
    return where


def _versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in _LOADERS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            logger.debug(f"CompilePackage: no installed distribution named {name!r}")
    return versions


@configurable(category="runnable", group="model")
class CompilePackage:
    """Compile a model package into a standalone compiled folder, for one backend.

    An ``output_dir`` (default ``<package>/compiled/<name>``) that already holds a compiled package is
    replaced; one holding anything else is refused untouched. The new folder is written as
    ``<output_dir>.partial`` and moved into place only when parity holds.

    Args:
        package: The model package folder (``$VAR`` and ``~`` expanded).
        name: The compiled folder's name under ``<package>/compiled/``; also how a consumer tells the forms apart.
        backend: The backend that writes the artifact (``!class:torpaido.backends.TorchScript {}``);
            ``None`` = TorchScript.
        overrides: Dotted keys into the package's graph and their values, applied after it and its weight slots —
            the same setting the package op takes (``{flow.detect.op.device: cpu, flow.detect.op.masks_output: ""}``).
            They are part of what was compiled and are written in the manifest.
        outputs: The record entries the artifact must deliver; ``None`` = what the last writing step writes.
        envelope: ``{"<entry>.<part>": "varies" | value}`` — which input parts vary (a dynamic dimension, a
            scalar input) and which are fixed to a value. What it does not name is fixed at the samples' value.
        samples: A source of records (anything iterable) the compiler measures on and verifies against.
        sample_count: How many records to take from ``samples``.
        output_dir: Where to write; blank = ``<package>/compiled/<name>``.
        tolerances: ``{atol, rtol}`` defaults and per-entry overrides (``{"scores": {"atol": 1e-3}}``) for parity.
        description: One line saying what this compiled form is, for a reader of the manifest.

    Example::

        runnable: !class:torpaido.CompilePackage
          package: ./runs/a/package
          name: torchscript
          backend: !class:torpaido.backends.TorchScript {}
          outputs: [predictions]
          envelope: {signal.samplerate: 100e6, signal.data: varies}
          samples: !class:recordstream.sources.range.RangeSource {source: ..., stop: 8}
    """

    def __init__(
        self,
        package: str = "",
        name: str = "torchscript",
        backend: Any = None,
        overrides: Optional[Dict[str, Any]] = None,
        outputs: Optional[List[str]] = None,
        envelope: Optional[Dict[str, Any]] = None,
        samples: Any = None,
        sample_count: int = 4,
        output_dir: str = "",
        tolerances: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> None:
        self.package = package
        self.name = name
        # `Any`: a backend is any object meeting the Backend protocol; a sample source any iterable of records.
        self.backend = backend
        # `Any` values: an override replaces any setting of any class the package's graph holds.
        self.overrides: Dict[str, Any] = dict(overrides) if overrides else {}
        self.outputs = list(outputs) if outputs else None
        self.envelope = dict(envelope) if envelope else None
        self.samples = samples
        self.sample_count = int(sample_count)
        self.output_dir = output_dir
        self.tolerances = dict(tolerances) if tolerances else None
        self.description = description

    def run(self) -> str:
        """Compile; returns the compiled folder (the path a :class:`~torpaido.compiled.CompiledPackage` takes)."""
        if not self.package:
            raise ValueError("CompilePackage: 'package' is blank — name the package folder")
        if not self.name:
            raise ValueError("CompilePackage: 'name' is blank — name the compiled folder")
        if self.samples is None:
            raise ValueError("CompilePackage: 'samples' is not set — name a source of records")
        loaded = load_package_graph(self.package, self.overrides, owner="CompilePackage")
        records = [copy.deepcopy(record) for record in islice(iter(self.samples), self.sample_count)]
        if not records:
            raise ValueError("CompilePackage: 'samples' yielded no record")
        backend = self.backend if self.backend is not None else _default_backend()
        out = _expand(self.output_dir) if self.output_dir else loaded.root / "compiled" / self.name
        if out.exists() and any(out.iterdir()) and not (out / COMPILED_FILE).is_file():
            raise ValueError(
                f"CompilePackage: {out} holds files and is not a compiled package (it has no {COMPILED_FILE}) — "
                "name an empty or new folder"
            )

        steps, result = parse_flow(loaded.document["flow"], loaded.result)
        graph = measure(
            steps, result, records, outputs=self.outputs, name=loaded.root.name, locations=_locations(loaded.request)
        )
        graph.prune_sidecars()
        program = build_program(graph, records, envelope=self.envelope)

        staging = out.with_name(out.name + ".partial")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        files = backend.compile(program, staging)
        tolerances = {**DEFAULT_TOLERANCES, **(self.tolerances or {})}
        rows, failure = self._parity(
            program, backend, staging, [path.name for path in files], steps, result, records, tolerances
        )
        if failure is not None:
            ordinal, difference, node, at_node = failure
            _write_parity(
                staging,
                "failed",
                rows,
                tolerances,
                first_disagreeing_node=node,
                first_node_differences=[d.as_dict() for d in at_node],
            )
            there = f" ({at_node[0].entry}.{at_node[0].part} {at_node[0].measure})" if at_node else ""
            raise ValueError(
                f"CompilePackage: parity failed on record {ordinal}: {difference.entry}.{difference.part} "
                f"{difference.measure}; the first node that disagrees is {node!r}{there} — "
                f"the report is {staging / PARITY_FILE}"
            )
        _write_parity(staging, "passed", rows, tolerances)
        class_names = loaded.root / CLASS_NAMES_FILE
        if class_names.is_file():
            shutil.copy2(class_names, staging / CLASS_NAMES_FILE)
        write_compiled(
            staging, self._manifest(loaded, backend, program, files, staging, len(records), class_names.is_file())
        )
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
        logger.info(
            f"CompilePackage: wrote {out} ({backend.name}; {len(graph.pruned)} steps pruned; "
            f"parity on {len(records)} records)"
        )
        return str(out)

    def _parity(
        self,
        program: Program,
        backend: Any,
        folder: Path,
        files: List[str],
        steps: Sequence[Any],
        result: str,
        records: List[Record],
        tolerances: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Optional[Tuple[int, Difference, str, List[Difference]]]]:
        """Every record through the Python graph and the artifact; the rows, and the first failure with its node."""
        runner = backend.runner(folder, files)
        entries = [port.entry for port in program.interface.outputs]
        entries = list(dict.fromkeys(entries + list(program.interface.produces)))
        rows: List[Dict[str, Any]] = []
        failure: Optional[Tuple[int, Difference, str, List[Difference]]] = None
        for ordinal, record in enumerate(records, 1):
            reference = run_steps_multi(copy.deepcopy(record), list(steps), result)[0]
            produced = _join_outputs(program, runner(interface_arrays(program.interface, copy.deepcopy(record))))
            differences = compare(reference, produced, entries, tolerances)
            rows.append({"record": ordinal, "differences": [d.as_dict() for d in differences]})
            if differences and failure is None:
                node, at_node = _bisect(program, steps, result, record, tolerances)
                failure = (ordinal, differences[0], node, at_node)
        return rows, failure

    def _manifest(
        self,
        loaded: LoadedPackage,
        backend: Any,
        program: Program,
        files: List[Path],
        folder: Path,
        count: int,
        class_names: bool,
    ) -> Dict[str, Any]:
        interface = program.interface
        return {
            "format": COMPILED_FORMAT,
            "backend": backend.name,
            "description": self.description,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": command_line(),
            "source": {
                "package": self.package,
                "package_sha256": file_digest(loaded.root / PACKAGE_FILE),
                "graph": loaded.info.get("graph"),
            },
            "files": [{"path": path.name, "sha256": file_digest(path)} for path in files],
            "class_names": CLASS_NAMES_FILE if class_names else None,
            "overrides": dict(self.overrides),
            "consumes": dict(interface.consumes),
            "produces": dict(interface.produces),
            "inputs": {port.name: port.as_dict() for port in interface.inputs},
            "outputs": {port.name: port.as_dict() for port in interface.outputs},
            "specialized": dict(interface.specialized),
            "constants": dict(interface.constants),
            "envelope": dict(interface.envelope),
            "pruned": list(program.graph.pruned),
            "parity": {"records": count, "status": "passed", "report": PARITY_FILE},
            "packages": _versions(),
        }


def _default_backend() -> Any:
    from torpaido.backends import TorchScript

    return TorchScript()


def _join_outputs(program: Program, arrays: Dict[str, Any]) -> Record:
    """The artifact's output arrays joined back into record items, entry by entry."""
    record: Record = {}
    interface = program.interface
    for entry, item_name in interface.produces.items():
        parts = {port.part: arrays[port.name] for port in interface.outputs if port.entry == entry}
        facts = {
            key.split(".", 1)[1]: value for key, value in interface.constants.items() if key.startswith(entry + ".")
        }
        record[entry] = item_rule(program.graph.output_values[list(interface.produces).index(entry)].item).join(
            parts, facts
        )
    return record


def _bisect(
    program: Program, steps: Sequence[Any], result: str, record: Record, tolerances: Dict[str, Any]
) -> Tuple[str, List[Difference]]:
    """The first node, in schedule order, whose eager torch outputs disagree with its Python op's on ``record``,
    judged by the same tolerances (an entry's own tolerance applies wherever that entry is written), with that
    node's measured differences — what the parity report shows so the reader does not re-derive the gap."""
    tracer = Tracer(FlowGraph(flow=list(steps), outputs=result), where="parity", copy_snapshots=True)
    tracer.run(copy.deepcopy(record))
    eager = program.eager(copy.deepcopy(record))
    for node in program.graph.op_nodes():
        reference = {entry: tracer.value(node.name, entry, "output") for entry in node.writes}
        differences = compare(reference, eager.get(node.name, {}), list(node.writes), tolerances)
        if differences:
            return node.name, differences
    nodes = program.graph.op_nodes()
    return (nodes[-1].name if nodes else ""), []


def _write_parity(
    folder: Path, status: str, rows: List[Dict[str, Any]], tolerances: Dict[str, Any], **extra: Any
) -> None:
    import yaml

    report: Dict[str, Any] = {"status": status, "tolerances": tolerances, **extra, "records": rows}
    (folder / PARITY_FILE).write_text(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))


def _unused(records: Iterable[Record]) -> None:  # pragma: no cover - keeps the Iterable import meaningful to readers
    list(records)
