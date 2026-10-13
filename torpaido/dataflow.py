"""Dataflow is MEASURED on sample records: which entry each node read and wrote — never trusted from declarations.

An op's ``consumes`` / ``produces`` are declarative (recordstream's own words: "nothing validates them
against the op's behavior"), and a compiler that trusts them prunes a step that was needed. So the
compiler runs each sample record through the graph on recordstream's own kernel — the ``Tracer``,
which snapshots every node's input and output record — and measures per node:

* **writes**: the entries that are new or changed in the output snapshot (the Tracer's ``changed``),
  and the entries it removed;
* **reads**: by ABLATION — the op is run again on its input snapshot with ONE entry left out; if it
  raises, drops the record, or any written entry comes out different, that entry was read. An op
  that writes nothing reads nothing (it computes nothing a value could reach).
* **passes**: the rest — entries that went through untouched, which are NOT edges.

Every sample record is measured; two records whose reads or writes differ at a node are refused
naming the node and both records — a data-dependent read has no static graph.

Each read is then resolved to the value it holds: the nearest ancestor (depth-first over the
structural ``inputs``, the ``from:`` chain first) that wrote the entry, or the source record. That
gives the value graph the lowering runs on, and narrows ``Node.inputs`` to the producers that
actually feed each node — so :meth:`~torpaido.ir.Graph.prune_sidecars` drops exactly the steps the
outputs never need.
"""

import dataclasses
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np
from recordstream import Record
from recordstream.core.families import _apply_op  # the engine's one op-application chokepoint (library ops included)
from recordstream.flow import FlowGraph, Tracer

from torpaido.frontend import SOURCE_NODE, graph_from_steps
from torpaido.ir import Graph, Node, Value

__all__ = ["measure", "same"]


@dataclasses.dataclass
class _Visit:
    """What one node did to one record: its input and output snapshots, and the entries it changed or removed."""

    input: Dict[str, Any]
    output: Dict[str, Any]
    changed: List[str]
    removed: List[str]
    reads: List[str] = dataclasses.field(default_factory=list)


def _is_tensor(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach")


def same(a: Any, b: Any) -> bool:
    """Whether two record values are the same value: arrays element for element, items field by field."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        if type(a) is not type(b) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        for attr in getattr(type(a), "_item_attrs", ()):
            if getattr(a, attr, None) != getattr(b, attr, None):
                return False
        plain_a, plain_b = np.asarray(a), np.asarray(b)
        if plain_a.dtype.kind in "fc":
            return bool(np.array_equal(plain_a, plain_b, equal_nan=True))
        return bool(np.array_equal(plain_a, plain_b))
    if _is_tensor(a) or _is_tensor(b):
        if not (_is_tensor(a) and _is_tensor(b)) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.is_floating_point() or a.is_complex():
            return bool(((a == b) | (a.isnan() & b.isnan())).all())
        return bool((a == b).all())
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        if type(a) is not type(b):
            return False
        return all(same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:  # a value with no usable equality
        return False


def _trace(steps: Sequence[Any], result: str, record: Record, ordinal: int) -> Dict[str, _Visit]:
    """One record through the graph on the Tracer: every op and fan-in node's snapshots (a subgraph container's not)."""
    tracer = Tracer(FlowGraph(flow=list(steps), outputs=result), where=f"record {ordinal}", copy_snapshots=True)
    tracer.run(deepcopy(record))
    visits: Dict[str, _Visit] = {}
    for row in tracer.to_dict()["nodes"]:
        if "inner" in row:
            continue  # a subgraph step: its inner nodes are measured, the container is an identity over them
        node = row["node"]
        if row.get("status") != "ok":
            why = row.get("error") or f"its status is {row.get('status')!r}"
            raise ValueError(
                f"torpaido: step {node!r} did not finish on record {ordinal} ({why}) — "
                "a compiled graph returns one record for every record it receives"
            )
        visits[node] = _Visit(
            input={key: tracer.value(node, key, "input") for key in row["input"]},
            output={key: tracer.value(node, key, "output") for key in row["output"]},
            changed=list(row["changed"]),
            removed=list(row["removed"]),
        )
    return visits


def _reads(op: Any, visit: _Visit) -> List[str]:
    """The entries whose absence changes what the op writes (or makes it fail) — in the input's order."""
    if op is None or not visit.changed:
        return []
    reads: List[str] = []
    for entry in visit.input:
        ablated = {key: value for key, value in visit.input.items() if key != entry}
        try:
            out = _apply_op(ablated, op)
        except Exception:
            reads.append(entry)
            continue
        if out is None or any(key not in out or not same(out[key], visit.output[key]) for key in visit.changed):
            reads.append(entry)
    return reads


def _at(node: Node) -> str:
    return f" at {node.where}" if node.where else ""


def _resolve(graph: Graph, parents: Mapping[str, List[str]], start: str, entry: str, inclusive: bool) -> Optional[str]:
    """The node whose write of ``entry`` reaches ``start``: itself when ``inclusive`` and it writes it, else the
    nearest ancestor over the STRUCTURAL ``parents`` (depth-first, the ``from:`` chain first) that writes it and
    is not followed by a removal; ``None`` when nothing does."""
    node = graph.nodes[start]
    if inclusive:
        if entry in node.writes:
            return start
        if entry in node.removes:
            return None
    seen: Set[str] = set()

    def search(name: str) -> Optional[str]:
        for parent in parents[name]:
            if parent in seen:
                continue
            seen.add(parent)
            held = graph.nodes[parent]
            if entry in held.writes:
                return parent
            if entry in held.removes:
                return None
            found = search(parent)
            if found is not None:
                return found
        return None

    return search(start)


def measure(
    steps: Sequence[Any],
    result: str,
    records: Sequence[Record],
    *,
    outputs: Optional[Sequence[str]] = None,
    name: str = "pipeline",
    locations: Optional[Mapping[str, str]] = None,
) -> Graph:
    """The value graph of a flow, measured on ``records``: what each node reads and writes, and what the graph yields.

    Args:
        steps: The parsed flow steps (``parse_flow`` / ``FlowGraph.steps``), their ops live.
        result: The step the flow yields.
        records: The sample records to measure on — at least one; every one must agree.
        outputs: The record entries the graph must yield; ``None`` = the entries written by the last writing
            node on the path to ``result``.
        name: The graph's name.
        locations: ``file:line:col`` per step, carried into refusals.

    Returns:
        The :class:`~torpaido.ir.Graph`: every op node with ``reads`` / ``writes`` / ``removes`` filled and its
        ``inputs`` narrowed to the producers of its reads, ``output_values`` set, ``outputs`` naming the nodes
        that produce them. Call :meth:`~torpaido.ir.Graph.prune_sidecars` to drop the rest.

    Raises:
        ValueError: no records; a node whose reads or writes differ between two records; a node that did
            not finish a record; an output nothing writes.
    """
    records = list(records)
    if not records:
        raise ValueError("torpaido: measuring the dataflow needs at least one sample record")
    graph = graph_from_steps(steps, result, name=name, locations=dict(locations or {}))
    source = graph.nodes[SOURCE_NODE]
    source.writes = {key: Value(f"{SOURCE_NODE}.{key}", key, type(value)) for key, value in records[0].items()}

    runs = [_trace(steps, result, record, ordinal) for ordinal, record in enumerate(records, 1)]
    for visits in runs:
        for node_name, visit in visits.items():
            visit.reads = _reads(graph.nodes[node_name].op, visit)
    first = runs[0]
    for ordinal, visits in enumerate(runs[1:], 2):
        for node_name, visit in first.items():
            other = visits[node_name]
            node = graph.nodes[node_name]
            if visit.reads != other.reads:
                raise ValueError(
                    f"torpaido: step {node_name!r} reads {sorted(visit.reads)} on record 1 and {sorted(other.reads)} "
                    f"on record {ordinal} — a data-dependent read has no static graph{_at(node)}"
                )
            if (visit.changed, visit.removed) != (other.changed, other.removed):
                raise ValueError(
                    f"torpaido: step {node_name!r} writes {sorted(visit.changed)} on record 1 and "
                    f"{sorted(other.changed)} on record {ordinal} — a data-dependent write has no static graph"
                    f"{_at(node)}"
                )

    # Writes first (every node), so a read can resolve to a producer anywhere upstream. The resolution walks
    # the STRUCTURAL edges (kept aside here): the narrowed inputs of an earlier node would cut the chain.
    structure = {name: list(node.inputs) for name, node in graph.nodes.items()}
    for node_name, visit in first.items():
        node = graph.nodes[node_name]
        node.writes = {key: Value(f"{node_name}.{key}", key, type(visit.output[key])) for key in visit.changed}
        node.removes = list(visit.removed)
    for node_name, visit in first.items():
        node = graph.nodes[node_name]
        if node.op is None:
            continue
        reads: Dict[str, Value] = {}
        producers: List[str] = []
        for entry in visit.reads:
            producer = _resolve(graph, structure, node_name, entry, inclusive=False)
            if producer is None:
                raise ValueError(
                    f"torpaido: step {node_name!r} reads {entry!r}, which no earlier step writes and the source "
                    f"record does not carry{_at(node)}"
                )
            reads[entry] = Value(f"{producer}.{entry}", entry, type(visit.input[entry]))
            if producer not in producers:
                producers.append(producer)
        node.reads = reads
        node.inputs = producers

    yielded = graph.outputs[0]
    graph.output_values = _output_values(graph, structure, yielded, outputs, first)
    graph.outputs = list(dict.fromkeys(value.name.rsplit(".", 1)[0] for value in graph.output_values))
    return graph


def _output_values(
    graph: Graph,
    structure: Mapping[str, List[str]],
    yielded: str,
    outputs: Optional[Sequence[str]],
    visits: Dict[str, _Visit],
) -> List[Value]:
    if outputs is None:
        node_name = yielded
        while not graph.nodes[node_name].writes:
            parents = structure[node_name]
            if not parents or parents[0] == SOURCE_NODE:
                raise ValueError(
                    f"torpaido: no step on the way to {yielded!r} writes an entry — nothing to compile; "
                    "name the outputs"
                )
            node_name = parents[0]
        entries: List[str] = list(graph.nodes[node_name].writes)
    else:
        entries = [str(entry) for entry in outputs]
    values: List[Value] = []
    written = sorted({key for visit in visits.values() for key in visit.changed})
    for entry in entries:
        producer = _resolve(graph, structure, yielded, entry, inclusive=True)
        if producer is None or producer == SOURCE_NODE:
            raise ValueError(f"torpaido: outputs name {entry!r}, which no step writes (the steps write: {written})")
        values.append(graph.nodes[producer].writes[entry])
    return values
