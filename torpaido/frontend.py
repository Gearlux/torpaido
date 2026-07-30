"""Front end: a record-pipeline GRAPH becomes a torpaido IR graph.

Compilation starts from a graph, so this reads one — ``recordstream``'s parsed
:class:`~recordstream.flow.FlowStep` list, where every edge is explicit:

    step.from_ / step.merge_from / step.bind   ->   Node.inputs
    step.name                                  ->   Node.outputs

That mapping is nearly an identity, which is the whole reason it belongs here rather than
downstream of a linearization pass. A *flattened* op list — the shape a sequential Python
runtime wants — re-encodes those same edges as imperative mutations of a per-record cell
store (``Save``/``Use``/``MergeFields``), which erases exactly the dependency information
:meth:`~torpaido.ir.Graph.prune_sidecars` walks: it follows ``node.inputs`` backwards from
the outputs, and a flattened list has no ``inputs``. So torpaido consumes the graph, and the
linearization it needs for codegen is its OWN topological order over its OWN IR — not
another package's runtime protocol.

Usage::

    from recordstream.flow import parse_flow
    from torpaido.frontend import graph_from_steps

    steps, outputs = parse_flow(doc["flow"], doc.get("outputs", ""))
    graph = graph_from_steps(steps, outputs, name="spectrogram")
    graph.prune_sidecars()          # drops everything the output does not depend on
"""

from typing import Any, Dict, List, Optional, Sequence

from torpaido.ir import Graph, Node, NodeType

__all__ = ["graph_from_steps", "step_inputs"]

#: The record entry a step reads when nothing names its producer — the pipeline's own input.
SOURCE_NODE = "source"


def step_inputs(step: Any) -> List[str]:
    """Every producer this step depends on, in dependency order: input, merges, then binds.

    The three flow-step slots are different KINDS of edge but the same kind of dependency, so
    the IR carries them in one ``inputs`` list. A ``bind`` reference may address a producer's
    record entry (``step[key]``) or its live ``@output`` (``step.attr``); only the producer
    NAME is a graph edge, so the suffix is stripped here.

    A producer reached through several slots (an input AND a bind, two binds on different
    entries) is ONE dependency, so the list is deduplicated in first-seen order — the IR
    records what a node depends on, not how many ways it says so.
    """
    refs: List[str] = []
    if step.from_ is not None:
        refs.append(str(step.from_))
    refs.extend(str(ref) for ref in (step.merge_from or ()))
    for ref in (step.bind or {}).values():
        head = str(ref).split("[", 1)[0].split(".", 1)[0]
        if head:
            refs.append(head)
    return list(dict.fromkeys(refs))


def graph_from_steps(
    steps: Sequence[Any],
    outputs: str = "",
    name: str = "pipeline",
    attributes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Graph:
    """Build a torpaido :class:`~torpaido.ir.Graph` from a parsed flow-step list.

    ``steps`` is what ``recordstream.flow.parse_flow`` returns (or ``FlowGraph.steps``);
    ``outputs`` names the step whose result the pipeline yields — blank means the last step,
    the same default the flow grammar uses. A step with no explicit ``from_`` reads the
    previous step, so this fills that implicit edge in: the IR must be explicit even where the
    authoring form is convenient.

    ``attributes`` optionally supplies per-step metadata (``{step name: {...}}``) to stamp on
    the emitted node — a backend's knobs, kept out of this module's vocabulary.

    Args:
        steps: Parsed flow steps, in schedule order (document order IS the schedule).
        outputs: Name of the yielded step; blank = the last step.
        name: Graph name, carried into the IR.
        attributes: Optional per-step attribute dicts to attach to the emitted nodes.

    Returns:
        A `Graph` with one INPUT node (the source record), one OP node per step, and the
        yielded step registered as the graph output.

    Raises:
        ValueError: ``outputs`` does not name a step, or a step references an unknown one.
    """
    graph = Graph(name)
    graph.add_node(Node(name=SOURCE_NODE, node_type=NodeType.INPUT))

    known = {SOURCE_NODE}
    previous: Optional[str] = None
    attributes = attributes or {}

    for step in steps:
        step_name = str(step.name)
        inputs = step_inputs(step)
        if not inputs:
            # No named producer: the step reads whatever ran before it (the source, first).
            inputs = [previous or SOURCE_NODE]
        unknown = [ref for ref in inputs if ref not in known]
        if unknown:
            raise ValueError(
                f"torpaido.frontend: step {step_name!r} references unknown step(s) {unknown} — "
                "a flow step may only read an EARLIER step (document order is the schedule)."
            )
        graph.add_node(
            Node(
                name=step_name,
                node_type=NodeType.OP,
                op_type=type(step.op).__name__ if step.op is not None else None,
                inputs=inputs,
                outputs=[step_name],
                attributes=dict(attributes.get(step_name, {})),
            )
        )
        known.add(step_name)
        previous = step_name

    yielded = str(outputs) if outputs else (previous or SOURCE_NODE)
    if yielded not in known:
        raise ValueError(f"torpaido.frontend: outputs {yielded!r} does not name a step ({sorted(known)}).")
    graph.outputs.append(yielded)
    return graph
