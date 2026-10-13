"""Front end: a record-pipeline GRAPH becomes a torpaido IR graph.

Compilation starts from a graph, so this reads one — ``recordstream``'s parsed
:class:`~recordstream.flow.steps.FlowStep` list, where every edge is explicit:

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

A step whose op is a subgraph is INLINED: its inner steps become ``<step>/<inner>`` nodes — the
names recordstream's Tracer gives them, so a measurement and a refusal name the same node — and
the step itself stays as an identity node over the inner result, which later steps read through.

Usage::

    from recordstream.flow import parse_flow
    from torpaido.frontend import graph_from_steps

    steps, outputs = parse_flow(doc["flow"], doc.get("outputs", ""))
    graph = graph_from_steps(steps, outputs, name="spectrogram")
    graph.prune_sidecars()          # drops everything the output does not depend on
"""

from typing import Any, Dict, List, Optional, Sequence, Set

from torpaido.ir import SOURCE_NODE, Graph, Node, NodeType

__all__ = ["SOURCE_NODE", "graph_from_steps", "step_inputs", "is_subgraph"]

#: The separator between a subgraph step and its inner steps, as recordstream's Tracer names them.
INNER = "/"


def is_subgraph(op: Any) -> bool:
    """True for an op whose body is a flow — read off the CLASS marker recordstream's own duck-typing uses."""
    return getattr(type(op), "FLOW_SUBGRAPH", False) is True


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
    locations: Optional[Dict[str, str]] = None,
) -> Graph:
    """Build a torpaido :class:`~torpaido.ir.Graph` from a parsed flow-step list.

    ``steps`` is what ``recordstream.flow.parse.parse_flow`` returns (or ``FlowGraph.steps``);
    ``outputs`` names the step whose result the pipeline yields — blank means the last step,
    the same default the flow grammar uses. A step with no explicit ``from_`` reads the
    previous step, so this fills that implicit edge in: the IR must be explicit even where the
    authoring form is convenient.

    ``attributes`` optionally supplies per-step metadata (``{step name: {...}}``) to stamp on
    the emitted node — a backend's knobs, kept out of this module's vocabulary. ``locations``
    gives each step's ``file:line:col`` (``{step name: where}``); an inner node of a subgraph
    inherits its step's location, suffixed with its own name, because a built subgraph keeps no
    trace of where its inner steps were written.

    Args:
        steps: Parsed flow steps, in schedule order (document order IS the schedule).
        outputs: Name of the yielded step; blank = the last step.
        name: Graph name, carried into the IR.
        attributes: Optional per-step attribute dicts to attach to the emitted nodes.
        locations: Optional ``file:line:col`` per step, carried into every refusal that names the node.

    Returns:
        A `Graph` with one INPUT node (the source record), one OP node per step (a subgraph step's
        inner steps as ``<step>/<inner>`` nodes, the step itself an identity over its result), and
        the yielded step registered as the graph output.

    Raises:
        ValueError: ``outputs`` does not name a step, or a step references an unknown one.
    """
    graph = Graph(name)
    graph.add_node(Node(name=SOURCE_NODE, node_type=NodeType.INPUT))
    known: Set[str] = {SOURCE_NODE}
    last = _add_steps(graph, steps, "", [SOURCE_NODE], known, attributes or {}, locations or {}, "")

    yielded = str(outputs) if outputs else (last or SOURCE_NODE)
    if yielded not in known:
        raise ValueError(f"torpaido.frontend: outputs {yielded!r} does not name a step ({sorted(known)}).")
    graph.outputs.append(yielded)
    return graph


def _add_steps(
    graph: Graph,
    steps: Sequence[Any],
    prefix: str,
    entry_inputs: List[str],
    known: Set[str],
    attributes: Dict[str, Dict[str, Any]],
    locations: Dict[str, str],
    inherited_where: str,
) -> Optional[str]:
    """Add one step list (a flow, or a subgraph's inside under ``prefix``); returns the last node added.

    Inside a subgraph a step with no named producer reads the record the subgraph received
    (``entry_inputs``) when it is the first inner step, else the inner step before it — the same
    rule the outer flow follows with the source record.
    """
    previous: Optional[str] = None
    for step in steps:
        step_name = prefix + str(step.name)
        refs = step_inputs(step)
        inputs = [prefix + ref for ref in refs] if refs else ([previous] if previous else list(entry_inputs))
        unknown = [ref for ref in inputs if ref not in known]
        if unknown:
            raise ValueError(
                f"torpaido.frontend: step {step_name!r} references unknown step(s) {unknown} — "
                "a flow step may only read an EARLIER step (document order is the schedule)."
            )
        where = locations.get(step_name, "") or (f"{inherited_where} (step {step_name})" if inherited_where else "")
        if step.op is not None and is_subgraph(step.op):
            # The inside becomes `<step>/<inner>` nodes reading what the step reads; the step itself is
            # an identity over the inner result, so a later `from: <step>` still resolves.
            inner_last = _add_steps(
                graph, step.op.flow_steps, step_name + INNER, inputs, known, attributes, locations, where
            )
            result = step_name + INNER + str(step.op.output_step)
            graph.add_node(
                Node(
                    name=step_name,
                    node_type=NodeType.OP,
                    op_type=type(step.op).__name__,
                    inputs=[result if result in known else str(inner_last)],
                    outputs=[step_name],
                    attributes=dict(attributes.get(step_name, {})),
                    where=where,
                )
            )
        else:
            graph.add_node(
                Node(
                    name=step_name,
                    node_type=NodeType.OP,
                    op_type=type(step.op).__name__ if step.op is not None else None,
                    inputs=inputs,
                    outputs=[step_name],
                    attributes=dict(attributes.get(step_name, {})),
                    op=step.op,
                    where=where,
                )
            )
        known.add(step_name)
        previous = step_name
    return previous
