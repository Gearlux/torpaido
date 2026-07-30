# Torpaido architecture

Decision records for the compilation engine. Each answers "why does this module exist?" for a
reader who can already see what it does.

---

## 1. The front end consumes a graph, not a flattened op list (`torpaido.frontend`, 2026-07-30)

### Context

Torpaido compiles record pipelines (and models) into inference artifacts. A pipeline reaches it
in one of two shapes, and until 2026-07-30 the producing package could emit either:

- a **step graph** — named steps with explicit edges (`from:` names the producer, `merge_from:`
  the merge sources, `bind:` a cross-step value);
- a **flat op list** — the same pipeline linearized, with the branch structure re-encoded as
  imperative mutations of a per-record cell store (`Save` a fork, `Use` to restart a branch,
  `MergeFields` to rejoin).

The flat form looks like the more "compiled" one — it is a straight sequence, closer to what a
backend eventually emits — which makes it a tempting input. It is the wrong one.

Every mandate in this project's `AGENTS.md` starts from dependencies. **Selective Pruning First**
requires reverse-dependency analysis; `Graph.prune_sidecars` implements it by walking
`node.inputs` backwards from `self.outputs`. **Metadata Promotion** requires knowing which values
a node actually consumes. **Unpacked Handover** requires knowing which producer feeds which
consumer so a boxed record can be replaced with direct tensor-to-tensor edges.

A flat op list carries none of that. Its ops have no `inputs`: the edges live in the *side
effects* of cell ops, and recovering them means re-deriving the graph — i.e. running the lifting
pass, to rebuild exactly what the lowering pass just destroyed.

### Decision

`graph_from_steps(steps, outputs)` reads a **parsed step list** and emits a `Graph`:

| step | node |
| --- | --- |
| `step.from_`, `step.merge_from`, `step.bind` | `Node.inputs` (deduplicated, first-seen order) |
| `step.name` | `Node.outputs`, and the node's own name |
| `type(step.op).__name__` | `Node.op_type` |

Two normalizations happen at the boundary, both because an IR must be explicit where an authoring
form may be convenient:

- a step with no named producer gets the **previous step** (or the source) as an explicit input;
- a `bind` reference's suffix is stripped — `producer[key]` and `producer.attr` address an *entry*
  or an `@output` of one producer, and only the producer is an edge.

The source record is a single `INPUT` node (`SOURCE_NODE`), so every op node has at least one
input and the graph has one root.

### Consequences

- `prune_sidecars` runs on an imported pipeline — it could not run on a lowered one at all.
- The producing package was able to delete its lowering pass entirely (see recordstream's
  architecture record §3), because the compiler was the last plausible consumer of the flat form
  and it never wanted it.
- Torpaido owns its own linearization. A backend emitting sequential code derives a topological
  order over *this* IR, where the edges are still present.
- A pipeline that cannot be expressed as a graph cannot be compiled — which is not a limitation,
  since the graph is the more general form.

### Example

```python
from recordstream.flow import parse_flow
from torpaido import graph_from_steps

flow = {
    "start": {},
    "left":  {"op": Threshold(low_level=0.5), "from": "start"},
    "right": {"op": Boost(), "from": "start"},
    "dead":  {"op": Debug(), "from": "start"},          # a sidecar: nothing depends on it
    "out":   {"from": "left", "merge_from": ["right"]},
}
graph = graph_from_steps(*parse_flow(flow, "out"))

graph.nodes["out"].inputs        # ['left', 'right'] — the fan-in, explicit
graph.prune_sidecars()
sorted(graph.nodes)              # ['left', 'out', 'right', 'source', 'start'] — 'dead' is gone
```

### What you may change (and where it's documented)

- **Adding an edge kind** (a new step-grammar slot upstream) means teaching `step_inputs` about
  it. If it is a real dependency it belongs in `Node.inputs`; if it is metadata it belongs in
  `Node.attributes`. Getting that wrong silently breaks pruning.
- **`Node.attributes`** is the sanctioned place for backend knobs, supplied per step via the
  `attributes=` argument — keep backend vocabulary out of the front end itself.
- **Do NOT add an op-list front end.** The reasoning above is the whole record; if a future
  producer only emits a flat list, the fix is upstream.
