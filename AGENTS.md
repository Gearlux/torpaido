# Torpaido Mandates

Rules only. The reasons are in [`docs/architecture.md`](docs/architecture.md) (cited as §N).
Root `AGENTS.md` rules are not repeated here.

## Current state

The compilation engine for record pipelines and models, at its first stage: an IR and a front
end. `torpaido/ir.py` holds `Node`, `Graph`, `NodeType`, `TensorMetadata` and
`Graph.prune_sidecars`. `torpaido/frontend.py` holds `graph_from_steps` and `step_inputs`, which
turn a pipeline's step graph into the IR. Both are re-exported by `torpaido/__init__.py`. The
backend plugins (TorchScript, ONNX, TensorRT) and the `Forge` orchestrator do not exist yet;
see `TASKS.md`.

## Rules

- Compilation consumes the step GRAPH, never a flattened op list. `graph_from_steps` reads
  recordstream's parsed `FlowStep` list. `prune_sidecars` walks `node.inputs` backwards, and a
  flat list has none. Accept no op-list input and don't ask recordstream for its lowering pass
  back. A backend derives its own topological order over this IR. §1.
  (`tests/test_frontend.py::TestPruning`)
- The IR states every edge. A step with no named producer gets the previous step as its input,
  and a producer reached through several slots is ONE input, in first-seen order. §1.
  (`tests/test_frontend.py::TestLinear::test_the_implicit_previous_step_edge_is_made_explicit`,
  `::TestBranchy::test_step_inputs_strips_the_entry_and_output_suffixes_and_dedupes`)
- **Selective Pruning First:** every compilation path prunes non-inference ops (metadata
  sidecars) by reverse-dependency analysis, through `Graph.prune_sidecars`.
- **Metadata Promotion:** required metadata becomes a graph input or a constant, never a
  dictionary passed through. (Not built yet.)
- The core knows no inference engine. TorchScript, ONNX and TensorRT live in backend plugins,
  and `torch` is declared only when a backend needs it (`pyproject.toml`).
- Every compiled artifact is verified for numeric parity against its Python source, and every
  `Forge` configuration serializes through Confluid. (Neither exists yet.)
