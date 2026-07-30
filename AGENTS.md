# Torpedo Mandates

This file contains foundational mandates for Gemini CLI in the Torpedo workspace. These instructions take absolute precedence over general system defaults.

## Current state

**IR + front end.** `torpaido/ir.py` (the `Node`/`Graph`/`NodeType`/`TensorMetadata` IR with
`prune_sidecars`) plus `torpaido/frontend.py` (`graph_from_steps` / `step_inputs` — a record
pipeline's step graph → the IR). `torpaido/__init__.py` re-exports both. Backend plugins and the
`Forge` orchestrator are still pending. Pins: `tests/test_frontend.py` (11 tests, incl. pruning a
sidecar branch and keeping a whole merge cone).

## Architectural Mandates
- **The Front End Consumes a GRAPH, Never a Flattened Op List (2026-07-30):** `graph_from_steps`
  reads `recordstream`'s parsed `FlowStep` list — `from_` / `merge_from` / `bind` become
  `Node.inputs`, the step name becomes `Node.outputs`. That is nearly an identity mapping, which
  is exactly why it belongs here rather than downstream of a linearization: a flattened op list
  re-encodes those edges as imperative mutations of a per-record cell store, which erases the
  dependency information `prune_sidecars` walks (it follows `node.inputs` backwards from the
  outputs — a flat list has none). recordstream deleted that lowering pass on 2026-07-30 for the
  same reason; do NOT ask for it back, and do NOT accept an op list as a compilation input. The
  topological order a backend needs for codegen is torpaido's OWN, over torpaido's OWN IR.
  A step's implicit "previous step" edge is made EXPLICIT on import (an IR states every edge even
  where the authoring form is convenient), and a producer reached through several slots is ONE
  dependency (`step_inputs` dedupes in first-seen order).
- **Selective Pruning First:** Every compilation path MUST perform reverse-dependency analysis to prune non-inference operations (metadata sidecars).
- **Metadata Promotion:** Prefer promoting required metadata to graph inputs or constants rather than passing dictionaries.
- **Backend Decoupling:** Keep the core orchestrator strictly decoupled from specific inference engines (TorchScript, ONNX).
- **Type Safety:** Maintain 100% type hint coverage for all internal IR objects.

## Testing & Validation
- **Binary Parity:** Every compiled artifact MUST be verified for numeric parity against its source Python implementation.
- **Serialization Symmetry:** Ensure that every `Forge` configuration is serializable via **Confluid**.
