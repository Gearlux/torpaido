# Torpaido Mandates

Rules only. The reasons are in [`docs/architecture.md`](docs/architecture.md) (cited as §N).
Root `AGENTS.md` rules are not repeated here.

## Current state

The compiler of model packages (matrainer's `package.yaml` + graphs + weights + `class_names.json`).
One config run, `runnable: !class:torpaido.CompilePackage`, writes a STANDALONE compiled folder
(`compiled.yaml`, the backend's files, a copy of `class_names.json`, `parity.yaml`), default
`<package>/compiled/<name>/`; `!class:torpaido.CompiledPackage {path: <compiled folder>}` runs it as
one op. Five passes: `matrainer.model_package.load_package_graph` → `frontend.graph_from_steps`
(subgraphs inlined) → `dataflow.measure` (reads/writes measured by ablation on the sample records,
then `Graph.prune_sidecars`) → `program.build_program` (one `torch.export` program from the
registered lowerings) → a backend (`backends.TorchScript`, `backends.TorchExport`) plus
`parity.compare` with a per-node bisect. Torpaido ships recordstream's op lowerings
(`lowering/recordstream.py`); a domain package ships its own behind a `[compile]` extra and the
entry-point group `torpaido.lowerings`. Not built: ONNX and the other backends, any domain lowering,
runtime branches, a draw's refusal by name — `TASKS.md`.

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
- A subgraph step is INLINED as `<step>/<inner>` nodes — the Tracer's names, so a measurement and a
  refusal name the same node — and the step itself stays an identity node over its inner result. An
  inner node inherits its step's location, suffixed `(step <name>)`. §3.
  (`tests/test_frontend.py::TestSubgraphs`)
- **Dataflow is MEASURED, never declared.** `measure` runs every sample record on recordstream's
  `Tracer`; a node's writes are the Tracer's `changed` (+ `removed`), its reads are found by
  ABLATION (the op rerun with one entry left out: raising, dropping, or a changed write = read). An
  op that writes nothing reads nothing. Reads and writes must agree across the samples or the compile
  is refused naming the node and both records. `consumes` / `produces` are never consulted. §2.
  (`tests/test_dataflow.py::test_each_node_reads_what_its_op_used_and_writes_what_it_changed`,
  `::test_a_read_that_depends_on_the_record_is_refused_naming_the_node_and_both_records`)
- A read resolves to the nearest ancestor over the STRUCTURAL edges (the `from:` chain first) that
  wrote the entry; a branch reads its own producer, not the latest write. Only after every read is
  resolved are `Node.inputs` narrowed to the producers. §2.
  (`tests/test_dataflow.py::test_a_branch_reads_its_own_producer_not_the_latest_write`)
- **Selective Pruning First:** every compilation path prunes through `Graph.prune_sidecars` (it
  returns the op nodes dropped, kept as `Graph.pruned` for the manifest). The outputs default to
  the entries the last WRITING node on the path to the yielded step wrote; `outputs:` chooses
  otherwise, and an entry nothing writes is refused listing what is written. `Graph.input_values`
  is DERIVED from the surviving nodes' reads. §2.
  (`tests/test_dataflow.py::test_pruning_drops_everything_the_outputs_never_need`,
  `::test_the_outputs_may_be_chosen`, `::test_an_output_nothing_writes_is_refused_naming_what_is_written`)
- **A lowering is ONE function per op class**, `(op, reads, ctx) -> writes`, registered with
  `@lowers(OpClass)` by EXACT class (a subclass inherits nothing), found by `lowering_for`, discovered
  by `discover()` (torpaido's own module, then the entry-point group `torpaido.lowerings` through
  confluid's `load_configurables`). A class has exactly one. §4. **`discover()` keeps what the group gave**
  (`DISCOVERED`: entry name → module, or the exception its import raised), and a held object without a
  model lowering is refused through `no_model_lowering(held, where)`: the modules found, the entries that
  failed to import, and — when none comes from the package defining the object — the reinstall (2026-10-13:
  a package installed before it declared its entry point looked like a lowering never written). A domain's
  refusal for a held model calls it; never write the bare "has no model lowering" line.
  (`tests/test_lowering.py::test_a_lowering_is_found_by_the_exact_op_class`,
  `::test_registering_a_second_lowering_for_a_class_is_refused`,
  `::test_discover_imports_the_entry_point_group_and_torpaido_s_own_lowerings`,
  `::test_a_missing_model_lowering_s_refusal_says_what_was_found_and_what_to_do`)
- A surviving node without a lowering is refused at its `file:line:col`, naming what it writes and
  how to register one; a pruned node needs none. A lowering's writes must equal the op's measured
  writes. A lowering needing a constant calls `ctx.constant`, which refuses a varying value naming
  the input it depends on. §4.
  (`tests/test_lowering.py::test_a_surviving_node_without_a_lowering_is_refused_at_its_location`,
  `::test_a_pruned_node_without_a_lowering_is_not_refused`,
  `::test_a_lowering_that_writes_other_entries_than_its_op_is_refused`,
  `::test_a_varying_value_where_a_constant_is_needed_is_refused_naming_the_dependency`)
- **An object an op HOLDS lowers through its own registry**: `@lowers_model(Class)` / `model_lowering_for(obj)`,
  exact class, one per class, found by the same `discover()`. The OP's lowering states the tensor contract its held
  object must meet; torpaido knows no contract. (`tests/test_lowering.py::test_a_model_lowering_is_found_by_the_held_object_s_exact_class`)
- `lower_convert_to_image` (`lowering/recordstream.py`) renders the `gray` colormap of a 2-D map only — quantized by
  truncation as the op does, resized by a bilinear filter WITH antialiasing (PIL's BILINEAR within 1 of 255) — and
  refuses another colormap or a 3-D array by name. A parity tolerance on an integer array is an absolute step
  (`parity.compare`: `{"image": {"atol": 1}}`). (`tests/test_convert_to_image.py`)
- `CompilePackage.overrides` are the package op's own dotted keys, applied through `load_package_graph` and written in
  the manifest: what was compiled is the overridden graph. (`tests/test_compile_package.py::test_overrides_reach_the_package_s_graph_and_are_written_in_the_manifest`)
- **Items split by ONE rule serving both faces.** `item_rule(type)` gives the registered
  `@lowers_item` rule or the introspected default (array → `data`; `NDArrayItem` → `data` + attrs;
  number → `value` scalar part; string/None → `value` fact; dataclass field by field: arrays and
  number lists are data parts, numbers scalar parts, the rest facts; `bool` is a fact). The compiler
  derives the interface from `split`; the compiled op feeds with `split` and writes back with
  `join`. Never a second conversion path. §5.
  (`tests/test_items.py`, `tests/test_compile_package.py::test_the_compiled_folder_runs_on_its_own_after_the_package_is_gone`)
- **`torch.export` is the hub.** `build_program` runs every lowering eagerly on the first sample
  (modules/buffers register, writes are checked, per-node outputs kept), then exports; a backend
  takes the `Program` and writes files, and gives a runner from the folder alone — the port names
  travel INSIDE the artifact. Backends are `@configurable(category="compiler_backend")`;
  `backend_named` resolves a manifest's `backend:`. §6.
  (`tests/test_program.py::test_a_backend_reproduces_the_python_graph_on_a_picture_of_another_size`)
- **The interface is what the lowerings READ, measured; the envelope decides what is fixed among it**
  (2026-10-13, user decision A). A probe forward on the first sample hands every input entry's parts
  and facts to the lowerings as `Watched` mappings (`lowering.items`; a read, a `get`, a walk or a
  `dict()` copy — `ItemValue.replace` — counts, `in` does not), and `_input_specs` keeps only what was
  seen: a field no lowering looked at (a capture's derived `duration`, its SNR fields) is neither an
  input nor `specialized`, and samples may differ there. Among the read parts: `<entry>.<part>: varies`
  = every dimension dynamic (a data part) or a 0-d input tensor (a scalar); a value = fixed and
  `specialized`, every sample checked; unnamed = what the samples do — shared → fixed, disagreeing →
  dynamic (a scalar as a 0-d input, as a tensor dimension already was). A read FACT must be shared.
  An envelope key naming no input part is refused listing them; one naming an unread part is refused
  listing what is read. Output dimensions are measured over the samples; output facts and constant
  parts land under `constants`. Never demand agreement on a field nobody reads, and never make an
  unread field an input. §7.
  (`tests/test_program.py::test_an_input_field_no_step_reads_is_not_part_of_the_interface`,
  `::test_a_read_scalar_the_samples_disagree_on_varies_without_being_named`,
  `::test_a_read_scalar_the_samples_disagree_on_is_an_input_and_a_read_fact_is_refused`,
  `::test_an_envelope_value_still_refuses_a_sample_that_disagrees_with_it`,
  `::test_an_envelope_naming_a_field_no_step_reads_is_refused_listing_what_is_read`)
- **Parity decides the compile.** Every sample record through the Python graph (`run_steps_multi`)
  and the artifact; `parity.compare` per entry by its item rule — part by part (arrays within
  `atol`/`rtol`, integers by whole steps, facts exact; a per-entry tolerance wins), or the rule's own
  `ItemRule.compare(expected, actual, tolerances)` when it returns a list (a set of boxes matched by
  overlap and label, never row by row — a model's row order is its ranking), the entry's tolerance
  mapping handed to it with the `atol`/`rtol` defaults merged in. A failure leaves `<name>.partial/`
  with `parity.yaml`, names the FIRST node whose eager torch outputs disagree with its op's (the
  bisect, judged by the same tolerances) and that node's measured gaps (`first_node_differences`, also
  in the error's text), and the compiled folder is never moved into place. §8.
  (`tests/test_parity.py::test_a_wrong_lowering_fails_the_compile_and_names_the_first_disagreeing_node`,
  `::test_a_rule_s_own_comparison_wins_over_the_part_by_part_one`)
- **The TorchScript file is a trace of the LOWERED module, never of the exported one** (2026-10-13):
  `torch.export` turns a size a lowering reads off a tensor (a picture as tall as its raster, resized
  to a fixed width) into a `sym_size.int` node, and `torch.jit.trace` bakes that node's int to the
  trace sample's value — the artifact then resized every record to the first sample's height
  (measured: 40 rows in, 30 out, every piece plan the sample's). `backends.TorchScript.compile`
  traces `program.module` (eager: the same size is an `aten::size` the trace keeps dynamic); the
  exported program still validates the lowering under the envelope and is what `TorchExport` saves.
  A lowering still reads a size off a tensor (`_extent`-style), never from Python arithmetic on a
  shape, so eager and export agree.
  (`tests/test_convert_to_image.py::test_a_torchscript_artifact_keeps_a_raster_following_height_dynamic`)
- **The compiled folder is STANDALONE** (user ruling 2026-10-13): `class_names.json` is copied in,
  the manifest names the package only as provenance (`source.package`, `source.package_sha256`), and
  `CompiledPackage(path=<compiled folder>)` runs it after the package is gone. Default location
  `<package>/compiled/<name>/`, `output_dir` elsewhere; a non-empty folder that is not a compiled
  package is never written into; a compile replaces its folder. §8.
  (`tests/test_compile_package.py::test_the_compiled_folder_runs_on_its_own_after_the_package_is_gone`,
  `::test_a_folder_that_is_not_a_compiled_package_is_never_written_into`)
- The op refuses a record the artifact was not compiled for, naming the part and both values (a
  specialized value, another dtype, another rank or fixed size, a missing entry); `consumes` /
  `produces` are read off the manifest's `consumes` / `produces`.
  (`tests/test_compile_package.py::test_a_record_outside_the_envelope_is_refused_naming_the_part_and_both_values`,
  `::test_the_op_declares_what_it_consumes_and_produces_from_the_manifest`)
- A package is loaded ONLY through `matrainer.model_package.load_package_graph(..., owner="CompilePackage")`
  — the op's own loader; step locations come from a second, document-stage load of its `request`.
  Never a `confluid.load` of a package written here.
- No new verb: a compile is a `runnable:` (`recordstream run compile.yaml`); the op is an `op:`.
  No fluent `.forge()` API, no `matrainer compile` subcommand.
- `torch` is imported only in `lowering/context.py`, `lowering/recordstream.py`, `program.py`,
  `backends/` and a backend's runner; `torpaido/__init__.py` reaches `CompilePackage` /
  `CompiledPackage` lazily (PEP 562) and the IR, front end and dataflow pass import no framework.
  `pyproject.toml` declares torch under `[torch]`, named by `[tool.aisland] frameworks`.
- `compiled.yaml` is format 1 and carries the generator header; a field a reader may ignore is
  added freely, a field that changes meaning bumps `COMPILED_FORMAT`; the op refuses another format.
  (`tests/test_compile_package.py::test_the_manifest_carries_the_generator_header`,
  `::test_a_compiled_package_of_another_format_is_refused`)
- Tests run from inside `torpaido/`; a Python snippet that imports workspace packages runs from a
  directory that is NOT the workspace root (the project folders shadow the packages there).
