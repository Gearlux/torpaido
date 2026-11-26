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

The mandates in this project's `AGENTS.md` start from dependencies. **Selective Pruning First**
requires reverse-dependency analysis; `Graph.prune_sidecars` implements it by walking
`node.inputs` backwards from `self.outputs`. **Metadata Promotion** requires knowing which values
a node actually consumes. The design goal **Unpacked Handover** (the README) likewise requires
knowing which producer feeds which consumer so a boxed record can be replaced with direct
tensor-to-tensor edges.

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

---

## 2. Dataflow is measured by ablation on sample records, never read off declarations (`torpaido.dataflow`, 2026-10-13)

### Context

Pruning and the artifact's interface both need to know which record entry each step READS and WRITES.
recordstream's ops declare `consumes` / `produces`, but the declarations are documentary — recordstream's
own words: "nothing validates them against the op's behavior" — and a step-level edge ("this step reads
the previous step's record") is a superset that prunes nothing: in a linear chain every step "needs" every
earlier one. The reference compiler never had this information and compiled every transform in,
training-only sidecars included.

### Decision

`measure(steps, result, records)` runs every sample record through the graph on recordstream's own
`Tracer` (snapshots per node, `changed` / `removed` per node) and derives, per node:

* **writes** = `changed`, **removes** = `removed`;
* **reads** by ABLATION: the op is rerun on its input snapshot with ONE entry left out; if it raises,
  drops the record, or any written entry comes out different (`same`, element for element), that entry
  was read. An op that writes nothing reads nothing.

Every sample record is measured; a node whose reads or writes differ between two records is refused
naming the node and both records — a data-dependent read has no static graph. A read then resolves to
the nearest ancestor over the STRUCTURAL `inputs` (depth-first, the `from:` chain first) that wrote the
entry; the resolution runs for every node before `Node.inputs` is narrowed to the producers, because a
narrowed earlier node would cut the chain for a later one. The outputs default to what the last writing
node on the path to the yielded step wrote.

Rejected: trusting declarations (prunes a needed step when a declaration is stale); a recording record
(a dict logging `__getitem__`) — type-dispatched ops iterate `items()` and would read everything, while
`field=""` ops that scan by type would read nothing.

### Consequences

* Pruning is exact, and the artifact's inputs are exactly the source entries the surviving nodes read.
* The compile needs real records and runs each op once per input entry per sample — on a model step that
  is one model call per entry. Acceptable for a compile; the cost on a large package is to be measured
  (`TASKS.md`).
* An op whose result depends on an entry only through a side effect the written entries do not show
  is measured as not reading it; such an op has no place in an inference graph.

### Example

```python
graph = measure(steps, "done", [record_a, record_b])
graph.nodes["classify"].reads          # {"image": Value("pre/bright.image", "image", Image)}
graph.prune_sidecars()                 # ["level", "stamp", "done"] — nothing the outputs need
[v.name for v in graph.input_values]   # ["source.image", "source.gain"]
```

### What you may change

A cheaper measurement (ablation on the first record, the pruned graph verified on the rest) keeps the
definition; a declaration-based shortcut does not and is refused. A new Tracer capability (per-step
locations, a public snapshot API) replaces the `to_dict()` + `value()` reads here.

---

## 3. A subgraph step is inlined under the Tracer's names (`torpaido.frontend`, 2026-10-13)

### Context

A package graph nests: its preprocessing is a `Subgraph` whose inside is a steps file. The compiler must
reach the inner steps (they are what gets lowered), and every refusal and every measurement must name a
node the way the person debugging the graph sees it.

### Decision

`graph_from_steps` inlines a step whose op is a subgraph (duck-typed on the class marker `FLOW_SUBGRAPH`,
as recordstream does): the inner steps become `<step>/<inner>` nodes — the names recordstream's `Tracer`
gives them — the first inner step reading what the step reads, and the step itself stays as an IDENTITY
node over its inner result so a later `from: <step>` resolves. An inner node inherits its step's
location, suffixed `(step <name>)`, because a built subgraph keeps no trace of where its inner steps were
written.

### Consequences

* `measure` matches Tracer rows to IR nodes by name; the container rows (the subgraph step) are skipped
  and the identity node is pruned silently.
* Nested subgraphs nest the prefix (`prep/inner/x`), as the Tracer does.

### Example

```python
graph.nodes["pre/scale"].inputs   # ["start"]       — the subgraph's own input
graph.nodes["pre"].op is None     # the container, an identity over "pre/bright"
graph.nodes["pre/a"].where        # "model.yaml:5:9 (step pre/a)"
```

### What you may change

If recordstream ever carries per-inner-step locations, read them instead of the suffix.

---

## 4. One lowering function per op class, by exact class, discovered by entry point (`torpaido.lowering.registry`, 2026-10-13)

### Context

The reference compiler gave every transform a hand-written TorchScript twin CLASS beside its numpy
original (2 600 lines of twins in the core, twenty more files in the domain) through a `CompilableModule`
interface or a `@compilable(CompilerClass)` decorator on the eager class; the twins drifted (a window's
`periodic` flag diverged until a test caught it), and the eager classes had to know the compiler.

### Decision

A lowering is a plain function `(op, reads, ctx) -> writes`: the configured op (its settings read off
it), `{entry: ItemValue}` for the entries the op was MEASURED to read, a `Context` (modules, buffers,
constants), and `{entry: ItemValue}` for exactly the entries the op writes. `@lowers(OpClass)` registers
it; the lookup is by EXACT class — a subclass may compute something else, and inheriting a lowering would
leave only the parity check between it and a wrong artifact. `discover()` imports torpaido's own module
and every module under the entry-point group `torpaido.lowerings` (confluid's `load_configurables`).
A domain package ships its lowerings behind a `[compile]` extra; the dependency points domain → compiler
(user decisions D1, D2 of 2026-10-13: torch as the lowering language; lowerings in the owning package).

Refusals are located: a surviving node without a lowering names its `file:line:col`, what it writes and
how to register one; a lowering writing other entries than its op is refused; `ctx.constant` refuses a
varying value naming the input it depends on.

### Consequences

* The numpy op stays untouched and framework-free; parity (§8) is what holds the two faces together.
* A pruned node needs no lowering — the training-only steps of a package never do.
* The function runs twice per compile (eagerly, then under `torch.export`), so everything it registers
  is keyed by name through the context.
* Discovery keeps its evidence: `discover()` records what the entry-point group gave (`DISCOVERED`, each
  entry's module or the exception its import raised), and a held object without a lowering is refused
  with that list — a lowering that exists but is not registered (a package installed before it declared
  its entry point) must not look like one never written (2026-10-13).

### Example

```python
@lowers(Magnitude)
def lower_magnitude(op, reads, ctx):
    spec = reads[op.field]
    db = 20.0 * torch.log10(spec.data + op.eps)
    return {op.field: spec.replace(data=db.to(torch.float32), scaling="db")}
```

### What you may change

Which modules torpaido imports itself (`lowering/recordstream.py` holds recordstream's ops). Not the
exact-class lookup, and not an inheritance of lowerings.

---

## 5. An item splits into parts and facts by one rule that serves compile time and run time (`torpaido.lowering.items`, 2026-10-13)

### Context

A record entry is a typed item (an `Image` with a layout, a `Label` with a vocabulary, a domain's region
set); a tensor program knows tensors and constants. The reference compiler threaded a `Dict[str, Tensor]`
side channel through every module and encoded label strings as zero-padded ASCII rows that the deployment
runtime decoded by a convention written in a rulebook.

### Decision

An `ItemRule` says, per item type, how an item splits into **data parts** (arrays and number lists →
tensors), **scalar parts** (numbers → a 0-d tensor when the envelope lets them vary, else constants) and
**facts** (strings, `None`, flags, vocabularies → constants written in the manifest). The default is
introspected from the type (dataclass field by field; `NDArrayItem` as data plus declared attributes;
number; string); `@lowers_item(Type)` registers an override. The SAME rule derives the artifact's
interface at compile time, feeds the engine in `CompiledPackage` (`split`) and writes the record back
(`join`). Labels are `int64` indices; the vocabulary is `class_names.json`, copied into the folder.

The rule also owns how two items of its type are judged EQUAL for parity: by default part by part (what
`split` gives), or by its own `compare(expected, actual, tolerances)` when the parts' order carries no
meaning — a detector's rows are ordered by its ranking, and two scores a hair apart swap places on a
picture one grey level different, so a row-by-row compare fails a correct artifact (measured 2026-10-13: a
compiled picture differs from numpy's by one level on 13 % of its pixels after the resize and quantization).
`parity.compare` hands such a rule the entry's own tolerance mapping (`predictions: {iou: 0.9}`), the
`atol` / `rtol` defaults merged in, and records what it returns as the entry's differences.

### Consequences

* The manifest's `inputs:` / `outputs:` are `<entry>.<part>` ports with dtype, shape and item type; a
  deployment runtime reads names, never conventions.
* A non-numeric list (a list of strings) is a fact and must be constant across the samples; a domain
  whose item needs another split registers a rule (a region set as one `[N, 4]` tensor).
* A domain's notion of "the same set" lives in its rule, beside its split, not in a tolerance table in
  the compiler: the compiler never learns what a box is.

### Example

```python
parts, facts = item_rule(Image).split(image)      # {"data": <uint8 HWC>}, {"layout": "HWC"}
item_rule(Label).split(Label(3))                  # {"value": 3}, {"classes": None}
item_rule(Image).join({"data": array}, {"layout": "HWC"})


@lowers_item(BoxSet)
class BoxSetRule(ItemRule):
    def split(self, item): ...
    def join(self, parts, facts): ...

    def compare(self, expected, actual, tolerances):   # a set: rows matched by overlap and label
        floor = tolerances.get("iou", 0.99)
        unmatched = [row for row in expected.rows if not any(iou(row, other) >= floor for other in actual.rows)]
        return [("boxes", f"{len(unmatched)} rows unmatched")] if unmatched else []
```

### What you may change

Add introspected kinds (a `PIL` image, a `pathlib.Path` fact) and domain rules; never a second
conversion path beside `split` / `join`, and never a comparison table in the compiler that knows an
item type by name — a rule's `compare` is where that knowledge goes.

---

## 6. `torch.export` is the hub; a backend writes files and gives a runner from the folder alone (`torpaido.program`, `torpaido.backends`, 2026-10-13)

### Context

The reference compiler scripted one hand-assembled `nn.Module` with `torch.jit.script`; TorchScript was
baked into its interfaces, and a failure anywhere surfaced as a TorchScript error deep inside the module.

### Decision

`build_program` turns the measured, pruned graph into ONE `nn.Module` (`LoweredGraph`: inputs by port
order, nodes in schedule order, each lowering run with the values it reads, outputs flattened to tensors),
runs it eagerly on the first sample (modules and buffers register, writes are checked, per-node outputs
kept for the bisect), and exports it with `torch.export` under the dynamic shapes the envelope states.
A `Backend` takes the `Program` and writes files (`TorchScript`: `torch.jit.trace` of the LOWERED
module → `model.pt`; `TorchExport`: `torch.export.save` → `program.pt2`; `Onnx`:
`torch.onnx.export(program.exported, dynamo=True)` → `model.onnx`, run by onnxruntime, added 2026-10-15) and
gives a `Runner` from the folder alone: the port names travel inside the artifact — an extra file of the
torch forms, the ONNX graph's own input and output names. The TorchScript trace is of the
eager lowered module, not of the exported program's module (changed 2026-10-13): export turns a size a
lowering reads off a tensor — a picture as tall as its raster, resized to a fixed width — into a
`sym_size.int` node, and `torch.jit.trace` bakes that node's int to the trace sample's value, so the
artifact resized every record to the first sample's height (measured: 40 rows in, 30 out) while the
eager trace keeps the same size as an `aten::size`. The exported program still validates every lowering
under the envelope and is what `TorchExport` saves — the hub stays; TorchScript's file is traced beside
it. Backends are
`@configurable(category="compiler_backend")`; `backend_named` resolves a manifest's `backend:`.
TorchScript came first (user decision D4) because the deployment runtime that exists loads it.

### Consequences

* One lowering language, N artifacts; a backend never sees a lowering or a record.
* TorchScript is in maintenance upstream; its API warnings are silenced inside the backend, where the
  user's choice of it is explicit.
* The ONNX backend costs no second lowering: the same exported program, under the port names, is one
  `model.onnx` (the weights inline, `onnx.checker`'s full check, opset 18 and up — a lower `opset_version`
  is refused rather than raised silently, since the exporter wrote 18 when asked for 17). The exported
  program's `sym_size` becomes a `Shape` node, so a raster-following height stays dynamic there too
  (measured 2026-10-15: 30, 34, 40 and 7 rows in and out). A dimension `varies` freed and export then fixed
  (a channel count a convolution needs) is written under the exporter's name for it (`[s95, s85, "3"]`), not
  as a size; the manifest stays the contract. The `[onnx]` extra's packages are checked by name before any
  import, so a missing one is a refusal naming `torpaido[onnx]`, never a `ModuleNotFoundError` from inside
  torch.
* **ONNX has no complex tensors, so the ONNX file is the lowered module behind a real-pair adapter**
  (2026-10-15): a complex port enters the adapter as a `float32 [..., 2]` tensor (I then Q) and is turned
  complex before the lowered graph, a complex output leaves as a pair, the model's metadata names those ports
  and the runner converts both ways — `compiled.yaml` keeps `complex64`, a flow sees the record's complex
  arrays. The adapter is re-exported under the interface's dynamic shapes (one path for every program) and
  handed to the exporter; the exporter refused a complex graph input outright (`aten.sym_size.int ... no
  decompositions registered for the complex-valued input`, measured on the signal-detection package). The
  exporter translates a handful of ops on a complex value (the FFT, a slice, a concatenation, the magnitude,
  an element-wise product) and not a framing, a mask or a size read, which is the domain lowering's business:
  the spectrogram frames on the real view and turns complex at the FFT. A stable sort (`aten.sort.stable`, a
  detector library's ranking) has no translation of the exporter's own and is written as ONNX's sort (TopK)
  through the exporter's translation table; a tie's order is then the runtime's, which the set comparison of
  a region set tolerates by design. A 0-d input is fed 0-d (`np.ascontiguousarray` made it `[1]`, and every
  scalar derived from it came back as a one-element list — measured on a real package's `time_end_s`).
  Measured on the signal-detection package (an RF-DETR segmenter on 1024 x 1024 pictures): a 26 s compile of
  which the export is 5 s, one 126 MB file, onnxruntime 0.06-0.13 s a capture against 0.16-0.18 s eager torch,
  parity on four captures without a difference under the shipped tolerances.
* **A refusal of an op the exporter cannot translate is located by the STEP** (2026-10-15): `LoweredGraph.forward`
  runs every lowering under `fx_traceback.annotate({STEP_ANNOTATION: step})`, so every fx node of a program
  exported under `preserve_node_meta()` names the step whose lowering wrote it — the export's own stack trace
  stops at the lowered graph's `forward` line and never reaches the lowering. The ONNX backend parses the
  exporter's conversion error for the node, reads the step off it, and refuses naming the op, whether its value
  was complex, the step, its op class and where the lowering is defined — instead of the exporter's forty
  lines.

### Example

```python
program = build_program(graph, samples, envelope={"image.data": "varies"})
files = TorchScript().compile(program, folder)          # [folder / "model.pt"]
runner = TorchScript().runner(folder, ["model.pt"])     # knows the port order from the artifact
runner({"image.data": array})                           # {"scores.data": ..., "label.value": ...}
Onnx(opset_version=18).compile(program, folder)         # [folder / "model.onnx"]: the ports are the graph's own names
# compiled.yaml: inputs: {input.data: {dtype: complex64, shape: [null]}}   the ONNX input: input.data float32 [N, 2]
```

### What you may change

New backends behind their own extras; `LoweredGraph.forward` stays a varargs tensor interface (a C++
runtime wants positional tensors).

---

## 7. The interface is what the lowerings read; the envelope fixes or frees among it, and the manifest states what was fixed (`torpaido.program`, 2026-10-13)

### Context

A graph may resolve a setting per record (an FFT length from the sample rate); an exported program cannot.
The reference compiler read such values from its side channel at run time, or baked them silently. A
domain's capture item also carries fields derived from the samples (a duration, SNR measurements, a
modulation name) that no step of the inference graph reads; a first version of this compiler made every
field of an input entry a port and demanded the samples agree on each — a compile of a real package
stopped on two captures 24.9 ms and 21.2 ms long, over a `duration` nothing read (user decision A,
2026-10-13: measure the reads, and let a read scalar the samples disagree on vary).

### Decision

The interface is MEASURED: a probe forward on the first sample hands every input entry's parts and facts
to the lowerings through recording mappings (`Watched`: a read, a `get`, a walk or a `dict()` copy counts,
`in` does not), and only what some lowering looked at enters the interface. Among that,
`envelope: {<entry>.<part>: varies | value}`: `varies` makes every dimension of a data part dynamic
(`Dim.AUTO`) or a scalar part a 0-d input tensor; a value fixes the part, checked against every sample
and written under `specialized`; an unnamed part follows the samples — a value they share is fixed and
stated, a scalar or a tensor dimension they disagree on is dynamic. A read fact (a layout, a scaling) is
always fixed and the samples must share it. Output dimensions are measured over the samples; output facts
and constant parts land under `constants`. A lowering that needs a constant from a varying part is refused
naming the dependency and the two fixes.

### Consequences

* The manifest states exactly what the artifact assumes and nothing more: a capture's unread fields
  never appear, `CompiledPackage` never checks them, and samples may differ there.
* Nothing read is specialized silently: `compiled.yaml` says which sample rate, layout and band the
  artifact answers for, and `CompiledPackage` refuses a record outside it.
* The envelope is for intent the samples cannot show: `varies` on a part the samples happen to share, or
  a value to pin. An envelope key naming an unread part is refused listing what is read.
* `varies` on a single sample's output cannot be observed as dynamic in the manifest (there is one
  shape); give the compiler samples that differ. The probe costs one extra lowering pass on the first
  sample (one model call).

### Example

```yaml
# a Signal has data, samplerate, center_freq, duration, bandwidth, snr_db, …; the spectrogram lowering reads
# data, samplerate and center_freq. Samples: two captures of different length, rate and centre.
envelope: {input.data: varies}
# compiled.yaml → inputs: {input.data: {shape: [null]}, input.samplerate: {shape: []}, input.center_freq: {shape: []}}
#                 specialized: {}        — duration, bandwidth, snr_db … are nobody's business
```

```python
reads["clip"].parts["rate"]          # seen: clip.rate enters the interface
"rate" in reads["clip"].parts        # not a look
clip.replace(data=scaled)            # a dict() copy: every part the item carries forward is seen
```

### What you may change

A per-dimension spelling (`image.data: [varies, varies, 3]`) if a package needs it; the rule that a shared
read value is fixed and stated. Never a declared list of "interface fields" per item type — the reads
are measured, like the dataflow (record 2).

---

## 8. Parity on real records, bisected to the first disagreeing node; the compiled folder stands alone (`torpaido.compile`, `torpaido.compiled`, `torpaido.manifest`, 2026-10-13)

### Context

The reference compiler verified on synthetic zero tensors of declared sizes and compared the final tensor
with `allclose`; its output was one `.pt` whose contents the consumer had to know, with `lexicon.json`
beside it that nobody read at deployment.

### Decision

`CompilePackage` runs every sample record through the Python graph and the artifact and compares each
output entry by its item rule (`parity.compare`: part by part — arrays within `atol`/`rtol`, integers by
whole steps, facts exactly, a per-entry tolerance winning — or the rule's own `compare`, record 5, for a
set whose rows have no order). A failure writes `parity.yaml` into `<name>.partial/`, names the FIRST node
whose eager torch outputs disagree with its op's Tracer snapshot (the bisect, judged by the same
tolerances) together with that node's own measured gaps (`first_node_differences`, repeated in the
error's text), and the folder is never moved into place. The folder is STANDALONE (user ruling D3, 2026-10-13): `compiled.yaml` (format 1,
generator header), the backend's files, a copy of `class_names.json`, `parity.yaml`; the package is named
only as provenance (`source.package`, `source.package_sha256`). `CompiledPackage(path=<folder>)` runs
it after the package is gone, refusing a record outside the envelope by part and both values. The
default location is `<package>/compiled/<name>/`; replacing the package deletes it (it was of the old
weights).

### Consequences

* A compile is a verified claim: what landed in the folder answered as the Python graph did on every
  sample record.
* A domain item needing a tolerant comparison (boxes matched by IoU) carries it in its rule's `compare`
  (record 5); the default is part-wise closeness. The compiler never learns an item type by name.
* The report says where AND by how much: the first disagreeing node's gap is in `parity.yaml` and in
  the error, so a lowering is fixed from the report, not re-measured.

### Example

```
CompilePackage: parity failed on record 1: scores.data max |a - b| = 139.4 (atol 1e-05, rtol 0.0001);
the first node that disagrees is 'pre/bright' (image.data max |a - b| = 139.4 (atol 1e-05, rtol 0.0001)) — the report is …/compiled/ts.partial/parity.yaml
```

```yaml
# parity.yaml of that failure
status: failed
first_disagreeing_node: pre/bright
first_node_differences:
- {entry: image, part: data, measure: 'max |a - b| = 139.4 (atol 1e-05, rtol 0.0001)'}
```

### What you may change

Fields a reader may ignore in `compiled.yaml`; a transport form (a zip of the folder). Never a folder
that reads the package again, and never a move into place before parity passed.
