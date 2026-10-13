# Compiling a model package

The user guide of the compiler. The decisions behind each mechanism are in
[`architecture.md`](architecture.md) (§2 to §8); this page says how to use it.

## 1. What goes in, what comes out

**In:** a model package — the folder a training run hands off: `package.yaml`, the inference graph and
every file it includes, the weights mapped to their slots, `class_names.json`. The compiler loads it
through the package's own loader, so the compiled form matches exactly the graph the package op runs.

**Out:** a standalone compiled folder, default `<package>/compiled/<name>/`:

| file | what it is |
| --- | --- |
| `compiled.yaml` | the manifest: the backend, its files with digests, every input and output tensor (`<entry>.<part>`: dtype, shape, item type), `specialized` (inputs fixed), `constants` (outputs constant), `envelope`, `pruned`, the parity verdict, the installed package versions, the package as provenance |
| the backend's file(s) | `model.pt` (TorchScript), `program.pt2` (torch.export) |
| `class_names.json` | the package's vocabulary, copied — the artifact emits `int64` label indices |
| `parity.yaml` | per sample record, per output entry: the differences measured (empty when parity holds) |

Nothing in the folder reads the package again; move it, ship it, run it alone.

## 2. The compile config

```yaml
runnable: !class:torpaido.CompilePackage
  package: ./runs/detector/package
  name: torchscript
  backend: !class:torpaido.backends.TorchScript {}
  outputs: [predictions]
  envelope:
    signal.samplerate: 100e6
    signal.data: varies
  samples: !class:recordstream.sources.range.RangeSource
    source: !include ../data/captures.source.yaml
    stop: 8
  sample_count: 8
  tolerances: {atol: 1e-5, rtol: 1e-4, predictions: {atol: 1e-3}}
  description: the detector, scripted for the 100 MSa/s analyzer
```

```bash
recordstream run compile/detector.yaml
```

| setting | meaning |
| --- | --- |
| `package` | the package folder (`$VAR` and `~` expanded) |
| `name` | the compiled folder's name under `<package>/compiled/`; how a consumer tells the forms apart |
| `backend` | the backend object; left out = TorchScript |
| `outputs` | the record entries the artifact delivers; left out = what the last writing step writes |
| `envelope` | what varies and what is fixed (§4) |
| `samples` | a source of records (anything iterable) — measured on, verified against |
| `sample_count` | how many records to take from `samples` (default 4) |
| `output_dir` | somewhere else than `<package>/compiled/<name>` |
| `tolerances` | `atol` / `rtol` defaults and per-entry overrides for parity; an entry whose item rule compares its own way (a set of boxes matched by overlap) reads its keys from the entry's mapping (§8) |
| `description` | one line for the manifest |

A compile that fails parity leaves `<name>.partial/` with its `parity.yaml` and raises; a compile that
passes replaces `<name>/`. A folder that holds other files is never written into.

## 3. Running the compiled folder

```yaml
flow:
  model:
    op: !class:torpaido.CompiledPackage
      path: $PROJECT_ROOT/runs/detector/package/compiled/torchscript
```

The op reads the record's input entries (the manifest's `consumes`), splits them into arrays by their
item rules, checks dtype, rank, every fixed size and every specialized value, runs the engine, and
writes the output entries back as record items. The rest of the record passes through. Refusals name
what disagreed:

```
CompiledPackage: …/torchscript was compiled for signal.samplerate = 100000000.0, and this record has 61440000.0 — …
CompiledPackage: …/torchscript: image.data is float32, and the compiled package takes uint8
CompiledPackage: …/torchscript: image.data has shape (3, 7, 3), and the compiled package takes [4, 6, 3]
CompiledPackage: …/torchscript: the record has no entry 'image' (it has: gain, tag)
```

`consumes` / `produces` are read off the manifest, so a chain check sees the compiled op as it sees any
op.

## 4. The envelope

An input part is `<entry>.<part>` — `signal.data`, `signal.samplerate`, `image.layout`.

| the envelope says | the artifact |
| --- | --- |
| `signal.data: varies` | every dimension of that tensor is dynamic |
| `signal.samplerate: varies` | the scalar is an input tensor (0-d, `float64` / `int64`) |
| `signal.samplerate: 100e6` | fixed: baked in, under `specialized`; checked against every sample |
| *(not named)* | fixed at the samples' value, which every sample must share; a tensor dimension the samples disagree on is dynamic |

Facts (a layout, a scaling name) are always fixed; the samples must agree on them.

Refusals, with what to do:

```
torpaido: the input gain.value is fixed, but the samples disagree: record 1 has 2.0 and record 2 has 3.0
  Let it vary (envelope: gain.value: varies) or compile from samples that agree.

torpaido: the envelope names 'picture.data', which is not an input part (gain.value, image.data, image.layout)

torpaido: step 'tile' (mydomain.ops.Tile) needs 'count' as a constant, but it depends on count.value, which
the envelope lets vary at cfg/model.yaml:3:3
  Fix it in the envelope (count.value: <value>) or leave it out so the samples' value is used.
```

## 5. Which steps are compiled

Dataflow is measured: each sample record runs through the graph, and each step's reads are found by
rerunning it with one entry left out. From the chosen outputs, everything not needed is pruned and listed
under `pruned`. On a detection package that is the box filter, the picture scale, the training target and
the trailing drop. The inputs are the source entries the surviving steps read.

Two records that make a step read different entries are refused:

```
torpaido: step 'fickle' reads ['level', 'tag'] on record 1 and ['level'] on record 2 — a data-dependent read
has no static graph at model.yaml:7:3
```

## 6. Writing a lowering

```python
# mydomain/lowering/torch.py
import torch
from torpaido.lowering import Context, ItemValue, lowers

from mydomain.processing import Spectrogram


@lowers(Spectrogram)
def lower_spectrogram(op: Spectrogram, reads: dict[str, ItemValue], ctx: Context) -> dict[str, ItemValue]:
    signal = reads[op.field]                                   # what the op was measured to read
    rate = signal.parts["samplerate"]                          # a number (fixed) or a 0-d tensor (varies)
    points = ctx.constant(op.fft_points(rate), "fft_size")     # refused when it depends on a varying input
    window = ctx.buffer("window", lambda: torch.hann_window(points, periodic=False))
    frames = signal.data.unfold(0, points, op.hop_samples(points)) * window
    spectrum = torch.fft.fft(frames, n=points, dim=-1).abs()
    return {op.output: ItemValue(SpectrogramItem, {"data": spectrum, "samplerate": rate}, {"scaling": "none"})}
```

Rules of the contract:

* **Read the settings off the op** (`op.fft_size`); read the entries the op reads from `reads`, by the
  op's own parameter (`op.field`), never by a hard-coded name.
* **Return exactly the entries the op writes.** A lowering that writes another set is refused.
* **A model is a module**: `ctx.module("net", op.model)` registers it once; its weights travel with the
  artifact. `ctx.buffer(name, factory)` keeps a tensor made once. `ctx.constant(value, name)` turns a
  value into a Python number and refuses a varying one.
* **No hidden state**: the function runs twice (eagerly, then under `torch.export`); everything it keeps
  goes through the context, by name.

Register the module:

```toml
[project.optional-dependencies]
compile = ["torpaido[torch]"]

[project.entry-points."torpaido.lowerings"]
mydomain = "mydomain.lowering.torch"
```

Test it the way the compiler will judge it — against the numpy op, on real-shaped inputs:

```python
def test_spectrogram_lowers_to_the_numpy_result():
    record = {"signal": capture}
    expected = Spectrogram(fft_size=640)(record)["spectrogram"]
    graph = measure(*parse_flow({"s": {"op": Spectrogram(fft_size=640)}}, "s"), [record])
    graph.prune_sidecars()
    program = build_program(graph, [record])
    assert compare({"spectrogram": expected}, program.eager(record)["s"], ["spectrogram"], {"atol": 1e-4}) == []
```

## 7. Items

An item type splits into parts and facts by `item_rule(type)`; register your own where the default is
wrong:

```python
@lowers_item(Regions)
class RegionsRule(ItemRule):
    def split(self, item):
        rows = np.column_stack([item.freq_lower_edge_hz, item.freq_upper_edge_hz, item.time_start_s, item.time_end_s])
        return {"boxes": rows, "labels": np.asarray(item.labels), "scores": np.asarray(item.scores)}, {"classes": None}

    def join(self, parts, facts):
        boxes = np.asarray(parts["boxes"])
        return Regions(list(boxes[:, 0]), list(boxes[:, 1]), list(boxes[:, 2]), list(boxes[:, 3]),
                       labels=np.asarray(parts["labels"]).tolist(), scores=np.asarray(parts["scores"]).tolist())
```

The rule is used by the compiler (the interface), by the compiled op (feeding and writing back) and by
parity (part by part) — one definition.

## 8. Parity

`parity.yaml`:

```yaml
status: passed
tolerances: {atol: 1.0e-05, rtol: 0.0001}
records:
- {record: 1, differences: []}
- {record: 2, differences: []}
```

Each output entry is compared by its item rule: part by part — arrays within `atol` / `rtol`, integers by
whole steps, facts exactly — unless the rule gives its own `compare(expected, actual, tolerances)`. A rule
does that when its rows have no order of their own: a detector's boxes are ranked by the model, and two
scores a hair apart swap places on a picture that differs by one grey level, so a row-by-row compare fails
a correct artifact. Such a rule matches the rows (by overlap and label) and reads its own keys from the
entry's tolerance mapping:

```yaml
tolerances:
  atol: 1.0e-05                                   # the defaults, for every part-by-part entry
  predictions: {iou: 0.9, score_atol: 0.02}       # what the box set's rule reads
```

```python
class BoxSetRule(ItemRule):
    def compare(self, expected, actual, tolerances):
        unmatched = ...  # rows of `expected` with no row of `actual` of the same label at IoU >= tolerances.get("iou", 0.99)
        return [("boxes", f"{len(unmatched)} rows unmatched")] if unmatched else []
```

On a failure the compile raises, and the report carries `first_disagreeing_node` — the first step whose
lowering, run eagerly in torch, disagrees with its numpy op on that record, judged by the same tolerances —
and `first_node_differences`, that node's own measured gaps (`[{entry, part, measure}]`), so the fix is a
lowering, named, with its error in hand, not a tensor diff:

```yaml
status: failed
first_disagreeing_node: picture/image
first_node_differences:
- {entry: image, part: data, measure: 'max |a - b| = 1.0 (atol 1e-05, rtol 0.0001)'}
```

## 9. Backends

| `backend:` | class | writes | runs with |
| --- | --- | --- | --- |
| `torchscript` | `torpaido.backends.TorchScript` | `model.pt` (a trace of the lowered module: a size read off a tensor stays dynamic) | `torch.jit.load` (the C++ runtimes that exist today) |
| `torch-export` | `torpaido.backends.TorchExport` | `program.pt2` | `torch.export.load` |

Both start from the same exported program; the manifest is the same whichever wrote the files.
