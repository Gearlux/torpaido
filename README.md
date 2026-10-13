# Torpaido

**Torpaido** compiles a **model package** — a model's inference graph, its weights and its class names
in one folder, as the experiment-tracking project writes it — into an artifact an inference engine runs
without Python: TorchScript today, a `torch.export` program, ONNX next.

Built on `Confluid` (configuration), `Loggair` (logging), `RecordStream` (the record pipelines it
compiles) and the model-package format of the experiment-tracking project.

## What a compile does

1. **Loads the package** through the package's own loader — the graph the compiled form must match is
   exactly the graph the package op runs.
2. **Measures the dataflow** on sample records: which record entry each step actually read and wrote,
   found by running each step with one entry left out — never trusted from a declaration. Everything
   the chosen outputs never need is pruned: the training target, a picture scale, a box filter, a
   trailing drop.
3. **Lowers** every surviving step through its registered lowering — a torch function of the configured
   op and the values it reads — into ONE `torch.export` program.
4. **Writes the backend's files** and **verifies parity**: the artifact is run on every sample record
   against the Python graph, entry by entry. A failure names the first step whose lowering disagrees
   with its op.

The result is a **standalone folder**: the manifest `compiled.yaml` (every input and output tensor with
its dtype, shape and the record item it stands for; what was fixed; what was pruned; the parity
verdict), the artifact, a copy of `class_names.json`, and `parity.yaml`. Nothing in it reads the
package again.

## Compile a package

A compile is a config run like every other run — a `runnable:`:

```yaml
# compile/detector.yaml — run with:  recordstream run compile/detector.yaml
runnable: !class:torpaido.CompilePackage
  package: ./runs/detector/package              # the folder PackageModel wrote
  name: torchscript                            # → <package>/compiled/torchscript/
  backend: !class:torpaido.backends.TorchScript {}
  outputs: [predictions]                       # the record entries the artifact must deliver
  envelope:                                    # what the artifact is compiled FOR (see below)
    signal.samplerate: 100e6                   #   fixed: baked in; a record at another rate is refused
    signal.data: varies                        #   symbolic: any length
  samples: !class:recordstream.sources.range.RangeSource   # the records parity is measured on
    source: !include ../data/captures.source.yaml
    stop: 8
```

```
runs/detector/package/compiled/torchscript/
    compiled.yaml      the manifest (below)
    model.pt           the TorchScript module
    class_names.json   the package's class vocabulary, copied
    parity.yaml        per record, per output entry: the Python graph vs the artifact
```

The manifest, abridged:

```yaml
format: 1
backend: torchscript
source: {package: ./runs/detector/package, package_sha256: 3f2a…, graph: graphs/cfg/model.yaml}
files: [{path: model.pt, sha256: 9c1d…}]
class_names: class_names.json
inputs:
  signal.data:        {entry: signal, part: data, dtype: complex64, shape: [null], item: mydomain.items.Signal}
outputs:
  predictions.boxes:  {entry: predictions, part: boxes, dtype: float64, shape: [null, 4], item: mydomain.items.Regions}
  predictions.labels: {entry: predictions, part: labels, dtype: int64, shape: [null], item: mydomain.items.Regions}
  predictions.scores: {entry: predictions, part: scores, dtype: float32, shape: [null], item: mydomain.items.Regions}
specialized: {signal.samplerate: 100000000.0, signal.center_freq: 2440000000.0}
constants: {predictions.classes: null}
pruned: [picture/select, picture/scale, picture/boxes, done]
parity: {records: 8, status: passed, report: parity.yaml}
```

A deployment runtime reads `compiled.yaml` for the tensor names and dtypes and `class_names.json` for
the labels (the artifact emits `int64` indices). No key is agreed by convention.

Every knob, documented: [`config/compile.example.yaml`](config/compile.example.yaml).

## Run the compiled folder

One op wherever a flow graph or a stream takes one — the same slot the package op takes:

```yaml
flow:
  model:
    op: !class:torpaido.CompiledPackage
      path: $PROJECT_ROOT/runs/detector/package/compiled/torchscript
```

The op splits the record's input entries into arrays by the same rules the compiler used, checks them
against the manifest, runs the engine, and writes the output entries back as record items. A record the
artifact was not compiled for is refused, naming the part and both values:

```
CompiledPackage: …/compiled/torchscript was compiled for signal.samplerate = 100000000.0, and this record
has 61440000.0 — compile with the envelope letting it vary (signal.samplerate: varies) to run such records
```

## The envelope: what is fixed, what may vary

`envelope:` names input parts as `<entry>.<part>`:

| the envelope says | the artifact |
| --- | --- |
| `signal.data: varies` | every dimension of that tensor is dynamic |
| `signal.samplerate: varies` | the scalar becomes an input tensor (0-d, `float64`) |
| `signal.samplerate: 100e6` | fixed: baked in and written under `specialized:`; a record with another value is refused |
| *(not named)* | fixed at the samples' value, which every sample must share — except a tensor dimension the samples **disagree** on, which is dynamic |

So nothing is silent: `compiled.yaml` says which sample rate, which layout, which band the artifact
answers for. A step whose lowering needs a constant that the envelope lets vary is refused:

```
torpaido: step 'raster/spectrogram' (mydomain.processing.Spectrogram) needs 'fft_size' as a constant, but it
depends on signal.samplerate, which the envelope lets vary at graphs/spectrogram_raster.steps.yaml:14:5
  Fix it in the envelope (signal.samplerate: <value>) or leave it out so the samples' value is used.
```

## Which steps are compiled

`outputs:` names the record entries the artifact delivers; left out, they are the entries written by the
last writing step on the way to the graph's yielded step (a trailing drop writes nothing, so the default
looks one step back). Pruning walks the MEASURED edges backwards from those entries, and `pruned:` in the
manifest lists what was left out. The artifact's inputs are whatever is read from the source record and
produced by nothing — stated in `inputs:` and `specialized:`.

## Make your ops compilable: a lowering

A lowering is a plain function of the configured op, the values it reads (`{entry: ItemValue}`) and a
context, returning the values it writes — exactly the entries the op writes in Python. It is registered on
the op's class and lives in your package; the op never imports the compiler.

```python
# mydomain/lowering/torch.py
import torch
from torpaido.lowering import Context, ItemValue, lowers

from mydomain.processing import Magnitude


@lowers(Magnitude)
def lower_magnitude(op: Magnitude, reads: dict[str, ItemValue], ctx: Context) -> dict[str, ItemValue]:
    spec = reads[op.field]                               # the entry the op was measured to read
    power = spec.facts["scaling"] in ("power", "density")  # a FACT: constant inside one artifact
    db = 10.0 * torch.log10(spec.data + op.eps * op.eps) if power else 20.0 * torch.log10(spec.data + op.eps)
    return {op.field: spec.replace(data=db.to(torch.float32), scaling="db")}
```

```toml
# mydomain/pyproject.toml
[project.optional-dependencies]
compile = ["torpaido[torch]"]

[project.entry-points."torpaido.lowerings"]
mydomain = "mydomain.lowering.torch"
```

`pip install mydomain[compile]` makes the ops compilable: the compiler imports every module listed
under the entry-point group `torpaido.lowerings` before it lowers. The context gives a lowering what a
program has to hold: `ctx.module(name, nn_module)` registers a model (its weights travel with the
artifact), `ctx.buffer(name, factory)` a tensor made once (a window), `ctx.constant(value, name)` a
Python number — refused when the envelope lets the value vary.

An op that HOLDS a model (a detection op holding a predictor) lowers through the model's own lowering, registered
with `@lowers_model(PredictorClass)`; the op's lowering says which tensors it hands over and expects back, so a
predictor from one package and a detection op from another meet on that contract.

A compile may change the package's graph the way the package op does — `overrides: {flow.detect.op.device: cpu,
flow.detect.op.masks_output: ""}` — and the manifest records what was compiled.

Torpaido ships the lowerings of recordstream's own ops: `Scale`, `ToType`, `ToTensor`, `RenameField`,
`DropField`, `PutField` and `RecordContract`.

A step that survives pruning and has no lowering is refused where it stands:

```
torpaido: step 'picture/scale' (mydomain.processing.PictureScale) has no lowering for torch at graphs/cfg/model.yaml:41:3
  It writes scale. Leave it out of outputs, or register one with @lowers(PictureScale) in a module listed
  under the entry point group 'torpaido.lowerings'.
```

## Items and tensors

A record entry is a typed item; the program knows tensors and constants. An item splits into:

| part kind | examples | at compile time | in the artifact |
| --- | --- | --- | --- |
| **data** | an array, a list of numbers | a tensor | an input / output tensor |
| **scalar** | a sample rate, a level | a 0-d tensor when the envelope says `varies`, else a constant | an input tensor, or `specialized:` |
| **fact** | a layout, a scaling name, a class vocabulary, a string | a Python value | `specialized:` (inputs) / `constants:` (outputs) |

The default rule is introspected — a dataclass field by field, an `NDArrayItem` as its data plus its
declared attributes, a number as one scalar part, a string as one fact — and overridden per type where
the default is wrong:

```python
from torpaido.lowering import ItemRule, lowers_item


@lowers_item(Regions)
class RegionsRule(ItemRule):
    def split(self, item: Regions) -> tuple[dict, dict]:
        boxes = np.column_stack([item.freq_lower_edge_hz, item.freq_upper_edge_hz, item.time_start_s, item.time_end_s])
        return {"boxes": boxes, "labels": np.asarray(item.labels), "scores": np.asarray(item.scores)}, {"classes": None}

    def join(self, parts: dict, facts: dict) -> Regions: ...
```

The same rule serves twice — the compiler derives the interface from its `split`, the compiled op
feeds the engine with `split` and writes the record back with `join` — so the two cannot disagree.

## Backends

| backend | `backend:` | writes | runs with | extra |
| --- | --- | --- | --- | --- |
| `torpaido.backends.TorchScript` | `torchscript` | `model.pt` | `torch.jit.load` — the C++ runtimes that exist today | `[torch]` |
| `torpaido.backends.TorchExport` | `torch-export` | `program.pt2` | `torch.export.load` | `[torch]` |

`torch.export` is the hub: one lowered program, every backend writes it out. The port names travel
inside the artifact, so a runner built from the folder alone knows the input order.

## Parity

Every sample record goes through the Python graph and the artifact; each output entry is compared part by
part — arrays within `atol` / `rtol` (defaults `1e-5` / `1e-4`, per entry under `tolerances:`), integers
and facts exactly. A failing compile leaves `compiled/<name>.partial/` with `parity.yaml` and names the
first step whose lowering disagrees with its op:

```
CompilePackage: parity failed on record 1: predictions.boxes max |a - b| = 0.0312 (atol 1e-05, rtol 0.0001);
the first node that disagrees is 'raster/spectrogram' — the report is …/compiled/torchscript.partial/parity.yaml
```

## Installation

```bash
pip install "torpaido[torch] @ git+https://github.com/Gearlux/torpaido.git@main"
```

The `[torch]` extra holds the lowering language and the torch backends; the core (the IR, the front
end, the dataflow pass) needs no framework.

## Architecture

The decisions behind each mechanism — measured dataflow, the lowering registry, the item rules, the
`torch.export` hub, the envelope, parity — are recorded in [`docs/architecture.md`](docs/architecture.md);
the user guide is [`docs/compiling.md`](docs/compiling.md).

## License

MIT
