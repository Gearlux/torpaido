# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a1] — 2026-11-26

First public pre-release. The surface it ships:

### Compiling
- **`CompilePackage`** — a runnable (`recordstream run compile.yaml`) that compiles a model package (its inference
  graph, weights and class names) into a standalone folder: `compiled.yaml` (every input and output tensor with its
  dtype, shape and the record item it stands for; what was fixed; what was pruned; the parity verdict), the
  backend's files, a copy of `class_names.json` and `parity.yaml`. By default in `<package>/compiled/<name>/`.
- **Measured dataflow** — which record entry each step read and wrote is found by running each step on the sample
  records with one entry left out, never taken from a declaration; everything the chosen `outputs` never need is
  pruned.
- **The envelope** — what the artifact is compiled for: a part named with a value is fixed (a record with another
  value is refused), `varies` keeps it dynamic, and an unnamed part follows the samples.
- **Parity** — every sample record goes through the Python graph and the artifact, entry by entry, by each item's
  rule; a failing compile leaves `<name>.partial/` and names the first step whose lowering disagrees with its op.

### Running
- **`CompiledPackage`** — an op (`!class:torpaido.CompiledPackage {path: <compiled folder>}`) that runs a compiled
  folder on records, with no access to the package; a record outside the envelope is refused naming the part and both
  values.

### Backends
- **`backends.TorchScript`** (`model.pt`) and **`backends.TorchExport`** (`program.pt2`), behind the `[torch]` extra.
- **`backends.Onnx`** (`model.onnx`, run by onnxruntime), behind the `[onnx]` extra; a complex port is carried as real
  pairs, and an op the exporter cannot translate is refused naming the step and the lowering.

### Lowerings
- **`@lowers(OpClass)`**, **`@lowers_model(Class)`** and **`@lowers_item(ItemType)`** — one lowering per op class,
  per held model class and per item type, found by exact class. A domain package registers its own through the
  entry-point group `torpaido.lowerings`. The lowerings of recordstream's ops ship here.
