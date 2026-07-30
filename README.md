# Torpedo

**Torpedo** is a high-performance compilation and optimization engine designed to transform functional **RecordStream** pipelines and PyTorch models into optimized, production-ready inference artifacts.

Part of the **Modular Quintet**: `Loggair`, `Confluid`, `Liquify`, `RecordStream`, and `Torpedo`.

## 🚀 Key Features

-   **Selective Pruning:** Automatically strips non-inference sidecars (like testing metadata) to maximize throughput.
-   **Metadata Promotion:** Replaces heavy dictionary lookups with direct graph inputs and constants.
-   **Pluggable Backends:** First-class support for **TorchScript**, **ONNX**, and **TensorRT**.
-   **Unpacked Handover:** Eliminates boxing/unboxing overhead in the compiled computational path.
-   **Confluid Integration:** Fully configurable via YAML manifests for 100% reproducible deployments.

## 🎯 Design Goals & Requirements

### Compilation Engine
- **Passive Inference Pruning:** Implement reverse-dependency analysis to strip non-inference sidecars (e.g. testing metadata) from binaries.
- **Metadata Promotion:** Convert required dynamic metadata into graph inputs and static metadata into constants.
- **Unpacked Handover:** Replace boxed Python records with direct tensor-to-tensor edges in the compiled graph.

### Portability
- **Pluggable Backends:** Support TorchScript, ONNX, and TensorRT via a unified `Backend` protocol.
- **Fusion:** Enable the fusion of an entire RecordStream pipeline and Torch model into a single deployment artifact.

### Precision
- **Numeric Parity:** Ensure compiled outputs match the floating-point results of the source Python implementation.
- **Symmetry:** Compiled artifacts must be verifiable against their source Confluid manifests.

## 🛠 The front end — compilation starts from a GRAPH

Implemented today: `torpaido.frontend`, which turns a record pipeline's step graph into the
torpaido IR. Every edge is explicit on both sides, so the mapping is close to an identity:

| pipeline step | IR node |
| --- | --- |
| `step.from_` / `step.merge_from` / `step.bind` | `Node.inputs` |
| `step.name` | `Node.outputs` |

```python
from recordstream.flow import parse_flow
from torpaido import graph_from_steps

steps, outputs = parse_flow(document["flow"], document.get("outputs", ""))
graph = graph_from_steps(steps, outputs, name="spectrogram")

graph.prune_sidecars()      # reverse-dependency analysis: drop what the output never needs
```

The input shape is the **graph**, never a flattened op sequence. `prune_sidecars` walks
`node.inputs` backwards from the graph outputs; a flattened list re-encodes those edges as
imperative mutations of a per-record cell store and therefore carries no `inputs` at all — the
pruning pass could not run on it. Linearization is a job for a backend, over this IR, on the way
to codegen.

## 🛠 Usage (Preview — not yet implemented)

```python
from torpaido import Forge

forge = Forge(pipeline, backend="onnx")
compiled = forge.build(input_shape=(1, 3, 224, 224))
forge.save("production_model.onnx")
```

## 🔧 Installation

```bash
pip install git+https://github.com/Gearlux/torpaido.git@main
```

## 📄 License

MIT
