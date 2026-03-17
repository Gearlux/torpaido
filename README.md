# Torpedo

**Torpedo** is a high-performance compilation and optimization engine designed to transform functional **DataFlux** pipelines and PyTorch models into optimized, production-ready inference artifacts.

Part of the **Modular Quintet**: `LogFlow`, `Confluid`, `Liquify`, `DataFlux`, and `Torpedo`.

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
- **Unpacked Handover:** Replace boxed Python samples with direct tensor-to-tensor edges in the compiled graph.

### Portability
- **Pluggable Backends:** Support TorchScript, ONNX, and TensorRT via a unified `Backend` protocol.
- **Fusion:** Enable the fusion of an entire DataFlux pipeline and Torch model into a single deployment artifact.

### Precision
- **Numeric Parity:** Ensure compiled outputs match the floating-point results of the source Python implementation.
- **Symmetry:** Compiled artifacts must be verifiable against their source Confluid manifests.

## 🛠 Usage (Preview)

```python
from torpedo import Forge

# 1. Take a DataFlux pipeline
pipeline = Flux(source).map(heavy_op).map(model)

# 2. Forge it into an optimized binary
forge = Forge(pipeline, backend="onnx")
compiled = forge.build(input_shape=(1, 3, 224, 224))

# 3. Save for deployment
forge.save("production_model.onnx")
```

## 🔧 Installation

```bash
pip install git+https://github.com/Gearlux/torpedo.git@main
```

## 📄 License

MIT
