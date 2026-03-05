# Torpedo

**Torpedo** is a high-performance compilation and optimization engine designed to transform functional **DataFlux** pipelines and PyTorch models into optimized, production-ready inference artifacts.

Part of the **Modular Quintet**: `LogFlow`, `Confluid`, `Liquify`, `DataFlux`, and `Torpedo`.

## 🚀 Key Features

-   **Selective Pruning:** Automatically strips non-inference sidecars (like testing metadata) to maximize throughput.
-   **Metadata Promotion:** Replaces heavy dictionary lookups with direct graph inputs and constants.
-   **Pluggable Backends:** First-class support for **TorchScript**, **ONNX**, and **TensorRT**.
-   **Unpacked Handover:** Eliminates boxing/unboxing overhead in the compiled computational path.
-   **Confluid Integration:** Fully configurable via YAML manifests for 100% reproducible deployments.

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
pip install torpedo
```

## 📄 License

MIT
