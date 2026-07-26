# torpaido — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

- [ ] **Core Interfaces:** Define `Backend` and `CompiledModule` protocols. @high @architecture
- [ ] **TorchScript Plugin:** Port and refine the reference compiler logic. @high @feature
- [ ] **Pruning Engine:** Implement reverse-dependency analysis for selective tracing. @medium @performance
- [ ] **Metadata Logic:** Implement static and dynamic promotion rules. @medium @feature
- [ ] **ONNX Plugin:** Implement the universal ONNX exporter. @high @feature
- [ ] **TensorRT Plugin:** Implement hardware-specific NVIDIA optimization. @low @performance
- [ ] **SampleFlux Bridge:** Add `.forge()` fluent API to the SampleFlux engine. @medium @integration
- [ ] **Torpaido (Compiler):** Implement the high-performance compilation engine for SampleFlux pipelines (TorchScript, ONNX, TensorRT). @high @performance
