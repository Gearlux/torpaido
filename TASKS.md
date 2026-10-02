# torpaido — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

- [ ] **Core Interfaces:** Define `Backend` and `CompiledModule` protocols. @high @architecture
- [ ] **TorchScript Plugin:** Port and refine the reference compiler logic. @high @feature
- [ ] **Metadata Logic:** Implement static and dynamic promotion rules. @medium @feature
- [ ] **ONNX Plugin:** Implement the universal ONNX exporter. @high @feature
- [ ] **TensorRT Plugin:** Implement hardware-specific NVIDIA optimization. @low @performance
- [ ] **RecordStream Bridge:** Add `.forge()` fluent API to the RecordStream engine. @medium @integration
- [ ] **Docs and comments describe the old surface** @docs @small — the three documents disagree about the project's name and its configuration vocabulary:
  - `torpaido/README.md:1,3,5,13,28` and `torpaido/RATIONALE.md:1,4,11,16,21,24,35` — call it "Torpedo", say "Modular Quintet" (README) and "Modular Quartet" (RATIONALE and `pyproject.toml:4`), and talk about "YAML manifests" for a workspace that retired manifests.
  - `torpaido/docs/architecture.md:24-27` (Context, "Every mandate in this project's `AGENTS.md`…") — names **Unpacked Handover** as an `AGENTS.md` mandate, but `AGENTS.md` never had one (it appears only in `README.md:12,20` and `RATIONALE.md`); reword the sentence or add the mandate.
