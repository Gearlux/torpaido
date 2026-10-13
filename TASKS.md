# torpaido — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

- [ ] **ONNX backend:** `torpaido.backends.Onnx` — `torch.onnx.export(program.exported, dynamo=True)` on the same `Program`, `onnx.checker` on the file, a runner on `onnxruntime`, the `[onnx]` extra (`onnx`, `onnxruntime`, `onnxscript`); its test skipped when the extra is absent. @high @feature
- [ ] **`ConvertToImage`: the other colormaps and a 3-D array** @low @feature — the lowering renders the `gray` colormap of a 2-D map; a matplotlib colormap is a 256-entry lookup table (a buffer indexed by the quantized map) and a 3-D array the op's channel coercion. Both are refused by name today. @2026-10-13
- [ ] **A draw is refused by name:** a `RandomApply`, and a `Transform` whose `get_params` draws per record, are refused with "step 'x' draws a parameter per record; a compiled graph is deterministic — remove the step" before lowering is attempted (today they are refused as "no lowering"). @medium @feature
- [ ] **The measurement's cost on a real package:** ablation reruns a step once per input entry per sample (a model step: one model call per entry). Measure it on the signal-detection package; if it hurts, measure on the first record and verify the pruned graph against the full graph on the rest (architecture §2 names the allowed shortcut). @medium @performance
- [ ] **Runtime branches:** a `requires:` gate or a per-record condition compiles through `torch.cond` (ONNX `If`); until then such a step is refused with its name. @low @feature
- [ ] **`varies` per dimension** (`image.data: [varies, varies, 3]`) when a package needs a fixed channel count beside free spatial dimensions; today `varies` frees every dimension and the default frees the dimensions the samples disagree on. @low @feature
- [ ] **Output dimensions on one sample:** `varies` on an input cannot show on an output when there is one sample (one shape); derive output dynamism from the exported program's symbolic shapes instead of measuring over samples. @low @feature
- [ ] **More backends behind their extras:** AOTInductor (`.so` for libtorch C++, the modern replacement of TorchScript), ExecuTorch, TensorRT, CoreML, OpenVINO. @low @feature
