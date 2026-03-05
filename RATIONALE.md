# Torpedo: Rationale & Architectural Design

## Executive Summary
**Torpedo** is the high-performance compilation and optimization engine for the Modular Quartet. It transforms functional **DataFlux** pipelines and PyTorch models into optimized, production-ready inference artifacts (TorchScript, ONNX, TensorRT).

---

## Core Architectural Pillars

### 1. Selective Inference Pruning (Passive Pruning)
Unlike the reference implementation which compiles full object triplets, Torpedo performs reverse-dependency analysis to isolate the **Computational Kernel**.
- **The Path:** Only operations contributing to the final inference tensor are traced.
- **The Result:** Testing-only sidecars (bounding boxes, metric collectors) are surgically stripped from the binary, maximizing throughput.

### 2. Metadata Promotion
Torpedo avoids passing heavy Python dictionaries into compiled modules.
- **Static Promotion:** Constants in the `metadata` are baked into the graph as initializers.
- **Dynamic Promotion:** Required variable metadata (e.g., `samplerate`) is promoted to secondary graph inputs.

### 3. "Unpacked" Handover Pattern
At runtime, DataFlux uses "Boxed" immutable samples for safety. Torpedo "unpacks" the box during compilation, creating direct tensor-to-tensor links between nodes. This eliminates the overhead of dictionary lookups and tuple creation in the hot path.

### 4. Pluggable Backend Architecture
Torpedo is designed as an orchestrator for multiple inference engines:
- **TorchScript Backend:** Refined port of the reference logic for native PyTorch environments.
- **ONNX Backend:** Universal bridge for cross-platform and cloud deployment.
- **TensorRT Backend:** Hardware-specific acceleration for NVIDIA environments.

---

## Design Goals
- **Research Purity:** Keep the Python code functional and clean.
- **Production Performance:** Single-command path from Python to optimized C++-ready binary.
- **Full Traceability:** Integrate with **LogFlow** to monitor the compilation lifecycle.
- **Symmetry:** Every compiled artifact must be reconstructible and verifiable against its source manifest.
