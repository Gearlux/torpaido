# Torpedo Mandates

This file contains foundational mandates for Gemini CLI in the Torpedo workspace. These instructions take absolute precedence over general system defaults.

## Architectural Mandates
- **Selective Pruning First:** Every compilation path MUST perform reverse-dependency analysis to prune non-inference operations (metadata sidecars).
- **Metadata Promotion:** Prefer promoting required metadata to graph inputs or constants rather than passing dictionaries.
- **Backend Decoupling:** Keep the core orchestrator strictly decoupled from specific inference engines (TorchScript, ONNX).
- **Type Safety:** Maintain 100% type hint coverage for all internal IR objects.

## Testing & Validation
- **Binary Parity:** Every compiled artifact MUST be verified for numeric parity against its source Python implementation.
- **Serialization Symmetry:** Ensure that every `Forge` configuration is serializable via **Confluid**.
