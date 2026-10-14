"""The lowering registry: one torch function per op CLASS, found by exact class, discovered by entry point.

A lowering is a plain function ``(op, reads, ctx) -> writes``: the configured op (its settings are read
off it), the values its measured reads hold (``{entry: ItemValue}``), the :class:`~torpaido.lowering.context.Context`
(modules, buffers, constants), and the values it writes (``{entry: ItemValue}``, exactly the entries the
op writes in Python). It is registered on the op's class with :func:`lowers`; the op never imports the
compiler, and the function lives wherever its author wants — torpaido ships recordstream's, a domain
package ships its own under a module named in the entry-point group ``torpaido.lowerings``.

The lookup is by EXACT class: a subclass may compute something else, and inheriting a lowering would
make the parity check the only thing standing between it and a wrong artifact.
"""

from typing import Any, Callable, Dict, Optional

from confluid import load_configurables

__all__ = [
    "DISCOVERED",
    "GROUP",
    "LOWERINGS",
    "MODEL_LOWERINGS",
    "Lowering",
    "ModelLowering",
    "discover",
    "lowering_for",
    "lowers",
    "lowers_model",
    "model_lowering_for",
    "no_model_lowering",
]

#: The entry-point group a package lists its lowering modules under.
GROUP = "torpaido.lowerings"

Lowering = Callable[[Any, Dict[str, Any], Any], Dict[str, Any]]

#: Op class -> its lowering.
LOWERINGS: Dict[type, Lowering] = {}


def lowers(op_class: type) -> Callable[[Lowering], Lowering]:
    """Register the decorated function as the lowering of ``op_class``.

    Raises:
        ValueError: the class already has one — two lowerings for one op would make the artifact depend
            on import order.

    Example::

        @lowers(Magnitude)
        def lower_magnitude(op, reads, ctx):
            spec = reads[op.field]
            return {op.field: spec.replace(data=20.0 * torch.log10(spec.data + op.eps))}
    """

    def decorator(function: Lowering) -> Lowering:
        existing = LOWERINGS.get(op_class)
        if existing is not None:
            raise ValueError(
                f"torpaido: {op_class.__name__} already has a lowering "
                f"({existing.__module__}.{existing.__qualname__}); a class has exactly one"
            )
        LOWERINGS[op_class] = function
        return function

    return decorator


def lowering_for(op: Any) -> Optional[Lowering]:
    """The lowering registered for ``op``'s exact class, or ``None``."""
    return LOWERINGS.get(type(op))


#: A lowering of an object an op HOLDS (a predictor, a model wrapper) rather than of the op itself:
#: ``(held, *tensors, ctx) -> tensors``. The op's lowering states the contract its held object must meet
#: (which tensors in, which out); this registry only finds the function by the object's exact class.
ModelLowering = Callable[..., Any]

#: Held-object class -> its lowering.
MODEL_LOWERINGS: Dict[type, ModelLowering] = {}


def lowers_model(model_class: type) -> Callable[[ModelLowering], ModelLowering]:
    """Register the decorated function as the lowering of the held objects of ``model_class``.

    A record op that holds a model (a detector op holding a predictor) lowers itself through the model's
    lowering; the op's lowering documents the tensors it hands over and expects back. Exact class, one
    per class, like :func:`lowers`.

    Raises:
        ValueError: the class already has one.
    """

    def decorator(function: ModelLowering) -> ModelLowering:
        existing = MODEL_LOWERINGS.get(model_class)
        if existing is not None:
            raise ValueError(
                f"torpaido: {model_class.__name__} already has a model lowering "
                f"({existing.__module__}.{existing.__qualname__}); a class has exactly one"
            )
        MODEL_LOWERINGS[model_class] = function
        return function

    return decorator


def model_lowering_for(model: Any) -> Optional[ModelLowering]:
    """The model lowering registered for ``model``'s exact class, or ``None``."""
    return MODEL_LOWERINGS.get(type(model))


#: What the last :func:`discover` found under :data:`GROUP`: entry name -> the module it imported, or the exception
#: its import raised. Kept so a refusal can say what WAS found (:func:`no_model_lowering`).
DISCOVERED: Dict[str, Any] = {}


def discover() -> None:
    """Import every lowering module: torpaido's own, then every module listed under :data:`GROUP`.

    Registration is an import side effect, so this is called before a compile and before a compiled
    package runs (its item rules may live in a domain package too). Repeated calls are cheap. What the
    group gave — each entry's module, or the exception its import raised — stays in :data:`DISCOVERED`.
    """
    import torpaido.lowering.recordstream  # noqa: F401  — registration is the import's side effect

    DISCOVERED.clear()
    DISCOVERED.update(load_configurables(GROUP))


def no_model_lowering(held: Any, where: str = "") -> ValueError:
    """The refusal for a held object (a predictor) without a model lowering, saying what WAS found.

    A lowering that exists but is not registered — the package that defines the object was installed before it
    declared its entry point, or without its lowering extra — looks exactly like a lowering that was never written,
    so the message lists the lowering modules the entry-point group loaded, the entries that failed to import, and,
    when none of the loaded modules comes from the package defining the object, the reinstall.

    Args:
        held: The object without a lowering.
        where: What holds it, ending in a space (``"step 'detect' (DetectionModelPredict) "``), or blank.
    """
    kind = type(held)
    package = kind.__module__.split(".", 1)[0]
    loaded = {name: module for name, module in DISCOVERED.items() if not isinstance(module, Exception)}
    failed = {name: exc for name, exc in DISCOVERED.items() if isinstance(exc, Exception)}
    found = ", ".join(f"{name} ({module.__name__})" for name, module in sorted(loaded.items())) or "none"
    lines = [
        f"torpaido: {where}holds a {kind.__module__}.{kind.__qualname__}, which has no model lowering.",
        f"  Lowering modules found under the entry point group {GROUP!r}: {found}"
        + (
            "; failed to import: "
            + ", ".join(f"{name} ({type(exc).__name__}: {exc})" for name, exc in sorted(failed.items()))
            if failed
            else ""
        )
        + ".",
    ]
    if all(module.__name__.split(".", 1)[0] != package for module in loaded.values()):
        lines.append(
            f"  None comes from {package!r}, the package that defines it: install that package with its lowering "
            "extra, or reinstall it so its entry point is registered (an editable install registers entry points at "
            "install time)."
        )
    lines.append(f"  Or register one with @lowers_model({kind.__name__}).")
    return ValueError("\n".join(lines))
