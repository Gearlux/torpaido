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


def discover() -> None:
    """Import every lowering module: torpaido's own, then every module listed under :data:`GROUP`.

    Registration is an import side effect, so this is called before a compile and before a compiled
    package runs (its item rules may live in a domain package too). Repeated calls are cheap.
    """
    import torpaido.lowering.recordstream  # noqa: F401  — registration is the import's side effect

    load_configurables(GROUP)
