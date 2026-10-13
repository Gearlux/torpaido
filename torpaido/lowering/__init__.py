"""Lowerings: how an op becomes torch code, and how a record item becomes tensors.

* :func:`lowers` registers a function ``(op, reads, ctx) -> writes`` for an op CLASS; :func:`lowering_for`
  finds it; :func:`discover` imports every lowering module (torpaido's own and the entry-point group
  ``torpaido.lowerings``). :func:`lowers_model` / :func:`model_lowering_for` do the same for an object an
  op HOLDS (a predictor), whose tensor contract the op's own lowering states.
* :class:`ItemValue` is what a lowering receives and returns per entry; :class:`ItemRule` /
  :func:`item_rule` / :func:`lowers_item` say how an item type splits into parts and facts.
* :class:`Context` is what a lowering may ask the program for: a module, a buffer, a constant.

The context imports torch; the registry and the items do not, so the compiled op can split and join
records in a process that only runs an engine.
"""

from typing import TYPE_CHECKING, Any

from torpaido.lowering.items import Facts, ItemRule, ItemValue, Parts, is_number, item_rule, lowers_item, to_python
from torpaido.lowering.registry import (
    GROUP,
    LOWERINGS,
    MODEL_LOWERINGS,
    Lowering,
    ModelLowering,
    discover,
    lowering_for,
    lowers,
    lowers_model,
    model_lowering_for,
)

if TYPE_CHECKING:
    from torpaido.lowering.context import Context

__all__ = [
    "GROUP",
    "LOWERINGS",
    "Context",
    "Facts",
    "ItemRule",
    "ItemValue",
    "Lowering",
    "MODEL_LOWERINGS",
    "ModelLowering",
    "Parts",
    "discover",
    "is_number",
    "item_rule",
    "lowering_for",
    "lowers",
    "lowers_item",
    "lowers_model",
    "model_lowering_for",
    "to_python",
]


def __getattr__(name: str) -> Any:  # PEP 562: torch is imported only when a lowering runs
    if name == "Context":
        from torpaido.lowering.context import Context

        return Context
    raise AttributeError(f"module 'torpaido.lowering' has no attribute {name!r}")
