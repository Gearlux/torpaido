"""Items <-> tensors: how ONE record item type splits into parts and facts, and joins back.

A record entry is a typed item; a tensor program knows only tensors and constants. An :class:`ItemRule`
says, for one item type, how an item splits into

* **parts** — what becomes a tensor: an array (``data``), a list of numbers, a number (a scalar part,
  a 0-d tensor when the envelope lets it vary, a constant otherwise);
* **facts** — what never varies inside one compiled artifact: a string, ``None``, a layout, a class
  vocabulary; written in the manifest, never in the graph.

The default rule is INTROSPECTED from the type — a dataclass field by field, an ``NDArrayItem`` as its
data plus its declared attributes, a number as one scalar part, a string as one fact — and a domain
package overrides it with :func:`lowers_item` where the default is wrong (a list of regions is one
``[N, 4]`` tensor, not four). The same rule serves twice: at compile time its parts are the tensors
the lowerings compute on; at run time the compiled op calls ``split`` on the record's items to feed
the engine and ``join`` on the engine's outputs to write the record back — one definition, so the two
faces cannot disagree.

:class:`ItemValue` is the compile-time carrier a lowering receives and returns: the item type, its
parts (tensors or Python numbers) and its facts.
"""

import dataclasses
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeGuard,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
from recordstream import NDArrayItem

__all__ = ["ItemRule", "ItemValue", "Parts", "Facts", "item_rule", "lowers_item", "is_number", "to_python", "Watched"]

Parts = Dict[str, Any]
Facts = Dict[str, Any]


def is_number(value: Any) -> TypeGuard[Union[int, float]]:
    """A scalar that becomes a 0-d tensor: a Python or numpy number — never a ``bool`` (a flag is a fact).

    A type guard, so a checker lets a guarded value through ``float()``; numpy scalars pass the guard too
    (they support ``float()``), which is what the callers do with them."""
    if isinstance(value, bool) or isinstance(value, np.bool_):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def is_tensor(value: Any) -> bool:
    """A torch tensor, recognised without importing torch (the run-time face may not have it loaded)."""
    return type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach")


def to_python(value: Any) -> Any:
    """A numpy scalar or a 0-d array/tensor as the Python number it holds; anything else unchanged."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if is_tensor(value) and value.ndim == 0:
        return value.item()
    return value


def _as_array(value: Any) -> np.ndarray:
    if is_tensor(value):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


class Watched(dict):  # type: ignore[type-arg]
    """A parts or facts mapping that records which keys a lowering LOOKED AT, into the ``seen`` set it shares.

    A key is seen when it is read (``[]``, ``get``) or when the mapping is walked (``items``, ``values``, ``keys``,
    iteration, a ``dict(...)`` copy — what ``ItemValue.replace`` does to carry an item forward, so every part it
    carries counts as read); ``in`` alone is not a look. The compiler hands an INPUT entry's parts and facts to the
    lowerings this way once, and what was never seen is not part of the artifact's interface.
    """

    __slots__ = ("seen",)

    def __init__(self, values: Mapping[str, Any], seen: Set[str]) -> None:
        super().__init__(values)
        self.seen = seen

    def __getitem__(self, key: str) -> Any:
        self.seen.add(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        self.seen.add(key)
        return super().get(key, default)

    def keys(self) -> Any:
        self.seen.update(super().keys())
        return super().keys()

    def values(self) -> Any:
        self.seen.update(super().keys())
        return super().values()

    def items(self) -> Any:
        self.seen.update(super().keys())
        return super().items()

    def __iter__(self) -> Any:
        self.seen.update(super().keys())
        return super().__iter__()

    def copy(self) -> Dict[str, Any]:
        self.seen.update(super().keys())
        return dict(super().items())


class ItemValue:
    """The compile-time form of one record entry: its item type, its parts (tensors or constants) and its facts.

    ``value.data`` is the ``data`` part, ``value.value`` the ``value`` part (a number) or fact (a string);
    ``replace(name=...)`` returns a copy with one part or fact changed, whichever ``name`` is. A :class:`Watched`
    mapping is kept as it is (it records the reads); any other mapping is copied.
    """

    __slots__ = ("item_type", "parts", "facts")

    def __init__(
        self, item_type: type, parts: Optional[Mapping[str, Any]] = None, facts: Optional[Mapping[str, Any]] = None
    ) -> None:
        self.item_type = item_type
        self.parts: Parts = parts if isinstance(parts, Watched) else dict(parts or {})
        self.facts: Facts = facts if isinstance(facts, Watched) else dict(facts or {})

    @property
    def data(self) -> Any:
        return self.parts["data"]

    @property
    def value(self) -> Any:
        return self.parts["value"] if "value" in self.parts else self.facts["value"]

    def replace(self, **changes: Any) -> "ItemValue":
        parts, facts = dict(self.parts), dict(self.facts)
        for name, value in changes.items():
            if name in parts:
                parts[name] = value
            elif name in facts:
                facts[name] = value
            else:
                raise KeyError(
                    f"{self.item_type.__name__}: no part or fact named {name!r} "
                    f"(parts: {sorted(parts)}, facts: {sorted(facts)})"
                )
        return ItemValue(self.item_type, parts, facts)

    def __repr__(self) -> str:
        return f"ItemValue({self.item_type.__name__}, parts={sorted(self.parts)}, facts={self.facts})"


class ItemRule:
    """How one item type splits into parts and facts and joins back; subclass and register with :func:`lowers_item`.

    A rule may also say how two items of its type are COMPARED for parity (:meth:`compare`): the default is part by
    part, which is wrong for a set whose rows have no order of their own (a detector's boxes) — such a rule matches
    the rows and reports what is unmatched.
    """

    item_type: type = object

    def split(self, item: Any) -> Tuple[Parts, Facts]:
        raise NotImplementedError(f"{type(self).__name__} must implement split(item) -> (parts, facts)")

    def join(self, parts: Parts, facts: Facts) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement join(parts, facts) -> item")

    def compare(self, expected: Any, actual: Any, tolerances: Mapping[str, Any]) -> Optional[List[Tuple[str, str]]]:
        """``[(part, measure), ...]`` of the disagreements between two items, or ``None`` to compare part by part.

        ``tolerances`` is the entry's own mapping (``atol``, ``rtol`` and whatever keys the rule reads).
        """
        return None


class ArrayRule(ItemRule):
    """A plain array: one ``data`` part."""

    item_type = np.ndarray

    def split(self, item: Any) -> Tuple[Parts, Facts]:
        return {"data": np.asarray(item)}, {}

    def join(self, parts: Parts, facts: Facts) -> Any:
        return _as_array(parts["data"])


class TensorRule(ItemRule):
    """A live torch tensor entry: one ``data`` part; joined back as a tensor."""

    def split(self, item: Any) -> Tuple[Parts, Facts]:
        return {"data": _as_array(item)}, {}

    def join(self, parts: Parts, facts: Facts) -> Any:
        import torch

        return torch.as_tensor(_as_array(parts["data"]))


class NDArrayItemRule(ItemRule):
    """An ``NDArrayItem``: its payload as ``data``, each declared attribute a scalar part (a number) or a fact."""

    def __init__(self, item_type: type) -> None:
        self.item_type = item_type

    def split(self, item: Any) -> Tuple[Parts, Facts]:
        parts: Parts = {"data": np.asarray(item).view(np.ndarray)}
        facts: Facts = {}
        for attr in getattr(self.item_type, "_item_attrs", ()):
            held = getattr(item, attr, None)
            (parts if is_number(held) else facts)[attr] = held
        return parts, facts

    def join(self, parts: Parts, facts: Facts) -> Any:
        attrs = {name: to_python(value) for name, value in parts.items() if name != "data"}
        attrs.update(facts)
        return self.item_type(_as_array(parts["data"]), **attrs)


class ScalarRule(ItemRule):
    """A number entry: one ``value`` part, joined back as the Python type it was."""

    def __init__(self, item_type: type) -> None:
        self.item_type = item_type

    def split(self, item: Any) -> Tuple[Parts, Facts]:
        return {"value": item}, {}

    def join(self, parts: Parts, facts: Facts) -> Any:
        value = to_python(parts["value"])
        return self.item_type(value) if self.item_type in (int, float, bool) else value


class FactRule(ItemRule):
    """A string or ``None`` entry: one ``value`` fact."""

    def __init__(self, item_type: type) -> None:
        self.item_type = item_type

    def split(self, item: Any) -> Tuple[Parts, Facts]:
        return {}, {"value": item}

    def join(self, parts: Parts, facts: Facts) -> Any:
        return facts["value"]


def _is_number_list(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) > 0 and all(is_number(v) for v in value)


def _wants_list(annotation: Any) -> bool:
    """Whether a dataclass field is declared as a list (``List[float]``, ``Optional[List[float]]``) — an array part
    then joins back as one."""
    origin = get_origin(annotation)
    if origin is Union:
        return any(_wants_list(arg) for arg in get_args(annotation) if arg is not type(None))
    return origin is list or annotation is list


class DataclassRule(ItemRule):
    """A dataclass item, field by field: arrays and number lists are data parts, numbers scalar parts, the rest
    facts."""

    def __init__(self, item_type: type) -> None:
        self.item_type = item_type
        try:
            self._hints = get_type_hints(item_type)
        except Exception:  # a forward reference the module cannot resolve: join falls back to the value as is
            self._hints = {}

    def split(self, item: Any) -> Tuple[Parts, Facts]:
        parts: Parts = {}
        facts: Facts = {}
        for field in dataclasses.fields(item):
            held = getattr(item, field.name)
            if isinstance(held, np.ndarray) or is_tensor(held):
                parts[field.name] = _as_array(held)
            elif _is_number_list(held):
                parts[field.name] = np.asarray(held)
            elif is_number(held):
                parts[field.name] = held
            else:
                facts[field.name] = held
        return parts, facts

    def join(self, parts: Parts, facts: Facts) -> Any:
        values: Dict[str, Any] = dict(facts)
        for name, held in parts.items():
            if _wants_list(self._hints.get(name)):
                values[name] = _as_array(held).tolist()
            elif is_number(held) or getattr(held, "ndim", None) == 0:
                number = to_python(held)
                wanted = self._hints.get(name)
                values[name] = wanted(number) if wanted in (int, float) else number
            else:
                values[name] = _as_array(held)
        return self.item_type(**values)


#: The rules a domain package registered (:func:`lowers_item`), by exact item type.
RULES: Dict[type, ItemRule] = {}


def lowers_item(item_type: type) -> Callable[[Type[ItemRule]], Type[ItemRule]]:
    """Register an :class:`ItemRule` subclass for ``item_type``, replacing the introspected default.

    Example::

        @lowers_item(Regions)
        class RegionsRule(ItemRule):
            def split(self, item): ...
            def join(self, parts, facts): ...
    """

    def decorator(rule_class: Type[ItemRule]) -> Type[ItemRule]:
        rule = rule_class()
        rule.item_type = item_type
        RULES[item_type] = rule
        return rule_class

    return decorator


def item_rule(item_type: type) -> ItemRule:
    """The rule for ``item_type``: a registered one, else the introspected default for its kind.

    Raises:
        ValueError: no default fits (a class that is not an array, a number, a string or a dataclass).
    """
    registered = RULES.get(item_type)
    if registered is not None:
        return registered
    if issubclass(item_type, NDArrayItem):
        return NDArrayItemRule(item_type)
    if issubclass(item_type, np.ndarray):
        return ArrayRule()
    if item_type.__module__.split(".", 1)[0] == "torch":
        return TensorRule()
    if item_type is bool or issubclass(item_type, (int, float, np.integer, np.floating)):
        return ScalarRule(item_type)
    if issubclass(item_type, (str, bytes)) or item_type is type(None):
        return FactRule(item_type)
    if dataclasses.is_dataclass(item_type):
        return DataclassRule(item_type)
    raise ValueError(
        f"torpaido: no item rule for {item_type.__module__}.{item_type.__qualname__} — it is not an array, a number, "
        f"a string or a dataclass; register one with @lowers_item({item_type.__name__})"
    )
