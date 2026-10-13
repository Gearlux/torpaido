"""The lowered program: the value graph as ONE ``nn.Module``, exported with ``torch.export`` — every backend's start.

:func:`build_program` takes a measured, pruned graph and the sample records and:

1. finds a lowering for every surviving op node (a missing one is refused with the step's location);
2. derives the INTERFACE from the graph's input values and the samples — each input entry split by its
   item rule into data parts (tensor inputs; a dimension that differs between the samples, or that the
   envelope lets vary, is dynamic), scalar parts (a tensor input when the envelope says ``varies``, else
   a constant written under ``specialized``) and facts (constants, under ``specialized``);
3. runs every lowering once, eagerly, on the first sample — the modules and buffers register, each
   lowering's writes are checked against the op's measured writes, and the per-node outputs are kept for
   the parity bisect;
4. exports the module with ``torch.export`` under the dynamic shapes the interface states.

The output ports are read off the eager outputs: dtype from the tensor, a dimension dynamic when it
differs between the samples.
"""

import dataclasses
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from recordstream import Record
from torch import Tensor, nn
from torch.export import Dim, ExportedProgram

from torpaido.ir import Graph, Node, Value
from torpaido.lowering.context import Context
from torpaido.lowering.items import ItemValue, is_number, item_rule, to_python
from torpaido.lowering.registry import GROUP, Lowering, discover, lowering_for

__all__ = ["Interface", "Port", "Program", "build_program", "dotted", "interface_arrays", "resolve_dotted"]

#: What the envelope may say about an input part instead of fixing it to a value.
VARIES = "varies"


def dotted(item_type: type) -> str:
    """A type as the manifest names it: ``recordstream.items.Image``, ``numpy.ndarray``, a builtin by its bare name."""
    if item_type.__module__ == "builtins":
        return item_type.__qualname__
    return f"{item_type.__module__}.{item_type.__qualname__}"


def resolve_dotted(name: str) -> type:
    """The type a manifest names (the inverse of :func:`dotted`)."""
    import builtins
    import importlib

    if "." not in name:
        return getattr(builtins, name)  # type: ignore[no-any-return]
    module, _, attr = name.rpartition(".")
    return getattr(importlib.import_module(module), attr)  # type: ignore[no-any-return]


@dataclasses.dataclass
class Port:
    """One tensor of the artifact's interface: ``<entry>.<part>``, its dtype, its shape (``None`` = dynamic), its
    item type."""

    name: str
    entry: str
    part: str
    dtype: str
    shape: List[Optional[int]]
    item: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry,
            "part": self.part,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "item": self.item,
        }

    @classmethod
    def from_dict(cls, name: str, info: Mapping[str, Any]) -> "Port":
        return cls(
            name, str(info["entry"]), str(info["part"]), str(info["dtype"]), list(info["shape"]), str(info["item"])
        )


@dataclasses.dataclass
class Interface:
    """What the artifact takes and gives: tensor ports in and out, what was fixed (inputs), what is constant
    (outputs)."""

    inputs: List[Port]
    outputs: List[Port]
    specialized: Dict[str, Any]  # input part/fact name -> the value the artifact was compiled for
    constants: Dict[str, Any]  # output part/fact name -> the value the artifact always gives
    envelope: Dict[str, Any]
    consumes: Dict[str, str]  # input entry -> its item type (dotted), the specialized entries included
    produces: Dict[str, str]  # output entry -> its item type (dotted)


def interface_arrays(interface: Interface, record: Record) -> Dict[str, np.ndarray]:
    """The artifact's input arrays for ``record``, by port name — and the checks the artifact's contract needs.

    Raises:
        ValueError: an input entry missing from the record; a part of another dtype, another rank or
            another fixed size; a specialized part or fact the record disagrees with.
    """
    arrays: Dict[str, np.ndarray] = {}
    split: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for entry, item_name in interface.consumes.items():
        if entry not in record:
            raise ValueError(f"the record has no entry {entry!r} (it has: {', '.join(map(str, record)) or 'nothing'})")
        split[entry] = item_rule(type(record[entry])).split(record[entry])
    for name, fixed in interface.specialized.items():
        entry, _, part = name.rpartition(".")
        parts, facts = split[entry]
        held = to_python(parts[part]) if part in parts else facts.get(part)
        if is_number(held) and is_number(fixed):
            agrees = math.isclose(float(held), float(fixed), rel_tol=1e-9, abs_tol=0.0)
        else:
            agrees = bool(held == fixed)
        if not agrees:
            raise ValueError(f"was compiled for {name} = {fixed!r}, and this record has {held!r}")
    for port in interface.inputs:
        parts, _ = split[port.entry]
        array = np.asarray(parts[port.part])
        if array.dtype.name != port.dtype:
            raise ValueError(f"{port.name} is {array.dtype.name}, and the compiled package takes {port.dtype}")
        if array.ndim != len(port.shape) or any(
            fixed is not None and fixed != size for fixed, size in zip(port.shape, array.shape)
        ):
            raise ValueError(f"{port.name} has shape {tuple(array.shape)}, and the compiled package takes {port.shape}")
        arrays[port.name] = array
    return arrays


class _InputSpec:
    """How ONE input entry's parts enter the program: which are tensors (and their dynamic dims), which are
    constants."""

    def __init__(self, value: Value) -> None:
        self.value = value
        self.tensor_parts: List[str] = []  # fed as tensors, in port order
        self.dynamic: Dict[str, List[bool]] = {}  # tensor part -> per-dimension "varies"
        self.constant_parts: Dict[str, Any] = {}  # fixed scalar parts, as Python numbers
        self.facts: Dict[str, Any] = {}


class LoweredGraph(nn.Module):
    """The value graph as a module: ``forward(*tensors) -> tuple(tensors)``, every node's lowering in schedule order.

    Built once eagerly (modules and buffers register, writes are checked, per-node outputs kept) and once
    more under ``torch.export``; both passes run the same code, so what the artifact does is what the
    eager pass measured.
    """

    def __init__(
        self, nodes: List[Node], lowerings: Dict[str, Lowering], specs: List[_InputSpec], outputs: List[Value]
    ) -> None:
        super().__init__()
        self._nodes = nodes
        self._lowerings = lowerings
        self._specs = specs
        self._outputs = outputs
        self._origins: Dict[int, str] = {}
        self._contexts = {
            node.name: Context(self, node.name, dotted(type(node.op)), node.where, self._origins) for node in nodes
        }
        #: Filled by every forward: node -> the values its lowering wrote (the eager pass keeps the reference).
        self.node_outputs: Dict[str, Dict[str, ItemValue]] = {}
        self.output_values: List[ItemValue] = []

    def forward(self, *tensors: Tensor) -> Tuple[Tensor, ...]:
        live: Dict[str, ItemValue] = {}
        self._origins.clear()
        feed = list(tensors)
        for spec in self._specs:
            parts: Dict[str, Any] = {}
            for part in spec.tensor_parts:
                tensor = feed.pop(0)
                parts[part] = tensor
                if tensor.ndim == 0:
                    self._origins[id(tensor)] = f"{spec.value.entry}.{part}"
            parts.update(spec.constant_parts)
            live[spec.value.name] = ItemValue(spec.value.item, parts, spec.facts)
        self.node_outputs = {}
        for node in self._nodes:
            reads = {entry: live[value.name] for entry, value in node.reads.items()}
            written = self._lowerings[node.name](node.op, reads, self._contexts[node.name])
            if set(written) != set(node.writes):
                raise ValueError(
                    f"torpaido: step {node.name!r} ({dotted(type(node.op))}): its lowering wrote {sorted(written)}, "
                    f"the op writes {sorted(node.writes)}{' at ' + node.where if node.where else ''}"
                )
            for entry, value in written.items():
                live[node.writes[entry].name] = value
            self.node_outputs[node.name] = dict(written)
        self.output_values = [live[value.name] for value in self._outputs]
        out: List[Tensor] = []
        for value in self.output_values:
            out.extend(part for part in value.parts.values() if isinstance(part, Tensor))
        return tuple(out)


@dataclasses.dataclass
class Program:
    """A compiled program before any backend wrote it out."""

    graph: Graph
    module: LoweredGraph
    exported: ExportedProgram
    interface: Interface
    example_inputs: Tuple[Tensor, ...]

    def tensors(self, record: Record) -> Tuple[Tensor, ...]:
        """The input tensors for ``record``, in port order (the interface's checks applied)."""
        arrays = interface_arrays(self.interface, record)
        return tuple(torch.as_tensor(arrays[port.name]) for port in self.interface.inputs)

    def eager(self, record: Record) -> Dict[str, Dict[str, Any]]:
        """Every node's written entries on ``record``, computed eagerly in torch and joined back to record items."""
        with torch.no_grad():
            self.module(*self.tensors(record))
        return {
            node: {entry: _join(value) for entry, value in written.items()}
            for node, written in self.module.node_outputs.items()
        }


def _join(value: ItemValue) -> Any:
    parts = {
        name: part.detach().cpu().numpy() if isinstance(part, Tensor) else part for name, part in value.parts.items()
    }
    return item_rule(value.item_type).join(parts, value.facts)


def _no_lowering(node: Node) -> ValueError:
    cls = type(node.op)
    at = f" at {node.where}" if node.where else ""
    writes = ", ".join(node.writes) or "nothing"
    return ValueError(
        f"torpaido: step {node.name!r} ({dotted(cls)}) has no lowering for torch{at}\n"
        f"  It writes {writes}. Leave it out of outputs, or register one with @lowers({cls.__name__}) in a module "
        f"listed under the entry point group {GROUP!r}."
    )


def _input_specs(
    graph: Graph, samples: List[Record], envelope: Dict[str, Any]
) -> Tuple[List[_InputSpec], Dict[str, Any]]:
    """The input specs and the ``specialized`` map, from the samples and the envelope; refusals for disagreements."""
    specs: List[_InputSpec] = []
    specialized: Dict[str, Any] = {}
    named: List[str] = []
    for value in graph.input_values:
        spec = _InputSpec(value)
        rule = item_rule(value.item)
        splits = [rule.split(record[value.entry]) for record in samples]
        parts0, facts0 = splits[0]
        for part, held in parts0.items():
            name = f"{value.entry}.{part}"
            named.append(name)
            wish = envelope.get(name)
            if is_number(held):
                if wish == VARIES:
                    spec.tensor_parts.append(part)
                    spec.dynamic[part] = []
                    continue
                for ordinal, (parts, _) in enumerate(splits[1:], 2):
                    if not same_number(parts[part], held):
                        raise ValueError(
                            f"torpaido: the input {name} is fixed, but the samples disagree: record 1 has "
                            f"{to_python(held)!r} and record {ordinal} has {to_python(parts[part])!r}\n"
                            f"  Let it vary (envelope: {name}: varies) or compile from samples that agree."
                        )
                if wish is not None and not same_number(wish, held):
                    raise ValueError(
                        f"torpaido: the envelope fixes {name} to {wish!r}, but record 1 has {to_python(held)!r}"
                    )
                spec.constant_parts[part] = to_python(held)
                specialized[name] = to_python(held)
            else:
                array = np.asarray(held)
                shapes = [np.asarray(parts[part]).shape for parts, _ in splits]
                if any(len(shape) != array.ndim for shape in shapes):
                    raise ValueError(
                        f"torpaido: the input {name} has rank {array.ndim} on record 1 and another rank on another "
                        f"sample ({[list(shape) for shape in shapes]}) — a tensor's rank cannot vary"
                    )
                if wish is not None and wish != VARIES:
                    raise ValueError(f"torpaido: the envelope says {name}: {wish!r}; a data part may only be 'varies'")
                spec.tensor_parts.append(part)
                spec.dynamic[part] = [
                    wish == VARIES or len({shape[axis] for shape in shapes}) > 1 for axis in range(array.ndim)
                ]
        for fact, held in facts0.items():
            name = f"{value.entry}.{fact}"
            named.append(name)
            for ordinal, (_, facts) in enumerate(splits[1:], 2):
                if facts.get(fact) != held:
                    raise ValueError(
                        f"torpaido: the input {name} is a fact of the artifact, but the samples disagree: record 1 has "
                        f"{held!r} and record {ordinal} has {facts.get(fact)!r}"
                    )
            wish = envelope.get(name)
            if wish is not None and wish != held:
                raise ValueError(f"torpaido: the envelope fixes {name} to {wish!r}, but record 1 has {held!r}")
            spec.facts[fact] = held
            specialized[name] = held
        specs.append(spec)
    unknown = [name for name in envelope if name not in named]
    if unknown:
        raise ValueError(
            f"torpaido: the envelope names {unknown[0]!r}, which is not an input part ({', '.join(sorted(named))})"
        )
    return specs, specialized


def same_number(a: Any, b: Any) -> bool:
    """Two scalars equal as numbers (an int written in YAML against the sample's float counts as equal)."""
    a, b = to_python(a), to_python(b)
    if is_number(a) and is_number(b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
    return bool(a == b)


def _dynamic_shapes(specs: List[_InputSpec]) -> Tuple[Any, ...]:
    """The ``dynamic_shapes`` of ``torch.export`` for ``forward(*tensors)``: one spec per tensor, nested once
    because the varargs tuple is itself the one argument export sees."""
    shapes: List[Any] = []
    for spec in specs:
        for part in spec.tensor_parts:
            dynamic = spec.dynamic[part]
            shapes.append({axis: Dim.AUTO for axis, varies in enumerate(dynamic) if varies} or None)
    return (tuple(shapes),)


def _tensor_dtype(tensor: Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def build_program(graph: Graph, samples: Sequence[Record], envelope: Optional[Mapping[str, Any]] = None) -> Program:
    """Lower the measured, pruned ``graph`` into one exported program (see the module docstring).

    Args:
        graph: A graph :func:`~torpaido.dataflow.measure` filled and :meth:`~torpaido.ir.Graph.prune_sidecars` pruned.
        samples: The records it was measured on (the first is the eager pass's input; all shape the interface).
        envelope: ``{"<entry>.<part>": "varies" | value}`` — what varies and what is fixed (module docstring).

    Raises:
        ValueError: a surviving node without a lowering; an envelope entry that names no input part or
            disagrees with the samples; samples that disagree on a fixed part; a lowering that writes other
            entries than its op; a lowering that needs a varying value as a constant.
    """
    discover()
    records = list(samples)
    nodes = graph.op_nodes()
    lowerings: Dict[str, Lowering] = {}
    for node in nodes:
        lowering = lowering_for(node.op)
        if lowering is None:
            raise _no_lowering(node)
        lowerings[node.name] = lowering
    wish = {str(key): value for key, value in (envelope or {}).items()}
    specs, specialized = _input_specs(graph, records, wish)
    module = LoweredGraph(nodes, lowerings, specs, graph.output_values)
    module.eval()

    inputs: List[Port] = []
    for spec in specs:
        parts, _ = item_rule(spec.value.item).split(records[0][spec.value.entry])
        for part in spec.tensor_parts:
            array = np.asarray(parts[part])
            shape: List[Optional[int]] = [
                None if varies else int(size) for varies, size in zip(spec.dynamic[part], array.shape)
            ]
            inputs.append(
                Port(
                    f"{spec.value.entry}.{part}",
                    spec.value.entry,
                    part,
                    array.dtype.name,
                    shape,
                    dotted(spec.value.item),
                )
            )
    consumes = {value.entry: dotted(value.item) for value in graph.input_values}
    produces = {value.entry: dotted(value.item) for value in graph.output_values}
    interface = Interface(inputs, [], specialized, {}, dict(wish), consumes, produces)

    program_inputs = [_feed(interface, record) for record in records]
    with torch.no_grad():
        eager_outputs = [module(*tensors) for tensors in program_inputs]
    # The output ports: dtype from the eager tensors, a dimension dynamic when the samples disagree on it.
    # ``module.output_values`` holds the LAST sample's values; every sample gives the same parts and facts,
    # since the graph's writes were measured equal on all of them.
    outputs, constants = _name_outputs(graph.output_values, list(module.output_values), eager_outputs)
    interface.outputs = outputs
    interface.constants = constants

    exported = torch.export.export(module, program_inputs[0], dynamic_shapes=_dynamic_shapes(specs), strict=False)
    return Program(graph, module, exported, interface, program_inputs[0])


def _feed(interface: Interface, record: Record) -> Tuple[Tensor, ...]:
    arrays = interface_arrays(interface, record)
    return tuple(torch.as_tensor(arrays[port.name]) for port in interface.inputs)


def _name_outputs(
    values: List[Value], first: List[ItemValue], eager_outputs: List[Tuple[Tensor, ...]]
) -> Tuple[List[Port], Dict[str, Any]]:
    """The output ports (entry.part, dtype, a shape whose dimensions the samples disagree on are dynamic) and the
    constants."""
    ports: List[Port] = []
    constants: Dict[str, Any] = {}
    index = 0
    for value, computed in zip(values, first):
        for part, held in computed.parts.items():
            if isinstance(held, Tensor):
                shapes = [tuple(outputs[index].shape) for outputs in eager_outputs]
                shape: List[Optional[int]] = [
                    None if len({shape[axis] for shape in shapes}) > 1 else int(shapes[0][axis])
                    for axis in range(len(shapes[0]))
                ]
                ports.append(
                    Port(
                        f"{value.entry}.{part}",
                        value.entry,
                        part,
                        _tensor_dtype(held),
                        shape,
                        dotted(computed.item_type),
                    )
                )
                index += 1
            else:
                constants[f"{value.entry}.{part}"] = to_python(held)
        for fact, held in computed.facts.items():
            constants[f"{value.entry}.{fact}"] = held
    return ports, constants
