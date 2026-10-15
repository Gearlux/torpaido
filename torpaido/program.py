"""The lowered program: the value graph as ONE ``nn.Module``, exported with ``torch.export`` — every backend's start.

:func:`build_program` takes a measured, pruned graph and the sample records and:

1. finds a lowering for every surviving op node (a missing one is refused with the step's location);
2. measures what the lowerings READ of each input entry: one probe run on the first sample hands every
   lowering the entry's parts and facts through a recording mapping; a part or fact no lowering looked at
   is not part of the interface (a capture's derived duration, its SNR fields), whatever the samples say
   about it;
3. derives the INTERFACE from the read parts and the samples — each input entry split by its item rule
   into data parts (tensor inputs; a dimension that differs between the samples, or that the envelope lets
   vary, is dynamic), scalar parts (a 0-d tensor input when the envelope says ``varies`` OR the samples
   disagree; else a constant written under ``specialized``) and facts (constants under ``specialized``,
   which the samples must share);
4. runs every lowering once more, eagerly, on every sample — the modules and buffers register, each
   lowering's writes are checked against the op's measured writes, and the per-node outputs are kept for
   the parity bisect;
5. exports the module with ``torch.export`` under the dynamic shapes the interface states.

The output ports are read off the eager outputs: dtype from the tensor, a dimension dynamic when it
differs between the samples.
"""

import dataclasses
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.fx.traceback as fx_traceback
from recordstream import Record
from torch import Tensor, nn
from torch.export import Dim, ExportedProgram

from torpaido.ir import Graph, Node, Value
from torpaido.lowering.context import Context
from torpaido.lowering.items import ItemValue, Watched, is_number, item_rule, to_python
from torpaido.lowering.registry import GROUP, Lowering, discover, lowering_for

__all__ = [
    "Interface",
    "Port",
    "Program",
    "STEP_ANNOTATION",
    "build_program",
    "dotted",
    "interface_arrays",
    "resolve_dotted",
]

#: What the envelope may say about an input part instead of fixing it to a value.
VARIES = "varies"
#: The key under which every fx node of an exported program names the step whose lowering wrote it
#: (``node.meta["custom"][STEP_ANNOTATION]``, kept by an export run under ``fx_traceback.preserve_node_meta()``) —
#: what a backend's refusal of an op it cannot translate is located by.
STEP_ANNOTATION = "torpaido_step"


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
        #: Filled by every forward: input value name -> the parts and facts some lowering looked at.
        self.reads: Dict[str, Set[str]] = {}

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
            seen = self.reads.setdefault(spec.value.name, set())
            live[spec.value.name] = ItemValue(spec.value.item, Watched(parts, seen), Watched(spec.facts, seen))
        self.node_outputs = {}
        for node in self._nodes:
            reads = {entry: live[value.name] for entry, value in node.reads.items()}
            with fx_traceback.annotate({STEP_ANNOTATION: node.name}):  # eagerly and under a trace: a no-op
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
    graph: Graph, samples: List[Record], envelope: Dict[str, Any], reads: Optional[Mapping[str, Set[str]]]
) -> Tuple[List[_InputSpec], Dict[str, Any]]:
    """The input specs and the ``specialized`` map, from the samples, the envelope and what the lowerings read.

    ``reads`` is ``None`` for the PROBE pass: every part enters, scalars and facts as the first sample has them, no
    agreement demanded — the pass only measures which parts the lowerings look at. With ``reads`` (input value name →
    the parts and facts seen), only those enter: a data part as a tensor (a dimension the samples disagree on, or
    that the envelope frees, dynamic); a scalar part as a 0-d tensor when the envelope says ``varies`` or the samples
    disagree, else fixed at their value; a fact fixed, the samples having to agree. An envelope value is checked
    against every sample. Refusals name the part and both values.
    """
    specs: List[_InputSpec] = []
    specialized: Dict[str, Any] = {}
    named: List[str] = []
    read_names: List[str] = []
    for value in graph.input_values:
        spec = _InputSpec(value)
        rule = item_rule(value.item)
        splits = [rule.split(record[value.entry]) for record in samples]
        parts0, facts0 = splits[0]
        seen = None if reads is None else reads.get(value.name, set())
        for part, held in parts0.items():
            name = f"{value.entry}.{part}"
            named.append(name)
            if seen is not None and part not in seen:
                continue
            read_names.append(name)
            wish = envelope.get(name)
            if is_number(held):
                if seen is None:  # the probe: the first sample's value, as a constant
                    spec.constant_parts[part] = to_python(held)
                    continue
                if wish is not None and wish != VARIES:
                    for ordinal, (parts, _) in enumerate(splits, 1):
                        if not same_number(wish, parts.get(part)):
                            raise ValueError(
                                f"torpaido: the envelope fixes {name} to {wish!r}, but record {ordinal} has "
                                f"{to_python(parts.get(part))!r}"
                            )
                    spec.constant_parts[part] = to_python(held)
                    specialized[name] = to_python(held)
                    continue
                agree = all(same_number(parts.get(part), held) for parts, _ in splits[1:])
                if wish == VARIES or not agree:
                    spec.tensor_parts.append(part)
                    spec.dynamic[part] = []
                    continue
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
            if seen is not None and fact not in seen:
                continue
            read_names.append(name)
            if seen is None:
                spec.facts[fact] = held
                continue
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
    if reads is not None:
        for key in envelope:
            if key not in named:
                raise ValueError(
                    f"torpaido: the envelope names {key!r}, which is not an input part ({', '.join(sorted(named))})"
                )
            if key not in read_names:
                raise ValueError(
                    f"torpaido: the envelope names {key!r}, which no step of the artifact reads "
                    f"(it reads {', '.join(sorted(read_names))})"
                )
    return specs, specialized


def _probe_tensors(specs: List[_InputSpec], record: Record) -> Tuple[Tensor, ...]:
    """The probe pass's input tensors for ``record``: every tensor part of every spec, in order, unchecked."""
    tensors: List[Tensor] = []
    for spec in specs:
        parts, _ = item_rule(spec.value.item).split(record[spec.value.entry])
        tensors.extend(torch.as_tensor(np.asarray(parts[part])) for part in spec.tensor_parts)
    return tuple(tensors)


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
    # The probe: every part enters, and the recording mappings say which ones the lowerings looked at.
    probe_specs, _ = _input_specs(graph, records, wish, None)
    probe = LoweredGraph(nodes, lowerings, probe_specs, graph.output_values)
    probe.eval()
    with torch.no_grad():
        probe(*_probe_tensors(probe_specs, records[0]))
    specs, specialized = _input_specs(graph, records, wish, probe.reads)
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
