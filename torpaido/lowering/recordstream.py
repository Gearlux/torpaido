"""The lowerings of recordstream's own ops — the ones every graph uses, shipped with the compiler.

Each mirrors the numpy op it lowers line for line (the op's module is the reference; the parity check
is the proof). An op that writes nothing (a drop, a contract check) lowers to nothing: its reads were
measured, its effect on the value graph is already in the edges.
"""

from typing import Any, Dict

import torch
from recordstream import Label
from recordstream.ops.contract import RecordContract
from recordstream.ops.numpy import Scale, ToType
from recordstream.ops.structure import DropField, PutField, RenameField
from recordstream.ops.torch import ToTensor

from torpaido.lowering.context import Context
from torpaido.lowering.items import ItemValue, is_number
from torpaido.lowering.registry import lowers

Values = Dict[str, ItemValue]


@lowers(Scale)
def lower_scale(op: Scale, reads: Values, ctx: Context) -> Values:
    """``(x - low) / (high - low) * span + target_min`` in the op's own float type rule."""
    out: Values = {}
    span = float(op.target_max) - float(op.target_min)
    for entry, value in reads.items():
        x = value.data
        low, high = op.source_min, op.source_max
        if low is None or high is None:
            info = torch.iinfo(x.dtype)  # an integer type: the op refuses a blank bound on a float
            low = info.min if low is None else low
            high = info.max if high is None else high
        out_type = x.dtype if x.is_floating_point() else torch.float32
        scaled = (x.to(out_type) - float(low)) / (float(high) - float(low)) * span + float(op.target_min)
        out[entry] = value.replace(data=scaled.to(out_type))
    return out


@lowers(ToType)
def lower_to_type(op: ToType, reads: Values, ctx: Context) -> Values:
    dtype = getattr(torch, op.dtype)
    return {entry: value.replace(data=value.data.to(dtype)) for entry, value in reads.items()}


@lowers(ToTensor)
def lower_to_tensor(op: ToTensor, reads: Values, ctx: Context) -> Values:
    """HWC -> CHW (a 2-D array gains a leading channel axis), the element type untouched."""
    (entry,) = reads  # the one array-bearing entry the op resolved
    data = reads[entry].data
    if data.ndim == 3:
        data = data.permute(2, 0, 1)
    elif data.ndim == 2:
        data = data.unsqueeze(0)
    return {op.output or entry: ItemValue(torch.Tensor, {"data": data})}


@lowers(RenameField)
def lower_rename_field(op: RenameField, reads: Values, ctx: Context) -> Values:
    return {op.dst: reads[op.src]}


@lowers(DropField)
def lower_drop_field(op: DropField, reads: Values, ctx: Context) -> Values:
    return {}


@lowers(RecordContract)
def lower_record_contract(op: RecordContract, reads: Values, ctx: Context) -> Values:
    return {}


@lowers(PutField)
def lower_put_field(op: PutField, reads: Values, ctx: Context) -> Values:
    """A constant ``Label``: a number is a scalar part, anything else a fact. A producer is refused."""
    value: Any = op.value
    if callable(value):
        raise ValueError(
            f"torpaido: step {ctx.node!r} (PutField) produces its value per record ({type(value).__name__}); "
            "a compiled graph writes a constant — set a number or a name"
        )
    if is_number(value):
        return {op.key: ItemValue(Label, {"value": value}, {"classes": None})}
    return {op.key: ItemValue(Label, {}, {"value": value, "classes": None})}
