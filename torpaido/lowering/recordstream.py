"""The lowerings of recordstream's own ops — the ones every graph uses, shipped with the compiler.

Each mirrors the numpy op it lowers line for line (the op's module is the reference; the parity check
is the proof). An op that writes nothing (a drop, a contract check) lowers to nothing: its reads were
measured, its effect on the value graph is already in the edges.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from recordstream import Image, Label
from recordstream.ops.contract import RecordContract
from recordstream.ops.image import ConvertToImage
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


@lowers(ConvertToImage)
def lower_convert_to_image(op: ConvertToImage, reads: Values, ctx: Context) -> Values:
    """A 2-D map through the ``gray`` colormap, quantized as the op quantizes, flipped and sized as the op sizes.

    Mirrors ``_render_rgb`` → ``_sized``: ``vmin`` / ``vmax`` (else the map's own finite extremes) become 0 and
    255 by truncation; the resize is a bilinear filter with antialiasing, which is what PIL's BILINEAR resize
    computes (measured within 1 of 255 on up- and down-scaling). Only the ``gray`` colormap of a 2-D map is
    lowered: a matplotlib colormap is a lookup table and a 3-D array another path, both refused by name.
    """
    if op.colormap != "gray":
        raise ValueError(
            f"torpaido: step {ctx.node!r} (ConvertToImage) uses the colormap {op.colormap!r}, which is not lowered "
            "yet — only 'gray' is"
        )
    entry = op.field or next(iter(reads))
    source = reads[entry]
    data = source.data
    if data.is_complex():
        raise ValueError(
            f"torpaido: step {ctx.node!r} (ConvertToImage) renders {entry!r}, which is complex — "
            "take its magnitude first"
        )
    data = data.squeeze()
    if data.ndim != 2:
        raise ValueError(
            f"torpaido: step {ctx.node!r} (ConvertToImage) renders {entry!r} of rank {data.ndim}; "
            "only a 2-D map is lowered yet"
        )
    values = data.to(torch.float32)
    finite = torch.isfinite(values)
    if op.vmin is not None:
        low: Any = float(op.vmin)
    else:
        low = torch.where(finite, values, torch.full_like(values, float("inf"))).amin()
    if op.vmax is not None:
        high: Any = float(op.vmax)
    else:
        high = torch.where(finite, values, torch.full_like(values, float("-inf"))).amax()
    filled = torch.where(torch.isnan(values), torch.zeros_like(values) + low, values)
    filled = torch.where(values == float("inf"), torch.zeros_like(values) + high, filled)
    filled = torch.where(values == float("-inf"), torch.zeros_like(values) + low, filled)
    span = high - low
    norm = torch.clamp((filled - low) / span, 0.0, 1.0) * 255.0
    gray = torch.where(torch.as_tensor(span > 0), norm, torch.zeros_like(norm)).to(torch.uint8)
    if op.flip_vertical:
        gray = torch.flip(gray, dims=[0])
    target = _target_size(op, int(gray.shape[0]), int(gray.shape[1]))
    if target is not None and target != (int(gray.shape[1]), int(gray.shape[0])):
        width, height = target
        resized = torch.nn.functional.interpolate(
            gray.to(torch.float32)[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        gray = resized[0, 0].round().clamp(0.0, 255.0).to(torch.uint8)
    rgb = gray.unsqueeze(-1).expand(gray.shape[0], gray.shape[1], 3).contiguous()
    return {op.output: ItemValue(Image, {"data": rgb}, {"layout": "HWC"})}


def _target_size(op: ConvertToImage, height_px: int, width_px: int) -> Optional[Tuple[int, int]]:
    """``(width, height)`` the op's sizing rule asks for, or ``None`` for the pixels as they are (the op's table)."""
    width, height = op.width, op.height
    if width is None and height is None:
        return None
    if not ((width or 0) > 0 or (height or 0) > 0):
        longest = max(height_px, width_px)
        if op.max_size <= 0 or longest <= op.max_size:
            return None
        scale = op.max_size / longest
        return max(1, int(width_px * scale)), max(1, int(height_px * scale))
    if width is not None and width > 0 and height is None:
        return width, height_px
    if height is not None and height > 0 and width is None:
        return width_px, height
    if width is not None and width > 0 and height is not None and height > 0:
        return width, height
    if width is not None and width > 0:
        return width, max(1, int(round(height_px * width / width_px)))
    assert height is not None
    return max(1, int(round(width_px * height / height_px))), height
