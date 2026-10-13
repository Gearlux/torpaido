"""Parity: the Python graph against the artifact, entry by entry, part by part, with a measured difference.

A comparison is by the entry's item rule — the same split the artifact's interface was derived from —
so an item of several parts (a box set's rows, labels and scores) is compared part by part. Arrays
are compared within ``atol`` / ``rtol`` (``max |a - b|`` reported), integers and facts exactly. A
tolerance per entry (``{"scores": {"atol": 0.5}}``) wins over the defaults.
"""

import dataclasses
import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from recordstream import Record

from torpaido.lowering.items import is_number, item_rule, to_python

__all__ = ["DEFAULT_TOLERANCES", "Difference", "compare", "tolerances_for"]

DEFAULT_TOLERANCES: Dict[str, float] = {"atol": 1e-5, "rtol": 1e-4}


@dataclasses.dataclass(frozen=True)
class Difference:
    """One disagreement: the entry, the part (blank for the whole entry) and what was measured."""

    entry: str
    part: str
    measure: str

    def as_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


def tolerances_for(entry: str, tolerances: Mapping[str, Any]) -> Dict[str, float]:
    """``{atol, rtol}`` for ``entry``: the defaults, the given defaults, then the entry's own."""
    merged = dict(DEFAULT_TOLERANCES)
    merged.update({key: float(value) for key, value in tolerances.items() if key in ("atol", "rtol")})
    own = tolerances.get(entry)
    if isinstance(own, Mapping):
        merged.update({key: float(value) for key, value in own.items() if key in ("atol", "rtol")})
    return merged


def _compare_part(entry: str, part: str, expected: Any, actual: Any, atol: float, rtol: float) -> List[Difference]:
    if is_number(expected) or getattr(expected, "ndim", None) == 0:
        a, b = to_python(expected), to_python(actual)
        if isinstance(a, (int, np.integer)) and not isinstance(a, bool) and isinstance(b, (int, np.integer)):
            return [] if a == b else [Difference(entry, part, f"{a} != {b}")]
        if is_number(a) and is_number(b):
            return [] if math.isclose(a, b, rel_tol=rtol, abs_tol=atol) else [Difference(entry, part, f"{a} != {b}")]
        return [] if a == b else [Difference(entry, part, f"{a!r} != {b!r}")]
    a, b = np.asarray(expected), np.asarray(actual)
    if a.shape != b.shape:
        return [Difference(entry, part, f"shape {a.shape} != {b.shape}")]
    if a.size == 0:
        return []
    if a.dtype.kind in "biu" and b.dtype.kind in "biu":
        # Integer arrays: exact, unless an absolute tolerance is given (a quantized picture may differ by a step).
        gap_int = np.abs(a.astype(np.int64) - b.astype(np.int64))
        if int(gap_int.max()) <= int(atol):
            return []
        over = int((gap_int > atol).sum())
        return [Difference(entry, part, f"{over} of {a.size} values differ by more than {atol:g}")]
    if a.dtype.kind in "fcbiu" and b.dtype.kind in "fcbiu":
        if np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
            return []
        gap = float(np.nanmax(np.abs(a.astype(np.complex128 if a.dtype.kind == "c" else np.float64) - b)))
        return [Difference(entry, part, f"max |a - b| = {gap} (atol {atol}, rtol {rtol})")]
    return [] if np.array_equal(a, b) else [Difference(entry, part, "values differ")]


def compare(
    reference: Record, produced: Record, entries: Sequence[str], tolerances: Mapping[str, Any]
) -> List[Difference]:
    """Every disagreement between ``reference`` and ``produced`` over ``entries`` — empty means parity."""
    differences: List[Difference] = []
    for entry in entries:
        if entry not in produced:
            differences.append(Difference(entry, "", "missing"))
            continue
        expected, actual = reference[entry], produced[entry]
        if type(expected) is not type(actual):
            differences.append(Difference(entry, "", f"type {type(expected).__name__} != {type(actual).__name__}"))
            continue
        rule = item_rule(type(expected))
        parts_a, facts_a = rule.split(expected)
        parts_b, facts_b = rule.split(actual)
        bounds = tolerances_for(entry, tolerances)
        for part, held in parts_a.items():
            if part not in parts_b:
                differences.append(Difference(entry, part, "missing"))
                continue
            differences.extend(_compare_part(entry, part, held, parts_b[part], bounds["atol"], bounds["rtol"]))
        for fact, held in facts_a.items():
            if facts_b.get(fact) != held:
                differences.append(Difference(entry, fact, f"{held!r} != {facts_b.get(fact)!r}"))
    return differences
