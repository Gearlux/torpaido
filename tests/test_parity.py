"""Parity is measured entry by entry on the sample records, and a failure names the FIRST node that disagrees."""

from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
import yaml
from recordstream import Image, Label, Record
from toy_graph import Brighten

import torpaido.lowering.registry as registry
from torpaido.compile import CompilePackage
from torpaido.manifest import COMPILED_FILE, PARITY_FILE
from torpaido.parity import Difference, compare


def test_compare_reports_each_differing_part_with_its_measure() -> None:
    reference: Record = {"scores": np.array([1.0, 2.0]), "label": Label(1), "image": Image(np.zeros((2, 2, 3)))}
    produced: Record = {"scores": np.array([1.0, 2.5]), "label": Label(0), "image": Image(np.zeros((2, 2, 3)))}

    differences = compare(reference, produced, ["scores", "label", "image"], {"atol": 1e-5, "rtol": 1e-4})

    assert differences == [
        Difference(entry="scores", part="data", measure="max |a - b| = 0.5 (atol 1e-05, rtol 0.0001)"),
        Difference(entry="label", part="value", measure="1 != 0"),
    ]


def test_compare_names_a_missing_entry_and_a_shape_mismatch() -> None:
    reference: Record = {"scores": np.array([1.0, 2.0]), "label": Label(1)}
    produced: Record = {"scores": np.array([1.0, 2.0, 3.0])}

    differences = compare(reference, produced, ["scores", "label"], {"atol": 1e-5, "rtol": 1e-4})

    assert differences == [
        Difference(entry="scores", part="data", measure="shape (2,) != (3,)"),
        Difference(entry="label", part="", measure="missing"),
    ]


def test_a_tolerance_per_entry_wins_over_the_default() -> None:
    reference: Record = {"scores": np.array([1.0, 2.0])}
    produced: Record = {"scores": np.array([1.0, 2.4])}

    assert compare(reference, produced, ["scores"], {"atol": 1e-5, "scores": {"atol": 0.5}}) == []


def test_a_wrong_lowering_fails_the_compile_and_names_the_first_disagreeing_node(
    package: Path, samples: List[Record], monkeypatch: pytest.MonkeyPatch
) -> None:
    def wrong(op: Brighten, reads: dict, ctx: object) -> dict:
        image = reads[op.image_field]
        return {op.image_field: image.replace(data=image.data.to(torch.float32) * (reads[op.gain_field].value + 1.0))}

    monkeypatch.setitem(registry.LOWERINGS, Brighten, wrong)

    with pytest.raises(ValueError, match="parity failed on record 1: scores.data max") as refused:
        CompilePackage(package=str(package), name="ts", samples=samples).run()
    assert "the first node that disagrees is 'pre/bright' (image.data max |a - b| = " in str(refused.value)
    partial = package / "compiled" / "ts.partial"
    assert partial.is_dir() and not (package / "compiled" / "ts").exists()
    report = yaml.safe_load((partial / PARITY_FILE).read_text())
    assert report["status"] == "failed"
    assert report["first_disagreeing_node"] == "pre/bright"
    (at_node,) = report["first_node_differences"]  # that node's own measured gap, so the reader does not re-derive it
    assert at_node["entry"] == "image" and at_node["part"] == "data" and at_node["measure"].startswith("max |a - b| = ")
    assert report["records"][0]["differences"][0]["entry"] == "scores"
    assert (partial / COMPILED_FILE).exists() is False


def test_a_rule_s_own_comparison_wins_over_the_part_by_part_one() -> None:
    from dataclasses import dataclass
    from typing import Any, List, Mapping, Optional, Tuple

    from torpaido.lowering import ItemRule, lowers_item

    @dataclass
    class Bag:  # rows in no order of their own
        values: List[float]

    @lowers_item(Bag)
    class BagRule(ItemRule):
        def split(self, item: Bag) -> tuple:
            return {"values": np.asarray(item.values)}, {}

        def join(self, parts: dict, facts: dict) -> Bag:
            return Bag(list(np.asarray(parts["values"])))

        def compare(self, expected: Any, actual: Any, tolerances: Mapping[str, Any]) -> Optional[List[Tuple[str, str]]]:
            missing = sorted(set(expected.values) - set(actual.values))
            allowed = int(tolerances.get("unmatched", 0))
            return [] if len(missing) <= allowed else [("values", f"{len(missing)} values unmatched: {missing}")]

    reference: Record = {"bag": Bag([1.0, 2.0, 3.0])}
    assert compare(reference, {"bag": Bag([3.0, 1.0, 2.0])}, ["bag"], {}) == []  # the same set, another order
    assert compare(reference, {"bag": Bag([1.0, 2.0])}, ["bag"], {}) == [
        Difference(entry="bag", part="values", measure="1 values unmatched: [3.0]")
    ]
    assert compare(reference, {"bag": Bag([1.0, 2.0])}, ["bag"], {"bag": {"unmatched": 1}}) == []
