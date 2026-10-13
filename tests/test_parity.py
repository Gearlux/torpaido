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
    assert "the first node that disagrees is 'pre/bright'" in str(refused.value)
    partial = package / "compiled" / "ts.partial"
    assert partial.is_dir() and not (package / "compiled" / "ts").exists()
    report = yaml.safe_load((partial / PARITY_FILE).read_text())
    assert report["status"] == "failed"
    assert report["first_disagreeing_node"] == "pre/bright"
    assert report["records"][0]["differences"][0]["entry"] == "scores"
    assert (partial / COMPILED_FILE).exists() is False
