"""Fixtures: a model package written from the toy graph, and the sample records the compiler measures on."""

from pathlib import Path
from typing import List

import pytest
from matrainer.model_package import PackageModel
from recordstream import Record
from toy_graph import GRAPH, HALF, WEIGHTS_SLOT, sample_record, write_weights


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project: ``cfg/model.yaml`` including ``graphs/half.steps.yaml``, and the classifier's weights."""
    root = tmp_path / "project"
    (root / "graphs").mkdir(parents=True)
    (root / "cfg").mkdir()
    (root / "runs").mkdir()
    (root / "graphs" / "half.steps.yaml").write_text(HALF)
    (root / "cfg" / "model.yaml").write_text(GRAPH)
    write_weights(root / "runs" / "best.pt")
    return root


@pytest.fixture
def package(project: Path, tmp_path: Path) -> Path:
    """The toy graph as a model package (graphs and weights copied, two class names)."""
    return Path(
        PackageModel(
            graph=str(project / "cfg" / "model.yaml"),
            weights={WEIGHTS_SLOT: str(project / "runs" / "best.pt")},
            output_dir=str(tmp_path / "pkg"),
            class_names=["red", "green"],
            description="the toy classifier",
        ).run()
    )


@pytest.fixture
def samples() -> List[Record]:
    """Two source records of DIFFERENT picture sizes, so a dynamic dimension is exercised, same gain."""
    return [sample_record(4, 6, fill=200), sample_record(5, 3, fill=40)]
