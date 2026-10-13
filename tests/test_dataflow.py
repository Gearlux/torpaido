"""Dataflow is MEASURED: which entry each node read and wrote, on real records — never trusted from declarations."""

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest
from recordstream import Image, Record
from recordstream.flow import parse_flow
from recordstream.ops.numpy import Scale
from toy_graph import Brighten, Fickle, Mean, live_steps, sample_record

from torpaido.dataflow import measure

#: What ``live_steps`` returns: the parsed steps and the yielded step.
Toy = Tuple[List[Any], str]


@pytest.fixture
def toy(project: Path) -> Toy:
    return live_steps(str(project / "runs" / "best.pt"))


def test_each_node_reads_what_its_op_used_and_writes_what_it_changed(toy: Toy, samples: List[Record]) -> None:
    graph = measure(*toy, samples)

    reads = {name: sorted(node.reads) for name, node in graph.nodes.items() if node.op is not None}
    writes = {name: sorted(node.writes) for name, node in graph.nodes.items() if node.op is not None}
    assert reads == {
        "pre/scale": ["image"],
        "pre/bright": ["gain", "image"],
        "level": ["image"],
        "stamp": [],
        "classify": ["image"],
        "done": [],
    }
    assert writes == {
        "pre/scale": ["image"],
        "pre/bright": ["image"],
        "level": ["level"],
        "stamp": ["note"],
        "classify": ["label", "scores"],
        "done": [],
    }


def test_a_value_names_its_producer_and_its_item_type(toy: Toy, samples: List[Record]) -> None:
    graph = measure(*toy, samples)

    assert graph.nodes["pre/scale"].reads["image"].name == "source.image"
    assert graph.nodes["pre/bright"].reads["image"].name == "pre/scale.image"
    assert graph.nodes["classify"].reads["image"].name == "pre/bright.image"
    assert graph.nodes["classify"].reads["image"].item is Image
    assert graph.nodes["classify"].writes["scores"].item is np.ndarray
    assert graph.nodes["pre/bright"].reads["gain"].item is float


def test_the_default_outputs_are_the_entries_the_last_writing_node_wrote(toy: Toy, samples: List[Record]) -> None:
    # ``done`` (a DropField) writes nothing, so the default looks one node back.
    graph = measure(*toy, samples)

    assert [value.name for value in graph.output_values] == ["classify.scores", "classify.label"]


def test_pruning_drops_everything_the_outputs_never_need(toy: Toy, samples: List[Record]) -> None:
    graph = measure(*toy, samples)

    dropped = graph.prune_sidecars()

    assert dropped == ["level", "stamp", "done"]
    assert [name for name, node in graph.nodes.items() if node.op is not None] == [
        "pre/scale",
        "pre/bright",
        "classify",
    ]
    assert [value.name for value in graph.input_values] == ["source.image", "source.gain"]


def test_the_outputs_may_be_chosen(toy: Toy, samples: List[Record]) -> None:
    graph = measure(*toy, samples, outputs=["level"])

    assert graph.prune_sidecars() == ["stamp", "classify", "done"]
    assert [value.name for value in graph.output_values] == ["level.level"]


def test_an_output_nothing_writes_is_refused_naming_what_is_written(toy: Toy, samples: List[Record]) -> None:
    with pytest.raises(
        ValueError, match=r"outputs name 'nothing', which no step writes .*image.*label.*level.*note.*scores"
    ):
        measure(*toy, samples, outputs=["nothing"])


def test_a_read_that_depends_on_the_record_is_refused_naming_the_node_and_both_records() -> None:
    steps, result = parse_flow({"level": {"op": Mean()}, "fickle": {"op": Fickle(), "from": "level"}}, "fickle")
    records = [sample_record(fill=255), sample_record(fill=40)]  # a mean above 100, then one below

    with pytest.raises(
        ValueError, match=r"step 'fickle' reads \['level', 'tag'\] on record 1 and \['level'\] on record 2"
    ):
        measure(steps, result, records)


def test_a_branch_reads_its_own_producer_not_the_latest_write() -> None:
    flow = {
        "start": {"op": Scale(field="image")},
        "left": {"op": Brighten(), "from": "start"},
        "right": {"op": Mean(), "from": "start"},
    }
    graph = measure(*parse_flow(flow, "right"), [sample_record()])

    assert graph.nodes["right"].reads["image"].name == "start.image"
    assert graph.nodes["left"].reads["image"].name == "start.image"


def test_measuring_needs_at_least_one_record(toy: Toy) -> None:
    with pytest.raises(ValueError, match="at least one sample record"):
        measure(*toy, [])


def test_a_refusal_carries_the_location_of_the_step(toy: Toy, samples: List[Record]) -> None:
    steps, result = parse_flow({"level": {"op": Mean()}, "fickle": {"op": Fickle(), "from": "level"}}, "fickle")
    records = [sample_record(fill=255), sample_record(fill=40)]

    with pytest.raises(ValueError, match=r"step 'fickle' reads .* at model.yaml:7:3"):
        measure(steps, result, records, locations={"fickle": "model.yaml:7:3"})
