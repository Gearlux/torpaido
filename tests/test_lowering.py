"""Lowerings are found by the op's class, discovered through an entry-point group, and refused with a location."""

from pathlib import Path
from typing import Any, List

import pytest
from recordstream import Record
from recordstream.ops.numpy import Scale
from toy_graph import Brighten, NoLower, Tile, live_steps, lower_brighten, sample_record

import torpaido.lowering.registry as registry
from torpaido.dataflow import measure
from torpaido.lowering import ItemValue, discover, lowering_for, lowers
from torpaido.program import build_program


def test_a_lowering_is_found_by_the_exact_op_class() -> None:
    class Brighter(Brighten):
        pass

    assert lowering_for(Brighten()) is lower_brighten
    assert lowering_for(Brighter()) is None


def test_registering_a_second_lowering_for_a_class_is_refused() -> None:
    class Twice:
        pass

    lowers(Twice)(lambda op, reads, ctx: {})
    with pytest.raises(ValueError, match="Twice already has a lowering"):
        lowers(Twice)(lambda op, reads, ctx: {})


def test_discover_imports_the_entry_point_group_and_torpaido_s_own_lowerings(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: List[str] = []

    def fake_load(group: str) -> dict:
        asked.append(group)
        return {}

    monkeypatch.setattr(registry, "load_configurables", fake_load)

    discover()

    assert asked == ["torpaido.lowerings"]
    assert lowering_for(Scale()) is not None


def test_a_surviving_node_without_a_lowering_is_refused_at_its_location(project: Path, samples: List[Record]) -> None:
    steps, result = live_steps(str(project / "runs" / "best.pt"), extra=NoLower())
    graph = measure(steps, result, samples, outputs=["extra"], locations={"extra": "cfg/model.yaml:20:5"})
    graph.prune_sidecars()

    with pytest.raises(ValueError) as refused:
        build_program(graph, samples)
    assert str(refused.value) == (
        "torpaido: step 'extra' (toy_graph.NoLower) has no lowering for torch at cfg/model.yaml:20:5\n"
        "  It writes extra. Leave it out of outputs, or register one with @lowers(NoLower) in a module listed "
        "under the entry point group 'torpaido.lowerings'."
    )


def test_a_pruned_node_without_a_lowering_is_not_refused(project: Path, samples: List[Record]) -> None:
    steps, result = live_steps(str(project / "runs" / "best.pt"), extra=NoLower())
    graph = measure(steps, result, samples, outputs=["scores", "label"])  # `extra` feeds neither
    assert "extra" in graph.prune_sidecars()

    program = build_program(graph, samples)

    assert [port.name for port in program.interface.outputs] == ["scores.data", "label.value"]


def test_a_varying_value_where_a_constant_is_needed_is_refused_naming_the_dependency(samples: List[Record]) -> None:
    from recordstream.flow import parse_flow

    steps, result = parse_flow({"tile": {"op": Tile()}}, "tile")
    records = [{**record, "count": 2} for record in samples]
    graph = measure(steps, result, records, locations={"tile": "cfg/model.yaml:3:3"})
    graph.prune_sidecars()

    with pytest.raises(ValueError) as refused:
        build_program(graph, records, envelope={"count.value": "varies"})
    assert str(refused.value) == (
        "torpaido: step 'tile' (toy_graph.Tile) needs 'count' as a constant, but it depends on count.value, "
        "which the envelope lets vary at cfg/model.yaml:3:3\n"
        "  Fix it in the envelope (count.value: <value>) or leave it out so the samples' value is used."
    )


def test_a_fixed_value_reaches_the_lowering_as_a_constant(samples: List[Record]) -> None:
    from recordstream.flow import parse_flow

    steps, result = parse_flow({"tile": {"op": Tile()}}, "tile")
    records = [{**samples[0], "count": 2}]  # one 4 x 6 picture: every dimension fixed
    graph = measure(steps, result, records)
    graph.prune_sidecars()

    program = build_program(graph, records, envelope={"count.value": 2})

    assert program.interface.specialized["count.value"] == 2
    assert [port.name for port in program.interface.inputs] == ["image.data"]
    assert program.interface.outputs[0].shape == [4, 12, 3]


def test_a_lowering_that_writes_other_entries_than_its_op_is_refused(project: Path, samples: List[Record]) -> None:
    class Sloppy(NoLower):
        pass

    lowers(Sloppy)(lambda op, reads, ctx: {"other": ItemValue(float, {"value": 1.0})})
    steps, result = live_steps(str(project / "runs" / "best.pt"), extra=Sloppy())
    graph = measure(steps, result, samples, outputs=["extra"])
    graph.prune_sidecars()

    with pytest.raises(
        ValueError, match=r"step 'extra' \(.*Sloppy\): its lowering wrote \['other'\], the op writes \['extra'\]"
    ):
        build_program(graph, samples)


def _unused(record: Record) -> Any:  # keeps the Record import honest for readers of the fixtures above
    return sample_record()


def test_a_model_lowering_is_found_by_the_held_object_s_exact_class() -> None:
    from torpaido.lowering import lowers_model, model_lowering_for

    class Held:
        pass

    class HeldToo(Held):
        pass

    @lowers_model(Held)
    def lower_held(model: Held, pictures: object, ctx: object) -> dict:
        return {}

    assert model_lowering_for(Held()) is lower_held
    assert model_lowering_for(HeldToo()) is None
    with pytest.raises(ValueError, match="Held already has a model lowering"):
        lowers_model(Held)(lambda model, pictures, ctx: {})
