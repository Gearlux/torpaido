"""The front end reads a GRAPH — which is what makes reverse-dependency pruning possible."""

from typing import Any, Dict

import pytest
from recordstream.flow import parse_flow

from torpaido.frontend import SOURCE_NODE, graph_from_steps, step_inputs
from torpaido.ir import NodeType


class _Op:
    """A stand-in pipeline op — the front end only reads its class name."""

    def __init__(self, tag: str = "") -> None:
        self.tag = tag

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return record


class _Other(_Op):
    pass


def _steps(flow: Dict[str, Any], outputs: str = "") -> Any:
    return parse_flow(flow, outputs)


class TestLinear:
    def test_a_chain_becomes_input_plus_one_node_per_step(self) -> None:
        steps, outputs = _steps({"a": _Op(), "b": _Op()})
        graph = graph_from_steps(steps, outputs)
        assert set(graph.nodes) == {SOURCE_NODE, "a", "b"}
        assert graph.nodes[SOURCE_NODE].node_type is NodeType.INPUT
        assert graph.nodes["a"].node_type is NodeType.OP

    def test_the_implicit_previous_step_edge_is_made_explicit(self) -> None:
        # The authoring form lets a step omit `from:`; an IR must state every edge.
        steps, outputs = _steps({"a": _Op(), "b": _Op()})
        graph = graph_from_steps(steps, outputs)
        assert graph.nodes["a"].inputs == [SOURCE_NODE]
        assert graph.nodes["b"].inputs == ["a"]

    def test_the_op_class_name_rides_along_as_op_type(self) -> None:
        steps, outputs = _steps({"a": _Op(), "b": _Other()})
        graph = graph_from_steps(steps, outputs)
        assert graph.nodes["a"].op_type == "_Op"
        assert graph.nodes["b"].op_type == "_Other"

    def test_the_yielded_step_is_the_graph_output(self) -> None:
        steps, outputs = _steps({"a": _Op(), "b": _Op()})
        assert graph_from_steps(steps, outputs).outputs == ["b"]


class TestBranchy:
    def _flow(self) -> Dict[str, Any]:
        return {
            "start": {},
            "left": {"op": _Op(), "from": "start"},
            "right": {"op": _Op(), "from": "start"},
            "out": {"from": "left", "merge_from": ["right"]},
        }

    def test_fan_out_and_fan_in_become_explicit_inputs(self) -> None:
        steps, outputs = _steps(self._flow(), "out")
        graph = graph_from_steps(steps, outputs)
        assert graph.nodes["left"].inputs == ["start"]
        assert graph.nodes["right"].inputs == ["start"]
        assert graph.nodes["out"].inputs == ["left", "right"]

    def test_a_bind_reference_is_a_real_edge(self) -> None:
        # bind: {param: "producer[key]"} — the KEY is not a graph edge, the producer is.
        flow = {
            "probe": _Op(),
            "main": {"op": _Op(), "from": "probe", "bind": {"level": "probe[image]"}},
        }
        steps, outputs = _steps(flow, "main")
        graph = graph_from_steps(steps, outputs)
        assert graph.nodes["main"].inputs == ["probe"]  # one dependency, reached two ways

    def test_step_inputs_strips_the_entry_and_output_suffixes_and_dedupes(self) -> None:
        # "p[image]" (a record entry) and "p.threshold" (a live @output) are two SLOTS on one
        # dependency; the suffix is not a graph edge, and the producer appears once.
        steps, _ = _steps({"p": _Op(), "c": {"op": _Op(), "bind": {"x": "p[image]", "y": "p.threshold"}}}, "c")
        consumer = next(s for s in steps if s.name == "c")
        assert step_inputs(consumer) == ["p"]


class TestPruning:
    def test_reverse_dependency_pruning_runs_on_the_imported_graph(self) -> None:
        # THE point of consuming a graph: prune_sidecars walks node.inputs backwards from the
        # outputs. A flattened op list carries no inputs, so this could not run at all.
        steps, outputs = _steps(
            {
                "start": {},
                "kept": {"op": _Op(), "from": "start"},
                "sidecar": {"op": _Op(), "from": "start"},
            },
            "kept",
        )
        graph = graph_from_steps(steps, outputs)
        assert "sidecar" in graph.nodes
        graph.prune_sidecars()
        assert set(graph.nodes) == {SOURCE_NODE, "start", "kept"}

    def test_pruning_keeps_a_whole_merge_cone(self) -> None:
        steps, outputs = _steps(
            {
                "start": {},
                "left": {"op": _Op(), "from": "start"},
                "right": {"op": _Op(), "from": "start"},
                "dead": {"op": _Op(), "from": "start"},
                "out": {"from": "left", "merge_from": ["right"]},
            },
            "out",
        )
        graph = graph_from_steps(steps, outputs)
        graph.prune_sidecars()
        assert set(graph.nodes) == {SOURCE_NODE, "start", "left", "right", "out"}


class TestErrors:
    def test_an_unknown_output_raises(self) -> None:
        steps, _ = _steps({"a": _Op()})
        with pytest.raises(ValueError, match="does not name a step"):
            graph_from_steps(steps, "nope")

    def test_attributes_are_stamped_per_step(self) -> None:
        steps, outputs = _steps({"a": _Op()})
        graph = graph_from_steps(steps, outputs, attributes={"a": {"backend": "onnx"}})
        assert graph.nodes["a"].attributes == {"backend": "onnx"}


class TestSubgraphs:
    """A subgraph step is INLINED: its inner steps become ``<step>/<inner>`` nodes — the names the Tracer uses."""

    def _graph(self, **kwargs: Any) -> Any:
        from recordstream.flow import Subgraph

        inner = Subgraph(steps={"a": {"op": _Op("a")}, "b": {"op": _Other("b")}})
        steps, outputs = _steps(
            {"start": _Op(), "pre": {"op": inner, "from": "start"}, "after": {"op": _Op(), "from": "pre"}}
        )
        return graph_from_steps(steps, outputs, **kwargs)

    def test_the_inner_steps_become_prefixed_nodes_reading_the_subgraph_input(self) -> None:
        graph = self._graph()
        assert graph.nodes["pre/a"].inputs == ["start"]
        assert graph.nodes["pre/b"].inputs == ["pre/a"]
        assert graph.nodes["pre/b"].op_type == "_Other"

    def test_the_subgraph_step_itself_is_an_identity_over_its_result(self) -> None:
        graph = self._graph()
        assert graph.nodes["pre"].op is None
        assert graph.nodes["pre"].inputs == ["pre/b"]
        assert graph.nodes["after"].inputs == ["pre"]

    def test_the_live_op_rides_on_its_node(self) -> None:
        graph = self._graph()
        assert graph.nodes["pre/a"].op.tag == "a"
        assert graph.nodes["start"].op_type == "_Op"

    def test_a_location_is_stamped_on_the_step_and_inherited_by_its_inner_nodes(self) -> None:
        graph = self._graph(locations={"pre": "model.yaml:5:9", "start": "model.yaml:2:3"})
        assert graph.nodes["start"].where == "model.yaml:2:3"
        assert graph.nodes["pre"].where == "model.yaml:5:9"
        assert graph.nodes["pre/a"].where == "model.yaml:5:9 (step pre/a)"
        assert graph.nodes["after"].where == ""
