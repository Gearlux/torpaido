"""The IR: a graph of nodes joined by VALUES — a record entry at a point in the graph, with its item type.

Two layers live on the same classes. The front end (:mod:`torpaido.frontend`) fills the STRUCTURAL
layer: one node per step, ``Node.inputs`` naming the steps a step reads from (a conservative superset
of what it needs). The dataflow pass (:mod:`torpaido.dataflow`) fills the MEASURED layer on top:
``Node.reads`` / ``Node.writes`` name the record entries the op actually read and wrote, each as a
:class:`Value` that says which node produced it, and ``Node.inputs`` is narrowed to those producers.
:meth:`Graph.prune_sidecars` walks ``inputs`` backwards from the outputs either way — on a measured
graph that walk is exact, on a structural one it drops only what no step reaches at all.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

#: The INPUT node standing for the record the graph receives; its values are ``source.<entry>``.
SOURCE_NODE = "source"


class NodeType(Enum):
    INPUT = auto()
    OUTPUT = auto()
    OP = auto()
    CONSTANT = auto()


@dataclass
class TensorMetadata:
    shape: List[int]
    dtype: str
    name: str


@dataclass(frozen=True)
class Value:
    """A record entry at one point in the graph: ``<producing node>.<entry>``, and the item type measured there.

    The record the graph receives produces ``source.<entry>`` values (``SOURCE_NODE`` in the front end).
    """

    name: str
    entry: str
    item: type


@dataclass
class Node:
    name: str
    node_type: NodeType
    op_type: Optional[str] = None
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[TensorMetadata] = None
    #: The live op the step runs; ``None`` for a fan-in step and for the identity node a subgraph step
    #: leaves behind once its inner steps are inlined.
    op: Any = None
    #: ``file:line:col`` of the step, for every refusal that names the node; blank when unknown.
    where: str = ""
    #: Measured by the dataflow pass: entry -> the value the op read / produced; the entries it removed.
    reads: Dict[str, Value] = field(default_factory=dict)
    writes: Dict[str, Value] = field(default_factory=dict)
    removes: List[str] = field(default_factory=list)


class Graph:
    """Internal Representation of an optimized inference graph."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        #: Filled by the dataflow pass: the values the graph yields (:attr:`input_values` is derived).
        self.output_values: List[Value] = []
        #: The op nodes :meth:`prune_sidecars` removed, in schedule order.
        self.pruned: List[str] = []

    @property
    def input_values(self) -> List[Value]:
        """The values the graph receives — every ``source.<entry>`` an op node still present reads, in first-read order.

        Derived from the nodes, so pruning a step that alone read an entry drops that entry from the inputs."""
        seen: Dict[str, Value] = {}
        for node in self.op_nodes():
            for value in node.reads.values():
                if value.name.startswith(SOURCE_NODE + ".") and value.name not in seen:
                    seen[value.name] = value
        return list(seen.values())

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node
        if node.node_type == NodeType.INPUT:
            self.inputs.append(node.name)
        elif node.node_type == NodeType.OUTPUT:
            self.outputs.append(node.name)

    def op_nodes(self) -> List[Node]:
        """The nodes that run an op, in schedule order (document order IS the schedule)."""
        return [node for node in self.nodes.values() if node.op is not None]

    def prune_sidecars(self) -> List[str]:
        """
        Architectural Mandate: Perform reverse-dependency analysis
        to prune non-inference operations (metadata sidecars).

        Returns the op nodes removed, in schedule order (fan-in and identity nodes go silently: they
        run nothing). The list is also kept as :attr:`pruned`, for a manifest to state what was left out.
        """
        required: Set[str] = set(self.outputs)
        queue = list(self.outputs)

        while queue:
            node_name = queue.pop(0)
            if node_name in self.nodes:
                node = self.nodes[node_name]
                for inp in node.inputs:
                    if inp not in required:
                        required.add(inp)
                        queue.append(inp)

        # Remove nodes not in required set
        dropped: List[str] = []
        for name in list(self.nodes.keys()):
            if name not in required:
                if self.nodes[name].op is not None:
                    dropped.append(name)
                del self.nodes[name]
        self.pruned = dropped
        return dropped
