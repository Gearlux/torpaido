from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


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


@dataclass
class Node:
    name: str
    node_type: NodeType
    op_type: Optional[str] = None
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[TensorMetadata] = None


class Graph:
    """Internal Representation of an optimized inference graph."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node
        if node.node_type == NodeType.INPUT:
            self.inputs.append(node.name)
        elif node.node_type == NodeType.OUTPUT:
            self.outputs.append(node.name)

    def prune_sidecars(self) -> None:
        """
        Architectural Mandate: Perform reverse-dependency analysis
        to prune non-inference operations (metadata sidecars).
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
        all_nodes = list(self.nodes.keys())
        for name in all_nodes:
            if name not in required:
                del self.nodes[name]
