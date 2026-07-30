"""Torpaido — the compilation and optimization engine for record pipelines and models.

Two surfaces so far: the internal representation (:mod:`torpaido.ir`) and the front end that
builds one from a record pipeline (:mod:`torpaido.frontend`). Compilation consumes a GRAPH —
see the front end's module docstring for why that is the input shape rather than a flattened
op sequence.
"""

from torpaido.frontend import SOURCE_NODE, graph_from_steps, step_inputs
from torpaido.ir import Graph, Node, NodeType, TensorMetadata

__all__ = [
    "Graph",
    "Node",
    "NodeType",
    "TensorMetadata",
    "SOURCE_NODE",
    "graph_from_steps",
    "step_inputs",
]
