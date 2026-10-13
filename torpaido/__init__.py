"""Torpaido: compiles a model package — its inference graph, its weights, its class names — into an inference artifact.

The IR and the front end need no framework; the two config-facing classes, :class:`CompilePackage` (a
runnable) and :class:`CompiledPackage` (an op), are reached lazily so that importing the package costs
no torch import — the lowering language and the backends import it when they run.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from torpaido.frontend import graph_from_steps, step_inputs
from torpaido.ir import Graph, Node, NodeType, TensorMetadata, Value

if TYPE_CHECKING:
    from torpaido.compile import CompilePackage
    from torpaido.compiled import CompiledPackage

try:
    __version__ = version("torpaido")
except PackageNotFoundError:  # an uninstalled checkout
    __version__ = "0.0.0.dev0"

__all__ = [
    "CompilePackage",
    "CompiledPackage",
    "Graph",
    "Node",
    "NodeType",
    "TensorMetadata",
    "Value",
    "graph_from_steps",
    "step_inputs",
    "__version__",
]


def __getattr__(name: str) -> Any:  # PEP 562: the config-facing classes, without importing torch at package import
    if name == "CompilePackage":
        from torpaido.compile import CompilePackage

        return CompilePackage
    if name == "CompiledPackage":
        from torpaido.compiled import CompiledPackage

        return CompiledPackage
    raise AttributeError(f"module 'torpaido' has no attribute {name!r}")
