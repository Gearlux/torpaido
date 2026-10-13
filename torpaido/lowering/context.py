"""What a lowering may ask the program for while it runs: a module, a buffer, a constant.

A lowering function runs TWICE per compile — once eagerly on the sample record (the per-node reference
the parity bisect compares against) and once under ``torch.export`` — so everything it registers is
keyed by name: the first call registers a module or a buffer on the program, every later call returns
the one registered. ``constant`` is where a value the envelope lets vary is refused: a lowering that
needs a Python number (an FFT length, a repeat count) cannot take a tensor there.
"""

from typing import Any, Callable, Dict, Union

from torch import Tensor, nn

__all__ = ["Context"]


class Context:
    """Handed to a lowering: the node it lowers, and the program its modules and buffers land on.

    Args:
        program: The ``nn.Module`` every node's modules and buffers are registered on.
        node: The node being lowered (its name, for registration keys and refusals).
        op_name: The op's dotted class name, for refusals.
        where: The step's ``file:line:col``, for refusals.
        origins: ``id(tensor) -> input part name`` for the 0-d tensors the envelope lets vary, so a
            refusal can say WHICH input a varying value came from.
    """

    def __init__(self, program: nn.Module, node: str, op_name: str, where: str, origins: Dict[int, str]) -> None:
        self._program = program
        self.node = node
        self.op_name = op_name
        self.where = where
        self._origins = origins

    def _key(self, name: str) -> str:
        return (self.node + "__" + name).replace("/", "__").replace(".", "_")

    def module(self, name: str, module: nn.Module) -> nn.Module:
        """``module`` registered on the program under ``name`` (once); the registered one on later calls."""
        key = self._key(name)
        held = self._program._modules.get(key)
        if held is None:
            self._program.add_module(key, module)
            return module
        return held

    def buffer(self, name: str, factory: Callable[[], Tensor]) -> Tensor:
        """A tensor made ONCE by ``factory`` and kept as a buffer (a window, a lookup table)."""
        key = self._key(name)
        held = self._program._buffers.get(key)
        if held is None:
            held = factory()
            self._program.register_buffer(key, held)
        return held

    def constant(self, value: Union[Tensor, int, float], name: str) -> Any:
        """``value`` as a Python number — refused when it is a tensor, i.e. something the envelope lets vary.

        Raises:
            ValueError: naming the step, the parameter and the input the value depends on.
        """
        if not isinstance(value, Tensor):
            return value
        origin = self._origins.get(id(value))
        depends = f"{origin}, which the envelope lets vary" if origin else "a value that varies with the input"
        fix = (
            f"Fix it in the envelope ({origin}: <value>) or leave it out so the samples' value is used."
            if origin
            else "Fix the inputs it is computed from in the envelope."
        )
        where = f" at {self.where}" if self.where else ""
        raise ValueError(
            f"torpaido: step {self.node!r} ({self.op_name}) needs {name!r} as a constant, but it depends on "
            f"{depends}{where}\n  {fix}"
        )
