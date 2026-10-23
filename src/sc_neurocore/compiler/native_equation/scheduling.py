# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation multiply scheduling

"""Allocate maintained multiply names before context-sized native expression lowering."""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence

from .. import expr_lut_tables
from ..verilog_compiler_config import Q88


class EquationSchedule:
    """Plan expression-unit numbering and selected one-clock product registers.

    Parameters
    ----------
    q : Q88
        Existing numeric/function configuration.
    register : callable
        Append a real native zero-reset, one-clock integer word register.
        Arguments are its input SSA identifier and explicit signed type.
    pipeline : bool, optional
        Register every product which the maintained emitter pipelines.
    points : sequence of str, optional
        Explicit multiply names, such as _mul2, mul2 or _dt_mul_v.

    Notes
    -----
    Naming follows the maintained Verilog expression emitter's traversal,
    including non-pipelined reciprocal, division, power, table and RNG units.
    Number allocation is separate from typed word construction: a subexpression
    may be resized into several contexts without acquiring several registers.
    The caller supplies native storage, never an executable RTL payload.
    """

    def __init__(
        self,
        q: Q88,
        register: Callable[[int, str], int],
        *,
        pipeline: bool = False,
        points: Sequence[str] = (),
    ) -> None:
        self.q = q
        self.register = register
        self.pipeline = pipeline
        self.points = frozenset(points)
        self.unit_count = 0
        self.register_count = 0
        self._seen: set[ast.AST] = set()
        self._units: dict[ast.AST, tuple[int, ...]] = {}
        self._registered: dict[str, int] = {}

    def plan(self, node: ast.AST) -> None:
        """Allocate unit names in the original compiler's evaluation order.

        Parameters
        ----------
        node : ast.AST
            Parsed original equation subtree. Replanning the same node is
            idempotent, including shared arguments inside generated clamps.

        Notes
        -----
        A random call reserves its unit before lowering a variable Poisson
        mean. Other calls lower their argument units first. Fractional-power
        exponent arithmetic consumes names even when its result is folded.
        """
        if node in self._seen:
            return
        self._seen.add(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in expr_lut_tables.RANDOM_FUNCTIONS:
                self._allocate(node, 1)
                if name == "poisson_grid" and node.args:
                    self.plan(node.args[0])
                return
            for argument in node.args:
                self.plan(argument)
            if name not in {"abs", "clip", "max", "min"}:
                self._allocate(node, 1)
            return
        if isinstance(node, ast.BinOp):
            self.plan(node.left)
            self.plan(node.right)
            if isinstance(node.op, (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)):
                self._allocate(node, 1)
            elif isinstance(node.op, ast.Pow):
                exponent = node.right
                if isinstance(exponent, ast.Constant) and isinstance(exponent.value, (int, float)):
                    if exponent.value in (2, 3):
                        self._allocate(node, int(exponent.value) - 1)
                        return
                    if type(exponent.value) is int and 4 <= exponent.value <= 8:
                        self._allocate(node, exponent.value - 1)
                        return
                self._allocate(node, 1)
            return
        for child in ast.iter_child_nodes(node):
            self.plan(child)

    def product(self, node: ast.BinOp, value: int, width: int) -> int:
        """Register a selected explicit multiplication, sharing repeated context reads.

        Parameters
        ----------
        node : ast.BinOp
            The original planned multiplication node.
        value : int
            SSA wide product before Q rounding.
        width : int
            Physical product width, normally twice the Q data width.

        Returns
        -------
        int
            Original combinational SSA value or the selected registered value.

        Raises
        ------
        ValueError
            If the node is not a planned explicit multiplication.
        """
        slots = self._units.get(node)
        if not isinstance(node.op, ast.Mult) or slots is None or len(slots) != 1:
            raise ValueError("equation product must be planned before pipeline construction")
        index = slots[0]
        return self.named_product(value, width, f"_mul{index}", aliases=(f"mul{index}",))

    def named_product(
        self, value: int, width: int, name: str, *, aliases: Sequence[str] = ()
    ) -> int:
        """Append selected storage for an integrator or expression product.

        Parameters
        ----------
        value : int
            Combinational wide product SSA identifier.
        width : int
            Physical signed register width.
        name : str
            Maintained pipeline name, unique within the compiled module.
        aliases : sequence of str, optional
            Other accepted spellings of the same product name.

        Returns
        -------
        int
            Original SSA value, or a shared real registered SSA value.
        """
        if not self.pipeline and not any(point in self.points for point in (name, *aliases)):
            return value
        if name not in self._registered:
            self._registered[name] = self.register(value, f"i{width}")
            self.register_count += 1
        return self._registered[name]

    def _allocate(self, node: ast.AST, count: int) -> None:
        """Reserve source-compatible unit indices without generating unused arithmetic."""
        self._units[node] = tuple(range(self.unit_count, self.unit_count + count))
        self.unit_count += count
