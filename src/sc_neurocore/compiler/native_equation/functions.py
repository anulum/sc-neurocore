# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native equation function and random calls

"""Reuse canonical function programs, RNG programs and legacy quantised lookup grids."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping

from .. import expr_lut_tables as tables
from ..function_unit_programs import program
from ..native_program import lower_program
from ..random_unit_programs import program as random_program
from .arithmetic import EquationArithmetic


def lower_lookup(name: str, argument: int, a: EquationArithmetic) -> int:
    """Lower a maintained lut-v1 function to an explicit native selection graph.

    Parameters
    ----------
    name : str
        Canonical legacy function name, including cbrt for fractional powers.
    argument : int
        Signed Q-word SSA identifier, already narrowed to the Q data width.
    a : EquationArithmetic
        Destination graph and numeric configuration.

    Returns
    -------
    int
        Signed Q-word SSA result, using the original table and endpoint clamps.
    """
    q, b = a.q, a.builder
    width, fraction = q.data_width, q.fraction
    generators: dict[str, Callable[[], list[int]]] = {
        "exp": lambda: tables.exp_lut_entries(width, fraction),
        "log": lambda: tables.log_lut_entries(fraction),
        "sqrt": lambda: tables.sqrt_lut_entries(fraction),
        "tanh": lambda: tables.tanh_lut_entries(fraction),
        "cosh": lambda: tables.cosh_lut_entries(width, fraction),
        "exprel": lambda: tables.exprel_lut_entries(width, fraction),
        "sigmoid": lambda: tables.sigmoid_lut_entries(fraction),
        "sin": lambda: tables.sin_lut_entries(fraction),
        "cos": lambda: tables.cos_lut_entries(fraction),
        "cbrt": lambda: tables.cbrt_lut_entries(fraction),
    }
    entries = tables.saturate_entries(generators[name](), width)
    low, step = -16.0, 0.125
    if name == "log":
        low, step = tables.LOG_LUT_MIN, tables.LOG_LUT_STEP
    elif name == "sqrt":
        low, step = tables.SQRT_LUT_MIN, tables.SQRT_LUT_STEP
    shift = fraction - round(-math.log2(step))
    minimum = round(low * (1 << fraction))
    offset_width = tables.lut_offset_width(width, minimum, max(0, -shift))
    ty = f"i{offset_width}"
    index = b.word_resize(argument, ty)
    index = b.word_binary("sub", index, a.constant(minimum, offset_width), ty)
    index = b.word_binary(
        "ashr" if shift >= 0 else "shl", index, a.constant(abs(shift), offset_width), ty
    )
    index = b.word_clamp(index, 0, len(entries) - 1, ty)

    def select(first: int, end: int) -> int:
        if end - first == 1:
            return a.constant(entries[first], width)
        middle = (first + end) // 2
        condition = b.word_compare("lt", index, a.constant(middle, offset_width))
        return b.word_select(condition, select(first, middle), select(middle, end), f"i{width}")

    return select(0, len(entries))


def lower_call(
    node: ast.Call,
    a: EquationArithmetic,
    value: Callable[[ast.AST, int], int],
    externs: Mapping[str, int],
) -> int:
    """Lower an already parsed equation call to real native scalar instructions.

    Parameters
    ----------
    node : ast.Call
        Named call in the existing equation vocabulary, with numbered RNG sites.
    a : EquationArithmetic
        Destination graph, word geometry and lowering configuration.
    value : callable
        Compile an argument under an explicit signed-word context width.
    externs : mapping of str to int
        Signed i64 step, seed and stream identifiers for random calls.

    Returns
    -------
    int
        Signed Q-word SSA result.

    Raises
    ------
    ValueError
        If the name, argument structure, function lowering or RNG bindings are invalid.
    """
    if not isinstance(node.func, ast.Name) or node.func.id not in tables.SUPPORTED_FUNCTIONS:
        raise ValueError("native equation calls require a supported named function")
    name, b, q = node.func.id, a.builder, a.q
    tables.check_call_arguments(name, node)
    dw = q.data_width
    if name in tables.RANDOM_FUNCTIONS:
        kind = "randn_exact" if name == "randn" and q.random_precision == "exact" else name
        mean = tables.poisson_mean(node) if name == "poisson" else None
        p = random_program(kind, tables.random_site(name, node), dw, q.fraction, mean)
        arg = value(node.args[0], dw) if name == "poisson_grid" else a.constant(0, dw)
        return b.word_resize(lower_program(p, b, [b.word_resize(arg, "i64")], externs), f"i{dw}")
    arguments = [value(arg, dw) for arg in node.args]
    unit = tables.V2_UNIT_FOR.get(name)
    if unit is not None and (q.function_lowering == "v2" or name in tables.EXACT_FUNCTIONS):
        p = program(unit, dw, q.fraction)
        return b.word_resize(
            lower_program(p, b, [b.word_resize(arg, "i64") for arg in arguments]), f"i{dw}"
        )
    if name in tables.V2_ONLY_FUNCTIONS:
        raise ValueError(f"Function '{name}' needs function_lowering='v2'.")
    return lower_lookup("sigmoid" if name == "expit" else name, arguments[0], a)
