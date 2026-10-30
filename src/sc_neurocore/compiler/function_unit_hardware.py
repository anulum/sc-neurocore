# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Isolated combinational function hardware

"""Emit standalone function units using the equation compiler's existing arithmetic."""

from __future__ import annotations

import ast

from . import expr_lut_tables, function_units
from .verilog_compiler_config import FUNCTION_LOWERINGS, Q88
from .verilog_expr_emitter import _VerilogExprEmitter

_RTL_HEADER = """// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Isolated combinational function hardware
"""


def function_unit_functions(function_lowering: str = "v2") -> tuple[str, ...]:
    """Return the canonical standalone functions supported by a lowering.

    Parameters
    ----------
    function_lowering : str, optional
        ``"v2"`` for integer function units or ``"lut-v1"`` for legacy tables
        and the exact functions shared by both lowerings.

    Returns
    -------
    tuple of str
        Sorted canonical names. Aliases, random generators and inline
        arithmetic constructs are not separate function units.

    Raises
    ------
    ValueError
        If the lowering is unknown.

    """
    if function_lowering not in FUNCTION_LOWERINGS:
        raise ValueError(f"Unknown function lowering: {function_lowering!r}")
    return tuple(
        sorted(
            name
            for name in function_units.FUNCTIONS
            if function_lowering == "v2" or name not in expr_lut_tables.V2_ONLY_FUNCTIONS
        )
    )


def emit_function_unit_verilog(
    function: str,
    *,
    data_width: int = 16,
    fraction: int = 8,
    function_lowering: str = "v2",
) -> str:
    """Emit one deterministic combinational module named ``function_unit``.

    Parameters
    ----------
    function : str
        A canonical function name or the ``expit`` alias for ``sigmoid``.
        ``cbrt`` uses the equation grammar's power-one-third lowering.
    data_width : int, optional
        Signed input and output width, between 2 and 32 bits.
    fraction : int, optional
        Fractional bits, less than the word width and at most 30. The sum
        of word and fractional widths must not exceed 61.
    function_lowering : str, optional
        ``"v2"`` or ``"lut-v1"``. Existing function arithmetic is preserved.

    Returns
    -------
    str
        Synthesisable Verilog with signed ``argument0`` and, for binary
        functions, ``argument1`` ports, and a signed ``result`` port. There
        is no clock, reset or neuron-update overhead; latency is zero cycles.

    Raises
    ------
    TypeError
        If the function or fixed-point geometry has the wrong runtime type.
    ValueError
        If the function, lowering or format is unsupported.

    """
    if not isinstance(function, str):
        raise TypeError("function must be a string")
    canonical = "sigmoid" if function == "expit" else function
    q = Q88(data_width=data_width, fraction=fraction, function_lowering=function_lowering)
    function_units.check_format(q.data_width, q.fraction)
    if canonical not in function_unit_functions(function_lowering):
        raise ValueError(f"{function!r} has no {function_lowering!r} function unit")
    arguments = [f"argument{index}" for index in range(function_units.FUNCTION_ARITY[canonical])]
    expression = (
        "argument0 ** (1.0 / 3.0)"
        if canonical == "cbrt"
        else f"{canonical}({', '.join(arguments)})"
    )
    emitter = _VerilogExprEmitter({}, {name: name for name in arguments}, q)
    result: str = emitter.visit(ast.parse(expression, mode="eval").body)
    ports = [f"input signed [{data_width - 1}:0] {name}" for name in arguments]
    ports.append(f"output signed [{data_width - 1}:0] result")
    return (
        _RTL_HEADER
        + f"\n// Function: {canonical}; lowering: {function_lowering}.\n"
        + f"// Fixed-point format: signed {data_width} bits, {fraction} fractional bits.\n"
        + f"// {', '.join(arguments)}: independent input codes; result: output code.\n"
        + "// Combinational: no clock, reset or stored state; latency is zero cycles.\n"
        + f"module function_unit({', '.join(ports)});\n"
        + "\n".join(emitter.intermediates)
        + f"\nassign result = {result};\nendmodule\n"
    )


__all__ = ["emit_function_unit_verilog", "function_unit_functions"]
