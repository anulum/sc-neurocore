# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Target-independent expression lowering tables

"""Target-independent numerics shared by every expression-lowering backend.

The Verilog backend (:mod:`sc_neurocore.compiler.verilog_expr_emitter`) and any
future C/C++/Rust backend must agree bit-for-bit on the fixed-point
transcendental lookup tables and on which functions the equation grammar
supports; otherwise a "bit-true" software kernel would silently diverge from the
generated RTL. This module is the single source of truth for both:

- the canonical function vocabulary (:data:`SUPPORTED_FUNCTIONS`),
- the compile-time constant folder (:func:`const_float`) used to recognise
  fractional exponents such as ``1.0 / 3.0``,
- the 256-point symmetric and positive-log sample grids, and
- the quantised LUT-entry generators for every supported transcendental,
  parameterised by the target fixed-point width and fraction.

The entry generators return integer Q-format values identical to the literals
the Verilog emitter previously computed inline, so extracting them here leaves
the generated RTL byte-for-byte unchanged.
"""

from __future__ import annotations

import ast
import math

# Function names the expression grammar accepts. Backends must implement every
# entry (transcendentals via the LUT generators below; abs/clip/max/min inline).
SUPPORTED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "exp",
        "log",
        "sqrt",
        "tanh",
        "cosh",
        "exprel",
        "sigmoid",
        "expit",
        "sin",
        "cos",
        "abs",
        "clip",
        "max",
        "min",
    }
)

# ``log`` cannot share the signed symmetric grid because its domain is strictly
# positive.  Use a power-of-two geometry so RTL indexing remains one subtract
# plus one shift, with the smallest Q8.8-positive value as the lower endpoint.
LOG_LUT_MIN = 1.0 / 256.0
LOG_LUT_STEP = 1.0 / 32.0
LOG_LUT_SIZE = 256


def const_float(node: ast.AST) -> float | None:
    """Constant-fold a literal or simple literal-arithmetic node to a float.

    Recognises fractional exponents such as ``1.0 / 3.0`` in ``x ** p`` by
    folding compile-time-constant expressions built from literals and ``+``,
    ``-``, ``*``, ``/`` (and unary minus).

    Parameters
    ----------
    node : ast.AST
        Expression node to fold.

    Returns
    -------
    float or None
        The folded value, or ``None`` if the node is not a compile-time
        constant.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = const_float(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.BinOp):
        a = const_float(node.left)
        b = const_float(node.right)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Div):
            return a / b if b != 0 else None
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
    return None


def symmetric_sample_points() -> list[float]:
    """Return the 256 sample points over ``[-16, 16)`` at 0.125 spacing.

    The symmetric transcendental LUTs (exp, tanh, sigmoid, sin, cos, cosh,
    exprel, cbrt) are tabulated on this grid; the value ``x == 0`` falls at
    index 128. Must match the LUT-call defaults (``lut_min=-16``, ``step=0.125``)
    in the emitting backends.

    Returns
    -------
    list of float
        The 256 tabulation points.
    """
    return [-16.0 + i * 0.125 for i in range(256)]


def log_sample_points() -> list[float]:
    """Return the 256 positive ``log`` points over ``[1/256, 8+1/256)``.

    The ``1/32`` spacing and ``1/256`` offset are both exactly representable in
    Q8.8 and Q16.16, so every lowering backend derives the same integer index.

    Returns
    -------
    list of float
        The 256 strictly positive tabulation points.
    """
    return [LOG_LUT_MIN + i * LOG_LUT_STEP for i in range(LOG_LUT_SIZE)]


def _signed_cap(data_width: int) -> int:
    """Return the largest signed value representable in ``data_width`` bits."""
    return (1 << (data_width - 1)) - 1


def exp_lut_entries(data_width: int, fraction: int) -> list[int]:
    """Quantised ``exp`` LUT over the symmetric grid, saturated to the word max.

    Parameters
    ----------
    data_width : int
        Fixed-point word width; sets the signed saturation cap.
    fraction : int
        Number of fractional bits (the Q-format scale ``1 << fraction``).

    Returns
    -------
    list of int
        256 integer Q-format entries.
    """
    cap = _signed_cap(data_width)
    scale = 1 << fraction
    return [min(int(round(math.exp(x) * scale)), cap) for x in symmetric_sample_points()]


def log_lut_entries(fraction: int) -> list[int]:
    """Quantised ``log`` LUT on the canonical positive 256-point grid.

    Parameters
    ----------
    fraction : int
        Number of fractional bits.

    Returns
    -------
    list of int
        256 integer Q-format entries.
    """
    scale = 1 << fraction
    return [int(round(math.log(value) * scale)) for value in log_sample_points()]


def sqrt_lut_entries(fraction: int) -> list[int]:
    """Quantised ``sqrt`` LUT (16 entries over ``[0, 7.5]`` at 0.5 spacing).

    Parameters
    ----------
    fraction : int
        Number of fractional bits.

    Returns
    -------
    list of int
        16 integer Q-format entries.
    """
    scale = 1 << fraction
    return [int(round(math.sqrt(max(i * 0.5, 0)) * scale)) for i in range(16)]


def tanh_lut_entries(fraction: int) -> list[int]:
    """Quantised ``tanh`` LUT over the symmetric grid.

    Parameters
    ----------
    fraction : int
        Number of fractional bits.

    Returns
    -------
    list of int
        256 integer Q-format entries.
    """
    scale = 1 << fraction
    return [int(round(math.tanh(x) * scale)) for x in symmetric_sample_points()]


def cosh_lut_entries(data_width: int, fraction: int) -> list[int]:
    """Quantised ``cosh`` LUT over the symmetric grid, saturated to the word max.

    Parameters
    ----------
    data_width : int
        Fixed-point word width; sets the signed saturation cap (cosh grows fast).
    fraction : int
        Number of fractional bits.

    Returns
    -------
    list of int
        256 integer Q-format entries.
    """
    cap = _signed_cap(data_width)
    scale = 1 << fraction
    return [min(int(round(math.cosh(x) * scale)), cap) for x in symmetric_sample_points()]


def cbrt_lut_entries(fraction: int) -> list[int]:
    """Quantised cube-root LUT over the symmetric grid (odd, sign-preserving).

    Parameters
    ----------
    fraction : int
        Number of fractional bits.

    Returns
    -------
    list of int
        256 integer Q-format entries.
    """
    scale = 1 << fraction

    def cbrt(z: float) -> float:
        return math.copysign(abs(z) ** (1.0 / 3.0), z)

    return [int(round(cbrt(x) * scale)) for x in symmetric_sample_points()]


def exprel_lut_entries(data_width: int, fraction: int) -> list[int]:
    """Quantised ``exprel(z) = (exp(z)-1)/z`` LUT, with the removable limit 1 at 0.

    Grows like ``exp(z)/z`` for large ``z``, so entries saturate to the word max.

    Parameters
    ----------
    data_width : int
        Fixed-point word width; sets the signed saturation cap.
    fraction : int
        Number of fractional bits.

    Returns
    -------
    list of int
        256 integer Q-format entries.
    """
    cap = _signed_cap(data_width)
    scale = 1 << fraction

    def exprel(z: float) -> float:
        return 1.0 if abs(z) < 1e-9 else math.expm1(z) / z

    return [min(int(round(exprel(x) * scale)), cap) for x in symmetric_sample_points()]


def sigmoid_lut_entries(fraction: int) -> list[int]:
    """Quantised logistic-sigmoid LUT over the symmetric grid.

    Parameters
    ----------
    fraction : int
        Number of fractional bits.

    Returns
    -------
    list of int
        256 integer Q-format entries.
    """
    scale = 1 << fraction
    return [int(round(1.0 / (1.0 + math.exp(-x)) * scale)) for x in symmetric_sample_points()]


def sin_lut_entries(fraction: int) -> list[int]:
    """Quantised ``sin`` LUT over the symmetric grid.

    Parameters
    ----------
    fraction : int
        Number of fractional bits.

    Returns
    -------
    list of int
        256 integer Q-format entries.
    """
    scale = 1 << fraction
    return [int(round(math.sin(x) * scale)) for x in symmetric_sample_points()]


def cos_lut_entries(fraction: int) -> list[int]:
    """Quantised ``cos`` LUT over the symmetric grid.

    Parameters
    ----------
    fraction : int
        Number of fractional bits.

    Returns
    -------
    list of int
        256 integer Q-format entries.
    """
    scale = 1 << fraction
    return [int(round(math.cos(x) * scale)) for x in symmetric_sample_points()]
