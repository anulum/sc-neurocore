# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation evaluation namespace — the numpy maths environment

"""NumPy evaluation namespace for equation-defined neurons.

The namespace is the set of maths functions and constants an equation string may
reference at runtime (``exp``, ``tanh``, ``exprel``, ``sigmoid``, …). It is a
single, self-contained responsibility split out of :class:`EquationNeuron`, and
its exact bindings are load-bearing: the Python runner and the fixed-point Verilog
datapath must agree bit-for-bit, so ``tanh`` must be :func:`numpy.tanh`,
``exprel`` must be this module's removable-singularity form, and so on. Do not
substitute a ``math`` equivalent or reorder the ``exprel`` limit expression.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.special

from sc_neurocore.neurons.equation_refusals import EquationRefusal


def _sigmoid(x: float) -> Any:
    """Logistic sigmoid with clipping for numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sqrt(x: Any) -> Any:
    """Square root that fails before NumPy warning machinery on invalid domains."""
    if np.any(np.asarray(x) < 0):
        raise EquationRefusal("sqrt domain error")
    return np.sqrt(x)


def _rsqrt(x: Any) -> Any:
    """Evaluate reciprocal roots of finite positive real scalars or arrays.

    Taking the square root first preserves subnormal inputs whose reciprocal
    would overflow. Invalid inputs refuse before NumPy emits a warning.
    """
    values = np.asarray(x)
    if np.iscomplexobj(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise EquationRefusal("rsqrt needs finite positive real inputs")
    return 1.0 / np.sqrt(values)


def _atan2(y: Any, x: Any) -> Any:
    """Return real finite-coordinate angles with the fixed-point zero convention.

    IEEE negative zero has no distinct fixed-point code. Canonicalising both
    zeros keeps the Python equation and generated datapaths on the same branch
    of the negative x-axis. Non-finite and complex coordinates are outside this
    function's compilation domain and raise ``EquationRefusal``.
    """
    first, second = np.asarray(y), np.asarray(x)
    if (
        np.iscomplexobj(first)
        or np.iscomplexobj(second)
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise EquationRefusal("atan2 needs finite real coordinates")
    return np.arctan2(np.where(first == 0, 0.0, first), np.where(second == 0, 0.0, second))


def _softplus(x: Any) -> Any:
    """Evaluate beta-one softplus on finite real inputs, including underflowing tails."""
    values = np.asarray(x)
    if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise EquationRefusal("softplus needs finite real inputs")
    with np.errstate(under="ignore"):
        return np.logaddexp(0.0, values)


def _hypot(first: Any, second: Any) -> Any:
    """Evaluate a two-coordinate norm without squaring finite floating inputs."""
    a, b = np.asarray(first), np.asarray(second)
    if (
        np.iscomplexobj(a)
        or np.iscomplexobj(b)
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        raise EquationRefusal("hypot needs finite real coordinates")
    return np.hypot(a, b)


def _exprel(x: Any) -> Any:
    """(exp(x) - 1) / x with the removable-singularity limit exprel(0) = 1.

    Lets conductance rate functions of the form a*(V-V0)/(1-exp(-(V-V0)/k))
    be written without the 0/0 singularity at V = V0 (it becomes a*k/exprel).
    """
    arr = np.asarray(x, dtype=float)
    safe = np.where(arr == 0.0, 1.0, arr)
    return np.where(np.abs(arr) < 1e-9, 1.0 + arr / 2.0, np.expm1(arr) / safe)


def build_eval_namespace() -> dict[str, Any]:
    """Return the maths namespace exposed to compiled equation expressions.

    The returned dict is fresh on every call so a neuron may own its namespace
    without aliasing another's. The bindings are the exact functions the
    fixed-point emitter mirrors; keep them identical to preserve co-simulation
    bit-exactness (see the module docstring).
    """
    return {
        "exp": np.exp,
        "log": np.log,
        "sqrt": _sqrt,
        "rsqrt": _rsqrt,
        "abs": abs,
        "sin": np.sin,
        "cos": np.cos,
        "tanh": np.tanh,
        "cosh": np.cosh,
        "sinh": np.sinh,
        "tan": np.tan,
        "arctan": np.arctan,
        "atan2": _atan2,
        "hypot": _hypot,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "exp2": np.exp2,
        "log2": np.log2,
        "log10": np.log10,
        "expm1": np.expm1,
        "log1p": np.log1p,
        "softplus": _softplus,
        "erf": scipy.special.erf,
        "floor": np.floor,
        "ceil": np.ceil,
        "trunc": np.trunc,
        "sign": np.sign,
        "round": np.rint,
        "heaviside": np.heaviside,
        "exprel": _exprel,
        "sigmoid": _sigmoid,
        "expit": _sigmoid,
        "pi": math.pi,
        "clip": np.clip,
        "max": max,
        "min": min,
    }
