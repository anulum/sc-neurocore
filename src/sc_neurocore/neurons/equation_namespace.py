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


def _sigmoid(x: float) -> Any:
    """Logistic sigmoid with clipping for numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sqrt(x: Any) -> Any:
    """Square root that fails before NumPy warning machinery on invalid domains."""
    if np.any(np.asarray(x) < 0):
        raise ValueError("sqrt domain error")
    return np.sqrt(x)


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
        "abs": abs,
        "sin": np.sin,
        "cos": np.cos,
        "tanh": np.tanh,
        "cosh": np.cosh,
        "sinh": np.sinh,
        "exprel": _exprel,
        "sigmoid": _sigmoid,
        "pi": math.pi,
        "clip": np.clip,
        "max": max,
        "min": min,
    }
