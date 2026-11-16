# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Numerical safety utilities for neuron models

"""Overflow-safe mathematical functions for biophysical neuron models.

58/113 neuron models use np.exp() in Boltzmann activation functions.
When membrane voltage exceeds ~±700 mV (possible during unstable Euler
integration), exp() returns inf/0 → NaN cascade → silent model death.

These utilities clip arguments before evaluation. Import into any model:

    from sc_neurocore.utils.numerics import safe_exp, safe_cosh, clip_gating, clip_voltage

Integer models of neuromorphic chips also need the division their hardware does, which truncates towards zero:
:func:`trunc_div`.
"""

from __future__ import annotations

import numpy as np


def safe_exp(x: float) -> float:
    """exp() with argument clipped to [-500, 500] to prevent overflow."""
    return float(np.exp(np.clip(x, -500, 500)))


def safe_cosh(x: float) -> float:
    """cosh() with argument clipped to [-500, 500] to prevent overflow."""
    return float(np.cosh(np.clip(x, -500, 500)))


def safe_tanh(x: float) -> float:
    """tanh() with argument clipped to [-500, 500]."""
    return float(np.tanh(np.clip(x, -500, 500)))


def boltzmann(v: float, v_half: float, k: float) -> float:
    """Boltzmann sigmoid: 1 / (1 + exp((v_half - v) / k)). Overflow-safe."""
    return 1.0 / (1.0 + safe_exp((v_half - v) / k))


def boltzmann_inv(v: float, v_half: float, k: float) -> float:
    """Inverse Boltzmann: 1 / (1 + exp((v - v_half) / k)). Overflow-safe."""
    return 1.0 / (1.0 + safe_exp((v - v_half) / k))


def clip_gating(x: float) -> float:
    """Clip gating variable to physiological range [0, 1]."""
    return float(np.clip(x, 0.0, 1.0))


def clip_voltage(v: float, v_min: float = -200.0, v_max: float = 100.0) -> float:
    """Clip membrane voltage to safe range (default [-200, 100] mV)."""
    return float(np.clip(v, v_min, v_max))


def trunc_div(numerator: int, divisor: int) -> int:
    """Divide two integers and round the quotient towards zero, as C, Rust and Verilog divide.

    Python's ``//`` floors instead, so ``-7 // 5`` is -2 where this gives -1. The quotient is odd in each argument, and
    exact for integers of any size.

    Parameters
    ----------
    numerator : int
        The integer divided.
    divisor : int
        The integer it is divided by; 0 raises :class:`ZeroDivisionError`.

    Returns
    -------
    int
        ``sign · (|numerator| // |divisor|)``.
    """
    quotient = abs(numerator) // abs(divisor)
    return -quotient if (numerator < 0) != (divisor < 0) else quotient
