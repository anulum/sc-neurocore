# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_bit_true_kernel.py

from __future__ import annotations

"""Unit tests for the bit-true fixed-point kernel generators.

Structural assertions on the emitted C / Rust — the numeric bit-for-bit proof
against the Verilog RTL is in ``tests/test_bit_true_cosim.py``.
"""
import pytest
from sc_neurocore.compiler.intelligence.bit_true_kernel import (
    _accumulate_bias,
    _ctype,
    _format_tables_c,
    _rtype,
    generate_bittrue_kernel,
    generate_bittrue_kernel_from_neuron,
)
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations


def _lif(dt: float = 1.0) -> EquationNeuron:
    return from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, C=1),
        init=dict(v=-65),
        dt=dt,
    )


def _adaptive_reset_neuron() -> EquationNeuron:
    """Build a two-state neuron whose recovery reset depends on its candidate."""
    return from_equations(
        "dv/dt = 0.04*v**2 + 5*v + 140 - u + I",
        "du/dt = a*(b*v - u)",
        threshold="v > 30",
        reset="v = c; u = u + d",
        params=dict(a=0.02, b=0.2, c=-65, d=8),
        init=dict(v=-65, u=-13),
        dt=1.0,
    )


def _wrapped_phase_neuron() -> EquationNeuron:
    """Build a discrete phase map with pre-step crossing and positive modulo."""
    candidate = "theta + dt * ((1.0 - cos(theta)) + (1.0 + cos(theta)) * gain * I)"
    previous_candidate = (
        "theta_prev + dt * ((1.0 - cos(theta_prev)) + (1.0 + cos(theta_prev)) * gain * I)"
    )
    return EquationNeuron(
        equations={"theta": f"({candidate}) % 6.283185307179586"},
        parameters={
            "dt": 0.1,
            "gain": 1.0,
            "theta_threshold": 3.141592653589793,
        },
        state={"theta": 0.0},
        threshold=f"theta_prev < theta_threshold <= ({previous_candidate})",
        dt=1.0,
        method="map",
    )


__all__ = [
    "pytest",
    "_accumulate_bias",
    "_ctype",
    "_format_tables_c",
    "_rtype",
    "generate_bittrue_kernel",
    "generate_bittrue_kernel_from_neuron",
    "EquationNeuron",
    "from_equations",
    "_lif",
    "_adaptive_reset_neuron",
    "_wrapped_phase_neuron",
]
