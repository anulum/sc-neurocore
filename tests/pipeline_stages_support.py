# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_pipeline_stages.py

from __future__ import annotations

"""End-to-end tests for pipeline stage insertion (Roadmap Item 1).

Tests verify:
- Pipeline registers appear in generated Verilog
- Latency port matches pipeline stage count
- Auto-pipeline from critical path depth
- User-specified pipeline points
- Non-pipelined output remains unchanged when stages=0
- Regression: all existing neuron types compile without error
"""
import pytest
from sc_neurocore.compiler.equation_compiler import compile_to_verilog
from sc_neurocore.compiler.static_analysis import (
    critical_path_depth,
    pipeline_analysis,
    pipeline_stages_needed,
)
from sc_neurocore.neurons.equation_builder import from_equations


@pytest.fixture
def lif_neuron():
    """Standard LIF neuron with one multiply chain."""
    return from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, C=1),
        init=dict(v=-65),
    )


@pytest.fixture
def hh_style_neuron():
    """Multi-multiply neuron (3 multiplies in sequence)."""
    return from_equations(
        "dv/dt = -(v - E_L)/tau_m + I * R / C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, R=1, C=1),
        init=dict(v=-65),
    )


@pytest.fixture
def izhikevich_neuron():
    """Two-state-variable neuron (Izhikevich)."""
    return from_equations(
        "dv/dt = 0.04 * v * v + 5 * v + 140 - u + I",
        "du/dt = a * (b * v - u)",
        threshold="v > 30",
        reset="v = c; u = u + d",
        params=dict(a=0.02, b=0.2, c=-65, d=8),
        init=dict(v=-65, u=-14),
    )


__all__ = [
    "pytest",
    "compile_to_verilog",
    "critical_path_depth",
    "pipeline_analysis",
    "pipeline_stages_needed",
    "from_equations",
    "lif_neuron",
    "hh_style_neuron",
    "izhikevich_neuron",
]
