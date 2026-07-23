# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_adaptive_runtime_precision.py

from __future__ import annotations

"""End-to-end tests for adaptive precision telemetry (Roadmap Item 6).

Tests verify:
- Dual-datapath generation (LP + HP sub-modules + wrapper)
- HP-authoritative outputs with no fabric clock gating
- Hysteresis thresholds (THRESH_UP, THRESH_DOWN)
- LP datapath remains available for telemetry
- All canonical Q-format LP/HP pairs
- Validation of invalid configurations
- Structural completeness of generated Verilog
"""
import pytest
from sc_neurocore.compiler.adaptive_runtime_precision import (
    PRECISION_PAIRS,
    compile_adaptive_precision,
)
from sc_neurocore.neurons.equation_builder import from_equations
import json
@pytest.fixture
def lif_neuron():
    """Standard LIF neuron for adaptive precision tests."""
    return from_equations(
        "dv/dt = -(v - E_L)/tau_m + I/C",
        threshold="v > -50",
        reset="v = -65",
        params=dict(E_L=-65, tau_m=10, C=1),
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
def _extract_manifest(verilog: str) -> dict:
    """Extract adaptive precision manifest JSON from generated RTL comments."""
    prefix = "// SC-NeuroCore Adaptive Precision Manifest: "
    for line in verilog.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise AssertionError("Adaptive precision manifest comment not found")

__all__ = ['pytest', 'PRECISION_PAIRS', 'compile_adaptive_precision', 'from_equations', 'json', 'lif_neuron', 'izhikevich_neuron', '_extract_manifest']
