# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_theta.py

from __future__ import annotations

"""Full pipeline test for ThetaNeuron (Ermentrout & Kopell 1986).

Canonical Type-I neuron on the unit circle: dθ/dt = (1-cosθ) + (1+cosθ)·I.
Mathematically equivalent to QIF via change of variables.
Analytical: ISI = π/√I (continuous time), f = √I/π Hz."""
import math
import numpy as np
import pytest
import sc_neurocore.accel.theta as backends
from sc_neurocore.neurons.models.theta import ThetaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: ThetaNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _wrap_phase(theta: float) -> float:
    return ((theta + math.pi) % (2.0 * math.pi)) - math.pi


def _exact_theta_candidate(theta: float, current: float, dt: float) -> tuple[float, bool]:
    y = math.tan(theta / 2.0)
    if current > 0.0:
        root_i = math.sqrt(current)
        phase = math.atan(y / root_i)
        next_phase = phase + root_i * dt
        return _wrap_phase(
            2.0 * math.atan(root_i * math.tan(next_phase))
        ), next_phase >= math.pi / 2.0
    if current == 0.0:
        denominator = 1.0 - y * dt
        if abs(denominator) <= 1e-15:
            return -math.pi, True
        return _wrap_phase(2.0 * math.atan(y / denominator)), denominator <= 0.0

    root_i = math.sqrt(-current)
    if math.isclose(y, -root_i, rel_tol=0.0, abs_tol=1e-15):
        return theta, False
    ratio = (y - root_i) / (y + root_i)
    evolved = ratio * math.exp(2.0 * root_i * dt)
    denominator = 1.0 - evolved
    spiked = ratio < 1.0 <= evolved or abs(denominator) <= 1e-15
    if spiked and abs(denominator) <= 1e-15:
        return -math.pi, True
    return _wrap_phase(2.0 * math.atan(root_i * (1.0 + evolved) / denominator)), spiked


__all__ = [
    "math",
    "np",
    "pytest",
    "backends",
    "ThetaNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "isi",
    "firing_rate",
    "_run",
    "_wrap_phase",
    "_exact_theta_candidate",
]
