# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_terman_wang.py

from __future__ import annotations

"""Module-specific behavioural tests for TermanWangOscillator.

The tests verify the two-state ODE, coupled RK4 integration, slow recovery,
finite-domain rejection, and public network/analysis wiring without
bucket-style coverage assertions."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.terman_wang import TermanWangOscillator
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: TermanWangOscillator, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _rhs(neuron: TermanWangOscillator, v: float, w: float, current: float) -> tuple[float, float]:
    f = 3.0 * v - v * v * v + 2.0
    g = neuron.alpha * (1.0 + np.tanh(v / neuron.beta))
    return f - w + current + neuron.rho, neuron.epsilon * (g - w)


def _rk4_reference(
    neuron: TermanWangOscillator, v: float, w: float, current: float
) -> tuple[float, float]:
    dt = neuron.dt
    k1 = _rhs(neuron, v, w, current)
    k2 = _rhs(neuron, v + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1], current)
    k3 = _rhs(neuron, v + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1], current)
    k4 = _rhs(neuron, v + dt * k3[0], w + dt * k3[1], current)
    return (
        v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


__all__ = [
    "time",
    "np",
    "pytest",
    "TermanWangOscillator",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "_run",
    "_rhs",
    "_rk4_reference",
]
