# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_expif.py

from __future__ import annotations

"""Source-fidelity and pipeline tests for the maintained ExpIF neuron.

The voltage flow follows Fourcaud-Trocmé et al. (2003), Equations 6 and
10, after division by leak conductance. ``v_rh`` is the soft exponential
threshold; ``v_threshold`` is a separate finite spike cutoff.
"""
import math
import os
import time
import numpy as np
import pytest
from sc_neurocore.analysis.spike_stats.basic import firing_rate, isi, spike_count
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.neurons.models.expif import ExpIFNeuron


def _run(neuron: ExpIFNeuron, current: float, steps: int) -> list[int]:
    """Return the step indices at which the neuron emits a spike."""
    return [index for index in range(steps) if neuron.step(current) == 1]


def _rhs(neuron: ExpIFNeuron, v: float, current: float) -> float:
    """Independent source-equation derivative with the event-surface bound."""
    bounded_v = min(v, neuron.v_threshold)
    exponential = neuron.delta_t * math.exp((bounded_v - neuron.v_rh) / neuron.delta_t)
    return (-(bounded_v - neuron.v_rest) + exponential + current) / neuron.tau


def _rk4_candidate(neuron: ExpIFNeuron, current: float) -> float:
    """Independent candidate-first classical RK4 update."""
    v0 = neuron.v
    k1 = _rhs(neuron, v0, current)
    k2 = _rhs(neuron, v0 + 0.5 * neuron.dt * k1, current)
    k3 = _rhs(neuron, v0 + 0.5 * neuron.dt * k2, current)
    k4 = _rhs(neuron, v0 + neuron.dt * k3, current)
    return v0 + (neuron.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _euler_candidate(neuron: ExpIFNeuron, current: float) -> float:
    """Return raw Euler solely to prove the maintained method is not Euler."""
    return neuron.v + neuron.dt * _rhs(neuron, neuron.v, current)


__all__ = [
    "math",
    "os",
    "time",
    "np",
    "pytest",
    "firing_rate",
    "isi",
    "spike_count",
    "SpikeMonitor",
    "Network",
    "Population",
    "Projection",
    "PoissonInput",
    "ExpIFNeuron",
    "_run",
    "_rhs",
    "_rk4_candidate",
    "_euler_candidate",
]
