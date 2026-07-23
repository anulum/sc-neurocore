# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_inhomogeneous_poisson.py

from __future__ import annotations

"""Full pipeline test for InhomogeneousPoissonNeuron (Cox 1955).

Doubly stochastic Poisson (time-varying rate):
P(spike) = max(0, rate_hz) · dt_ms / 1000
Bernoulli sampling per step. Stateless — no internal dynamics.
Expected spike count = N · rate · dt_ms / 1000.
Negative rate clipped to 0.
FULL PIPELINE WIRED + PERFORMANCE."""
import math
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.inhomogeneous_poisson import InhomogeneousPoissonNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
def _poisson_interval_probability(rate_hz: float, dt_ms: float) -> float:
    return -math.expm1(-max(0.0, rate_hz) * dt_ms / 1000.0)
def _run(neuron: InhomogeneousPoissonNeuron, rate: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(rate) == 1]

__all__ = ['math', 'time', 'np', 'pytest', 'InhomogeneousPoissonNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'isi', '_poisson_interval_probability', '_run']
