# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_arcane_neuron.py

from __future__ import annotations

"""Full pipeline test for ArcaneNeuron (Šotek & Arcane Sapience 2026).

Unified self-referential cognition neuron. 5 coupled subsystems:
FAST (τ=5ms), WORKING (τ=200ms), DEEP (τ=10s, identity), GATE
(attention), PREDICTOR (self-model). v_deep PERSISTS through reset.
Performance: ~27K isolation steps/s. FULL PIPELINE WIRED."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate
def _run(neuron: ArcaneNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]
def _exact_relaxation(state: float, steady_state: float, dt: float, tau: float) -> float:
    decay = np.exp(-dt / tau)
    return decay * state + (1.0 - decay) * steady_state
def _stable_sigmoid(x: float) -> float:
    if x >= 0.0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)

__all__ = ['time', 'np', 'pytest', 'ArcaneNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', '_run', '_exact_relaxation', '_stable_sigmoid']
