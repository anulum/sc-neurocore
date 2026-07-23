# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_spike_response.py

from __future__ import annotations

"""Full pipeline test for SpikeResponseNeuron (SRM0, Gerstner 1995).

Kernel-based: v(t) = η(tss) + κ(I). η is refractory afterpotential
(decays from eta_reset), κ is instantaneous input kernel.
No voltage accumulation — v computed fresh each step.

Timing: eta uses time_since_spike BEFORE increment. After spike,
tss=0; next step uses eta(0) = eta_reset, then tss becomes dt."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.spike_response import SpikeResponseNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate
def _kappa(I: float, dt: float, tau_kappa: float) -> float:
    return I * (1.0 - np.exp(-dt / tau_kappa))
def _eta(tss: float, eta_reset: float, tau_eta: float) -> float:
    if tss >= 100.0:
        return 0.0
    return eta_reset * np.exp(-tss / tau_eta)
def _run(neuron: SpikeResponseNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['np', 'pytest', 'SpikeResponseNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'isi', 'firing_rate', '_kappa', '_eta', '_run']
