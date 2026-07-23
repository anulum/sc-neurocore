# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_mat.py

from __future__ import annotations

"""Full pipeline test for MATNeuron (Kobayashi et al. 2009).

Multi-timescale Adaptive Threshold model.
dV/dt = (-(V-V_rest) + R·I) / tau_m
dtheta1/dt = -theta1/tau_1    (fast adaptation, tau=10)
dtheta2/dt = -theta2/tau_2    (slow adaptation, tau=200)
Threshold: V_th = V_base + theta1 + theta2.
On spike: V→V_reset, theta1 += h1, theta2 += h2.

Two adaptation timescales produce spike-frequency adaptation and
burst-rate adaptation. theta1 (fast, h1=5) captures short-term
refractoriness; theta2 (slow, h2=3) captures long-term adaptation.
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import numpy as np
import pytest
from tests.performance_guard import assert_throughput_guard
from sc_neurocore.neurons.models.mat import MATNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
def _run(neuron: MATNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['time', 'np', 'pytest', 'assert_throughput_guard', 'MATNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'isi', '_run']
