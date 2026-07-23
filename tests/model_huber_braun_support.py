# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_huber_braun.py

from __future__ import annotations

"""Full pipeline test for HuberBraunNeuron (Braun, Huber et al. 1998).

Cold receptor, temperature-dependent model with Gaussian noise:
3 currents: I_sd (slow depolarising, g=1.5), I_sr (slow repolarising,
g=0.4), I_L (leak, g=0.1).
2 gating variables: a_sd (tau=10), a_sr (tau=20).

sd_inf = 1/(1+exp(-(v+40)/6))   — activates at depolarised V
sr_inf = 1/(1+exp((v+40)/6))    — activates at hyperpolarised V
Complementary: sd_inf + sr_inf = 1 at v=-40.

Gaussian noise: η·randn() per step (η=0.012). Stochastic model.
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.huber_braun import HuberBraunNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
def _run(neuron: HuberBraunNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['time', 'np', 'pytest', 'HuberBraunNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'isi', '_run']
