# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_benda_herz.py

from __future__ import annotations

"""Full pipeline test for BendaHerzNeuron (Benda & Herz 2003).

Phenomenological spike-frequency adaptation. Stochastic spiking
from instantaneous f-I curve with adaptation variable A."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.benda_herz import BendaHerzNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count
def _rk4_reference(neuron: BendaHerzNeuron, current: float) -> tuple[float, float]:
    def rhs(a: float) -> tuple[float, float]:
        rate = neuron._f_onset(current - a)
        return -a / neuron.tau_a + neuron.delta_a * rate, rate

    k1, r1 = rhs(neuron.a)
    k2, r2 = rhs(neuron.a + 0.5 * neuron.dt * k1)
    k3, r3 = rhs(neuron.a + 0.5 * neuron.dt * k2)
    k4, r4 = rhs(neuron.a + neuron.dt * k3)
    next_a = neuron.a + (neuron.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    average_rate = (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
    probability = -np.expm1(-(average_rate * neuron.dt / 1000.0))
    return next_a, probability

__all__ = ['np', 'pytest', 'BendaHerzNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'firing_rate', 'spike_count', '_rk4_reference']
