# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_vip_neuron.py

from __future__ import annotations

"""Full pipeline test for VIPNeuron (Porter 1998 + Kv4 A-current).

VIP irregular-spiking interneuron: the slow-inactivating A-type current produces
firing accommodation. The candidate-first RK4 integrator advances the five-state
``(V, h, n, a, b)`` system."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.vip_neuron import VIPNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
def _spikes(neuron: VIPNeuron, current: float, steps: int) -> int:
    return sum(neuron.step(current) for _ in range(steps))

__all__ = ['np', 'pytest', 'VIPNeuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', '_spikes']
