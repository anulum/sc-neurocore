# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_booth_rinzel.py

from __future__ import annotations

"""Full pipeline test for BoothRinzelNeuron (Booth & Rinzel 1995).

2-compartment bistable motoneuron: soma (Na/K) + dendrite (Ca/KCa).
4 sub-steps per step. Exhibits bistability at high current."""
import numpy as np
from sc_neurocore.neurons.models.booth_rinzel import BoothRinzelNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi
import pytest
def _booth_state_tuple(neuron):
    return (neuron.vs, neuron.vd, neuron.h, neuron.n, neuron.q, neuron.ca)

__all__ = ['np', 'BoothRinzelNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'firing_rate', 'spike_count', 'isi', 'pytest', '_booth_state_tuple']
