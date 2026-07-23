# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_chialvo_map.py

from __future__ import annotations

"""Full pipeline test for ChialvoMapNeuron (Chialvo 1995).

2D discrete map neuron. x²·exp(y-x) dynamics. Intrinsically excitable
(spikes without input at default k=0.04)."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.chialvo_map import ChialvoMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi

__all__ = ['np', 'pytest', 'ChialvoMapNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'firing_rate', 'spike_count', 'isi']
