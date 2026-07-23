# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_plif.py

from __future__ import annotations

"""Full pipeline test for ParametricLIFNeuron (Fang et al. 2021).

Parametric LIF with learnable decay alpha = sigmoid(a).
V(t+1) = alpha·V(t)·(1-spike(t)) + I(t).
Spike threshold: I_crit = threshold·(1 - alpha)."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.plif import ParametricLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count

__all__ = ['np', 'pytest', 'ParametricLIFNeuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count']
