# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_loihi_cuba.py

from __future__ import annotations

"""Full pipeline test for LoihiCUBANeuron (Davies et al. 2018).

Intel Loihi fixed-point CUBA LIF:
u = u - u//tau_u + input
v = v - v//tau_v + u
Spike: v→v_reset. All integer arithmetic (// decay).
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.loihi_cuba import LoihiCUBANeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count
def _run(neuron: LoihiCUBANeuron, current: int, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['time', 'np', 'pytest', 'LoihiCUBANeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', '_run']
