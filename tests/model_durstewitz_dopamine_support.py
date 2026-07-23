# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_durstewitz_dopamine.py

from __future__ import annotations

"""Full pipeline test for DurstewitzDopamineNeuron.

Fires spontaneously at I=0 (~20 spikes/10k). Rate increases with I.
Performance: ~54K steps/s. Full pipeline wired."""
import math
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.durstewitz_dopamine import DurstewitzDopamineNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate
def _run(neuron: DurstewitzDopamineNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['math', 'time', 'np', 'pytest', 'DurstewitzDopamineNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', '_run']
