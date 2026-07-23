# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_loihi2.py

from __future__ import annotations

"""Full pipeline test for Loihi2Neuron (Intel Loihi 2, 2021).

Programmable 3-state-variable neuromorphic neuron:
s3 -= s3 // tau3
s2 = s2 - s2//tau2 + input + w23·s3
s1 = s1 - s1//tau1 + w12·s2 + w13·s3
Spike: s1→s1_reset, s3+=s3_incr. All integer arithmetic.
Cross-coupling (w12, w13, w23). Adaptation via s3.
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import pytest
from sc_neurocore.neurons.models.loihi2 import Loihi2Neuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count
def _run(neuron: Loihi2Neuron, current: int, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['time', 'pytest', 'Loihi2Neuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', '_run']
