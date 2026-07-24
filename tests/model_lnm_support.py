# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_lnm.py

from __future__ import annotations

"""Full pipeline test for LearnableNeuronModel (Jahns et al. 2025).

Fully parameterised learnable neuron:
V[t+1] = α·V[t] + β·I[t] + γ·f(V[t])
f(V) = 1/(1+exp(-f_slope·(V-f_shift)))  — learnable sigmoid.
α=0.9, β=0.1, γ=0.05. All trainable for SNN backprop.
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.lnm import LearnableNeuronModel
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: LearnableNeuronModel, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "time",
    "np",
    "pytest",
    "LearnableNeuronModel",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "firing_rate",
    "_run",
]
