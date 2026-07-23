# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_rulkov_map.py

from __future__ import annotations

"""Full pipeline test for RulkovMapNeuron (Rulkov 2001).

Discrete map-based neuron: x[n+1] = f(x[n], y[n]) + I (3 branches),
y[n+1] = y[n] - μ(x[n]+1) + μσ. No ODE — O(1) per step.
Exhibits spiking and bursting depending on parameters."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count
def _run(neuron: RulkovMapNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['np', 'pytest', 'RulkovMapNeuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', '_run']
