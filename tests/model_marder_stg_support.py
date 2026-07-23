# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_marder_stg.py

from __future__ import annotations

"""Full pipeline test for MarderSTGNeuron (Liu-Golowasch-Marder-Abbott 1998).

Single-compartment stomatogastric ganglion neuron with seven voltage-gated
currents (Na, CaT, CaS, A, KCa, Kd, H) plus leak, integrated with fourth-order
Runge-Kutta over thirteen states (v, m_na, h_na, m_cat, h_cat, m_cas, h_cas,
m_a, h_a, m_kca, m_kd, m_h, ca). All gates use the published voltage-dependent
time constants; the calcium reversal is Nernst-derived and intracellular calcium
relaxes towards rest with a 20 ms time constant. The neuron is an endogenous
burster (fires at zero injected current). ModelDB 93321.
"""
import math
import numpy as np
import pytest
from sc_neurocore.neurons.models.marder_stg import MarderSTGNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
_GATES = (
    "m_na",
    "h_na",
    "m_cat",
    "h_cat",
    "m_cas",
    "h_cas",
    "m_a",
    "h_a",
    "m_kca",
    "m_kd",
    "m_h",
)
def _run(neuron: MarderSTGNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['math', 'np', 'pytest', 'MarderSTGNeuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'isi', '_GATES', '_run']
