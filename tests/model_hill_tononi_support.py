# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_hill_tononi.py

from __future__ import annotations

"""Full pipeline test for HillTononiNeuron (Hill & Tononi 2005).

Thalamocortical sleep/wake model with 6 ionic currents:
I_Na(g=50, m³_inf·h), I_K(g=5, n⁴), I_h(g=1, m_h),
I_T(g=3, m²_inf·h_t), I_KNa(g=1.33, w_KNa), I_L(g=0.02).

6 state variables: v, h_na, n_k, m_h, h_t, na_i.
Na-dependent K current: w_KNa = 0.37/(1+(38.7/Na_i)^3.5).
Na/K pump: dNa_i = (-0.001·I_Na - pump_max·Na_i/(Na_i+Na_eq))·dt.
Intrinsic oscillator — fires at I=0.
FULL PIPELINE WIRED + PERFORMANCE."""
import math
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
def _run(neuron: HillTononiNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['math', 'time', 'np', 'pytest', 'HillTononiNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'isi', '_run']
