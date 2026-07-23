# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_morris_lecar.py

from __future__ import annotations

"""Full pipeline test for MorrisLecarNeuron (Morris & Lecar 1981).

Calcium-potassium oscillator, 2D:
C dV/dt = -g_Ca·m_∞(V)·(V-E_Ca) - g_K·w·(V-E_K) - g_L·(V-E_L) + I
dw/dt = λ(V)·(w_∞(V) - w)

m_∞(V) = 0.5·(1 + tanh((V-v1)/v2))  — instantaneous Ca activation
w_∞(V) = 0.5·(1 + tanh((V-v3)/v4))  — K activation steady state
λ(V) = φ·cosh((V-v3)/(2·v4))        — K opening rate

Three currents: I_Ca (instantaneous), I_K (w-gated), I_L (leak).
Type-II excitability: oscillation in frequency band, Hopf bifurcation.
FULL PIPELINE WIRED + PERFORMANCE."""
import math
import time
import numpy as np
import pytest
from tests.performance_guard import assert_throughput_guard
from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
def _run(neuron: MorrisLecarNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]
def _m_inf(v: float, v1: float, v2: float) -> float:
    return 0.5 * (1.0 + np.tanh((v - v1) / v2))
def _w_inf(v: float, v3: float, v4: float) -> float:
    return 0.5 * (1.0 + np.tanh((v - v3) / v4))
def _lam(v: float, v3: float, v4: float, phi: float) -> float:
    return phi * np.cosh((v - v3) / (2.0 * v4))

__all__ = ['math', 'time', 'np', 'pytest', 'assert_throughput_guard', 'MorrisLecarNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'isi', '_run', '_m_inf', '_w_inf', '_lam']
