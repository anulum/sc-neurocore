# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_prescott.py

from __future__ import annotations

"""Full pipeline test for PrescottNeuron (Prescott et al. 2008).

2D model with Type I/II/III excitability tunable via beta_w (slow
K⁺ nullcline shift). Slow oscillation regime at default parameters."""
import math
import numpy as np
import pytest
from sc_neurocore.neurons.models.prescott import PrescottNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count
def _run(neuron: PrescottNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]
def _prescott_rhs(
    neuron: PrescottNeuron, v: float, w: float, current: float
) -> tuple[float, float]:
    m_inf = 1.0 / (1.0 + math.exp(-(v + 20.0) / 15.0))
    w_inf = 1.0 / (1.0 + math.exp(-(v - neuron.beta_w) / neuron.gamma_w))
    i_fast = neuron.g_fast * m_inf * (v - neuron.e_fast)
    i_slow = neuron.g_slow * w * (v - neuron.e_slow)
    i_l = neuron.g_l * (v - neuron.e_l)
    return (
        -i_fast - i_slow - i_l + current,
        neuron.phi * (w_inf - w) / neuron.tau_w,
    )
def _prescott_rk4_after_call(neuron: PrescottNeuron, current: float) -> tuple[float, float]:
    v0, w0, dt = neuron.v, neuron.w, neuron.dt
    k1_v, k1_w = _prescott_rhs(neuron, v0, w0, current)
    k2_v, k2_w = _prescott_rhs(neuron, v0 + 0.5 * dt * k1_v, w0 + 0.5 * dt * k1_w, current)
    k3_v, k3_w = _prescott_rhs(neuron, v0 + 0.5 * dt * k2_v, w0 + 0.5 * dt * k2_w, current)
    k4_v, k4_w = _prescott_rhs(neuron, v0 + dt * k3_v, w0 + dt * k3_w, current)
    return (
        v0 + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0,
        w0 + dt * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0,
    )

__all__ = ['math', 'np', 'pytest', 'PrescottNeuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', '_run', '_prescott_rhs', '_prescott_rk4_after_call']
