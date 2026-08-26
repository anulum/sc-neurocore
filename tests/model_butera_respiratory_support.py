# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_butera_respiratory.py

from __future__ import annotations

"""Compatibility tests for the retained SC respiratory recurrence.

Pre-Bötzinger respiratory neuron with persistent Na⁺ current and
slow h_nap inactivation. Bursting at high current."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.sc_unit_capacitance_respiratory import (
    SCUnitCapacitanceRespiratoryNeuron as ButeraRespiratoryNeuron,
)
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


def _butera_rates(v: float) -> tuple[float, float, float, float, float, float]:
    m_na_inf = 1.0 / (1.0 + np.exp(-(v + 34.0) / 5.0))
    m_nap_inf = 1.0 / (1.0 + np.exp(-(v + 40.0) / 6.0))
    h_nap_inf = 1.0 / (1.0 + np.exp((v + 48.0) / 6.0))
    n_inf = 1.0 / (1.0 + np.exp(-(v + 29.0) / 4.0))
    tau_n = max(10.0 / max(np.cosh((v + 29.0) / 8.0), 1e-12), 0.01)
    tau_h = max(10000.0 / max(np.cosh((v + 48.0) / 12.0), 1e-12), 0.1)
    return m_na_inf, m_nap_inf, h_nap_inf, n_inf, tau_n, tau_h


def _butera_derivatives(
    state: tuple[float, float, float], current: float, params: dict[str, float]
) -> tuple[float, float, float]:
    v, n, h_nap = state
    m_na_inf, m_nap_inf, h_nap_inf, n_inf, tau_n, tau_h = _butera_rates(v)
    i_na = params["g_na"] * m_na_inf**3 * (1.0 - n) * (v - params["e_na"])
    i_nap = params["g_nap"] * m_nap_inf * h_nap * (v - params["e_na"])
    i_k = params["g_k"] * n**4 * (v - params["e_k"])
    i_l = params["g_l"] * (v - params["e_l"])
    return (
        -i_na - i_nap - i_k - i_l + current,
        (n_inf - n) / tau_n,
        (h_nap_inf - h_nap) / tau_h,
    )


def _butera_reference_rk4(
    neuron: ButeraRespiratoryNeuron, current: float
) -> tuple[float, float, float]:
    state = (neuron.v, neuron.n, neuron.h_nap)
    params = {
        "g_na": neuron.g_na,
        "g_nap": neuron.g_nap,
        "g_k": neuron.g_k,
        "g_l": neuron.g_l,
        "e_na": neuron.e_na,
        "e_k": neuron.e_k,
        "e_l": neuron.e_l,
    }
    dt = neuron.dt
    k1 = _butera_derivatives(state, current, params)
    k2 = _butera_derivatives(tuple(s + 0.5 * dt * k for s, k in zip(state, k1)), current, params)
    k3 = _butera_derivatives(tuple(s + 0.5 * dt * k for s, k in zip(state, k2)), current, params)
    k4 = _butera_derivatives(tuple(s + dt * k for s, k in zip(state, k3)), current, params)
    return tuple(
        s + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0 for s, a, b, c, d in zip(state, k1, k2, k3, k4)
    )


__all__ = [
    "np",
    "pytest",
    "ButeraRespiratoryNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "firing_rate",
    "spike_count",
    "isi",
    "_butera_rates",
    "_butera_derivatives",
    "_butera_reference_rk4",
]
