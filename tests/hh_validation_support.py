# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_hh_validation.py

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron


def simulate_hh(current: float, duration_ms: float = 50, dt: float = 0.01):
    """Reference HH integration returning (V, m, h, n) arrays."""
    g_Na, g_K, g_L = 120.0, 36.0, 0.3
    E_Na, E_K, E_L = 50.0, -77.0, -54.4

    def alpha_m(v):
        d = v + 40.0
        return np.where(np.abs(d) < 1e-7, 1.0, 0.1 * d / (1 - np.exp(-d / 10)))

    def beta_m(v):
        return 4.0 * np.exp(-(v + 65) / 18)

    def alpha_h(v):
        return 0.07 * np.exp(-(v + 65) / 20)

    def beta_h(v):
        return 1.0 / (1 + np.exp(-(v + 35) / 10))

    def alpha_n(v):
        d = v + 55.0
        return np.where(np.abs(d) < 1e-7, 0.1, 0.01 * d / (1 - np.exp(-d / 10)))

    def beta_n(v):
        return 0.125 * np.exp(-(v + 65) / 80)

    n_steps = int(duration_ms / dt)
    V, M, H, N = [np.zeros(n_steps) for _ in range(4)]
    V[0], M[0], H[0], N[0] = -65.0, 0.05, 0.6, 0.32

    for i in range(1, n_steps):
        v, m, h, n = V[i - 1], M[i - 1], H[i - 1], N[i - 1]
        M[i] = m + (alpha_m(v) * (1 - m) - beta_m(v) * m) * dt
        H[i] = h + (alpha_h(v) * (1 - h) - beta_h(v) * h) * dt
        N[i] = n + (alpha_n(v) * (1 - n) - beta_n(v) * n) * dt
        i_na = g_Na * M[i] ** 3 * H[i] * (v - E_Na)
        i_k = g_K * N[i] ** 4 * (v - E_K)
        i_l = g_L * (v - E_L)
        V[i] = v + (-i_na - i_k - i_l + current) * dt

    return V, M, H, N


__all__ = ["np", "HodgkinHuxleyNeuron", "simulate_hh"]
