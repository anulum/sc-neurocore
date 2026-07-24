# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestYamadaHHProperties from former test_model_yamada.py

"""Focused suite: TestYamadaHHProperties from former test_model_yamada.py."""

from __future__ import annotations

from tests.model_yamada_support import *  # noqa: F403


class TestYamadaHHProperties:
    def test_gating_bounded(self):
        n = YamadaNeuron()
        for _ in range(200000):
            n.step(50.0)
        assert -0.01 <= n.n <= 1.01, f"n = {n.n}"
        assert -0.01 <= n.q <= 1.01, f"q = {n.q}"

    def test_sigmoid_half_activations(self):
        """m_inf(-30) = 0.5, n_inf(-30) = 0.5, q_inf(-50) = 0.5."""
        m_inf = 1.0 / (1.0 + np.exp(-(-30.0 + 30.0) / 9.5))
        assert abs(m_inf - 0.5) < 1e-10
        n_inf = 1.0 / (1.0 + np.exp(-(-30.0 + 30.0) / 10.0))
        assert abs(n_inf - 0.5) < 1e-10
        q_inf = 1.0 / (1.0 + np.exp(-(-50.0 + 50.0) / 10.0))
        assert abs(q_inf - 0.5) < 1e-10

    def test_na_inactivation_via_n(self):
        """Na current uses (1-n) as inactivation: I_Na = g_Na·m_inf³·(1-n)·(V-E_Na)."""
        n = YamadaNeuron()
        m_inf = 1.0 / (1.0 + np.exp(-(n.v + 30.0) / 9.5))
        i_na = n.g_na * m_inf**3 * (1.0 - n.n) * (n.v - n.e_na)
        # At rest V=-60 < E_Na=60: (V-E_Na) < 0, m_inf small, (1-n)=0.9
        assert i_na < 0  # inward

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        n = YamadaNeuron(dt=dt)
        for _ in range(100000):
            n.step(50.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = YamadaNeuron()
            trace = [(n.step(50.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
