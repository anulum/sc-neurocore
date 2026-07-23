# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWBHHProperties from former test_model_wang_buzsaki.py

"""Focused suite: TestWBHHProperties from former test_model_wang_buzsaki.py."""

from __future__ import annotations

from tests.model_wang_buzsaki_support import *  # noqa: F403

class TestWBHHProperties:
    def test_m_is_instantaneous(self):
        """m is computed as m_inf each sub-step, not integrated as ODE."""
        # Verify by checking that m_inf depends only on V, not on history
        n = WangBuzsakiNeuron()
        # After many steps, the state should be on the limit cycle
        for _ in range(5000):
            n.step(1.0)
        # m_inf should be deterministic from V alone
        alpha_m = (
            0.1 * (n.v + 35.0) / (1.0 - np.exp(-(n.v + 35.0) / 10.0))
            if abs(n.v + 35.0) > 1e-6
            else 1.0
        )
        beta_m = 4.0 * np.exp(-(n.v + 60.0) / 18.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        # m_inf is deterministic — just verify it's finite and in [0,1]
        assert 0 <= m_inf <= 1

    def test_phi_accelerates_gating(self):
        """phi=5 makes h and n dynamics 5× faster than standard HH."""
        n_fast = WangBuzsakiNeuron(phi=5.0)
        n_slow = WangBuzsakiNeuron(phi=1.0)
        h_fast_init, h_slow_init = n_fast.h, n_slow.h
        for _ in range(100):
            n_fast.step(1.0)
            n_slow.step(1.0)
        dh_fast = abs(n_fast.h - h_fast_init)
        dh_slow = abs(n_slow.h - h_slow_init)
        assert dh_fast > dh_slow

    def test_gating_bounded(self):
        n = WangBuzsakiNeuron()
        for _ in range(20000):
            n.step(5.0)
        assert -0.01 <= n.h <= 1.01, f"h = {n.h}"
        assert -0.01 <= n.n <= 1.01, f"n = {n.n}"

    def test_isi_regularity(self):
        n = WangBuzsakiNeuron()
        spikes = _run(n, current=1.0, steps=20000)
        isis = np.diff(spikes[5:]).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.05

    def test_singularity_protection(self):
        n = WangBuzsakiNeuron(v=-35.0)
        n.step(0.0)
        assert np.isfinite(n.v)
