# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBertramDualTimescale from former test_model_bertram_phantom.py

"""Focused suite: TestBertramDualTimescale from former test_model_bertram_phantom.py."""

from __future__ import annotations

from tests.model_bertram_phantom_support import *  # noqa: F403

class TestBertramDualTimescale:
    def test_tau_ratio(self):
        """tau_s2 / tau_s1 = 5 (ultra-slow vs slow)."""
        n = BertramPhantomBurster()
        assert n.tau_s2 / n.tau_s1 == 5.0

    def test_s1_faster_than_s2(self):
        """s1 moves more per step than s2 (same driving, shorter tau)."""
        n = BertramPhantomBurster()
        s1_0, s2_0 = n.s1, n.s2
        n.step(200.0)
        ds1 = abs(n.s1 - s1_0)
        ds2 = abs(n.s2 - s2_0)
        assert ds1 > ds2

    def test_s1_approaches_equilibrium_faster(self):
        """After many steps, s1 converges toward s1_inf faster than s2."""
        n = BertramPhantomBurster()
        # Drive at constant current, measure convergence
        for _ in range(10000):
            n.step(200.0)
        # s1_inf at current v
        s1_inf = _boltz(n.v, n.v_s1, n.s_s1)
        s2_inf = _boltz(n.v, n.v_s2, n.s_s2)
        # Relative distance from equilibrium
        err_s1 = abs(n.s1 - s1_inf)
        err_s2 = abs(n.s2 - s2_inf)
        # s1 should be closer to its equilibrium (faster dynamics)
        assert err_s1 < err_s2 or err_s1 < 0.05

    def test_slow_variables_bounded(self):
        """s1, s2 ∈ [0, 1] (Boltzmann targets are in [0, 1])."""
        n = BertramPhantomBurster()
        s1_vals, s2_vals = [], []
        for _ in range(100_000):
            n.step(200.0)
            s1_vals.append(n.s1)
            s2_vals.append(n.s2)
        assert min(s1_vals) >= -0.01 and max(s1_vals) <= 1.01
        assert min(s2_vals) >= -0.01 and max(s2_vals) <= 1.01
