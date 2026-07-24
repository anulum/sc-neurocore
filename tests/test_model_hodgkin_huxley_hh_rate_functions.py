# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHHRateFunctions from former test_model_hodgkin_huxley.py

"""Focused suite: TestHHRateFunctions from former test_model_hodgkin_huxley.py."""

from __future__ import annotations

from tests.model_hodgkin_huxley_support import *  # noqa: F403


class TestHHRateFunctions:
    """Verify α and β rate functions at specific V values."""

    def test_alpha_m_singularity_protected(self):
        """At V=-40, d=0 → returns 1.0 (L'Hôpital limit)."""
        n = HodgkinHuxleyNeuron()
        am = n._alpha_m(-40.0)
        assert abs(am - 1.0) < 1e-6

    def test_alpha_n_singularity_protected(self):
        """At V=-55, d=0 → returns 0.1 (L'Hôpital limit)."""
        n = HodgkinHuxleyNeuron()
        an = n._alpha_n(-55.0)
        assert abs(an - 0.1) < 1e-6

    def test_beta_m_formula(self):
        """β_m(V) = 4·exp(-(V+65)/18). At V=-65: β_m = 4."""
        n = HodgkinHuxleyNeuron()
        bm = n._beta_m(-65.0)
        assert abs(bm - 4.0) < 1e-10

    def test_alpha_h_formula(self):
        """α_h(V) = 0.07·exp(-(V+65)/20). At V=-65: α_h = 0.07."""
        n = HodgkinHuxleyNeuron()
        ah = n._alpha_h(-65.0)
        assert abs(ah - 0.07) < 1e-10

    def test_gating_bounded(self):
        """m, h, n should stay in [0, 1]."""
        n = HodgkinHuxleyNeuron()
        for _ in range(5000):
            n.step(10.0)
        for name, val in [("m", n.m), ("h", n.h), ("n", n.n)]:
            assert -0.01 <= val <= 1.01, f"{name} = {val:.6f}"
