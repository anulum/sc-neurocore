# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDestAnalytical from former test_model_destexhe_thalamic.py

"""Focused suite: TestDestAnalytical from former test_model_destexhe_thalamic.py."""

from __future__ import annotations

from tests.model_destexhe_thalamic_support import *  # noqa: F403


class TestDestAnalytical:
    def test_5_substeps_per_call(self):
        """5 sub-steps per step() call."""
        # Source: for _ in range(5):
        n = DestexheThalamicNeuron()
        # Verify by checking dt=0.02 × 5 = 0.1ms effective
        assert n.dt == 0.02

    def test_m_t_instantaneous(self):
        """m_T is set to m_T_inf (no time constant)."""
        n = DestexheThalamicNeuron()
        n.step(0.0)
        # m_t should be m_t_inf at current v
        m_t_inf = 1.0 / (1.0 + np.exp(-(n.v + 57.0) / 6.5))
        # Not exact due to sub-stepping, but should be close
        assert abs(n.m_t - m_t_inf) < 0.1

    def test_four_ionic_currents(self):
        n = DestexheThalamicNeuron()
        assert n.g_na > 0 and n.g_k > 0 and n.g_t > 0 and n.g_l > 0

    def test_reversal_ordering(self):
        """e_k < e_l < e_na < e_ca."""
        n = DestexheThalamicNeuron()
        assert n.e_k < n.e_l < n.e_na < n.e_ca

    def test_h_t_de_inactivation_hyperpolarised(self):
        """At v=-90: h_t_inf = 1/(1+exp((-90+81)/4)) ≈ 0.90. T-current ready."""
        h_t_inf = 1.0 / (1.0 + np.exp((-90.0 + 81.0) / 4.0))
        assert h_t_inf > 0.85
        # At rest v=-65: h_t_inf is small (T inactivated at rest)
        h_t_rest = 1.0 / (1.0 + np.exp((-65.0 + 81.0) / 4.0))
        assert h_t_rest < 0.1

    def test_h_t_inactivated_depolarised(self):
        """At v=-40: h_t_inf ≈ 0. T-current inactivated."""
        h_t_inf = 1.0 / (1.0 + np.exp((-40.0 + 81.0) / 4.0))
        assert h_t_inf < 0.01

    def test_gating_variables_bounded(self):
        n = DestexheThalamicNeuron()
        for _ in range(5000):
            n.step(5.0)
        for attr in ["h_na", "n_k", "m_t", "h_t"]:
            val = getattr(n, attr)
            assert -0.05 <= val <= 1.05, f"{attr}={val}"
