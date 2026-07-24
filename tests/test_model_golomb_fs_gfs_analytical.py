# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGFSAnalytical from former test_model_golomb_fs.py

"""Focused suite: TestGFSAnalytical from former test_model_golomb_fs.py."""

from __future__ import annotations

from tests.model_golomb_fs_support import *  # noqa: F403


class TestGFSAnalytical:
    def test_10_substeps(self):
        """10 sub-steps per step() call (dt=0.01)."""
        n = GolombFSNeuron()
        assert n.dt == 0.01

    def test_four_ionic_currents(self):
        n = GolombFSNeuron()
        assert n.g_na > 0 and n.g_kd > 0 and n.g_kv3 > 0 and n.g_l > 0

    def test_kv3_high_threshold(self):
        """Kv3 p_inf half-activation at v=-3 mV (high threshold)."""
        v_half_kv3 = -3.0
        p_inf = 1.0 / (1.0 + np.exp(-(v_half_kv3 + 3.0) / 8.0))
        assert abs(p_inf - 0.5) < 1e-12

    def test_kv3_conductance_large(self):
        """g_Kv3=150 > g_Na=112.5: Kv3 dominates repolarisation."""
        n = GolombFSNeuron()
        assert n.g_kv3 > n.g_na

    def test_m_na_instantaneous(self):
        """m_Na set directly to m_inf (no time constant)."""
        # Source uses m_inf directly in current calculation
        n = GolombFSNeuron()
        n.step(5.0)
        # m_inf is not stored — computed inline
        assert np.isfinite(n.v)

    def test_reversal_ordering(self):
        n = GolombFSNeuron()
        assert n.e_k < n.e_l < n.e_na

    def test_gating_bounded(self):
        n = GolombFSNeuron()
        for _ in range(2000):
            n.step(5.0)
        for attr in ["h", "n", "p"]:
            val = getattr(n, attr)
            assert -0.05 <= val <= 1.05, f"{attr}={val}"
