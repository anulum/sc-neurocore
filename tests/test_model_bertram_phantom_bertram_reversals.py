# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBertramReversals from former test_model_bertram_phantom.py

"""Focused suite: TestBertramReversals from former test_model_bertram_phantom.py."""

from __future__ import annotations

from tests.model_bertram_phantom_support import *  # noqa: F403


class TestBertramReversals:
    def test_reversal_ordering(self):
        """e_k < e_l < e_ca (standard ionic ordering)."""
        n = BertramPhantomBurster()
        assert n.e_k < n.e_l < n.e_ca

    def test_ca_current_inward_at_rest(self):
        """I_Ca inward (negative) at rest: v=-50 < e_ca=25."""
        n = BertramPhantomBurster()
        m_inf = _boltz(n.v, n.v_m, n.s_m)
        i_ca = n.g_ca * m_inf * (n.v - n.e_ca)
        assert i_ca < 0

    def test_k_current_outward_at_rest(self):
        """I_K outward (positive) at rest: v=-50 > e_k=-75."""
        n = BertramPhantomBurster()
        n_inf = _boltz(n.v, n.v_n, n.s_n)
        i_k = n.g_k * n_inf * (n.v - n.e_k)
        assert i_k > 0

    def test_s1_s2_share_reversal(self):
        """Both slow currents use e_k as reversal potential."""
        n = BertramPhantomBurster()
        # From source: i_s1 = g_s1 * s1 * (v - e_k)
        # Both use e_k, not a separate reversal
        i_s1 = n.g_s1 * n.s1 * (n.v - n.e_k)
        i_s2 = n.g_s2 * n.s2 * (n.v - n.e_k)
        # Both outward at rest (v > e_k)
        assert i_s1 > 0 and i_s2 > 0
