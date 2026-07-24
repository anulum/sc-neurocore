# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHHCurrentBalance from former test_model_hodgkin_huxley.py

"""Focused suite: TestHHCurrentBalance from former test_model_hodgkin_huxley.py."""

from __future__ import annotations

from tests.model_hodgkin_huxley_support import *  # noqa: F403


class TestHHCurrentBalance:
    def test_i_na_inward_at_rest(self):
        """I_Na at rest: g_Na·m³·h·(V-E_Na). V=-65 < E_Na=50 → negative (inward)."""
        n = HodgkinHuxleyNeuron()
        i_na = n.g_na * n.m**3 * n.h * (n.v - n.e_na)
        assert i_na < 0

    def test_i_k_outward_at_rest(self):
        """I_K: g_K·n⁴·(V-E_K). V=-65 > E_K=-77 → positive (outward)."""
        n = HodgkinHuxleyNeuron()
        i_k = n.g_k * n.n**4 * (n.v - n.e_k)
        assert i_k > 0
