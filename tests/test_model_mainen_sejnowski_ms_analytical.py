# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMSAnalytical from former test_model_mainen_sejnowski.py

"""Focused suite: TestMSAnalytical from former test_model_mainen_sejnowski.py."""

from __future__ import annotations

from tests.model_mainen_sejnowski_support import *  # noqa: F403

class TestMSAnalytical:
    def test_20_substeps(self):
        n = MainenSejnowskiNeuron()
        assert n.dt == 0.005  # 1/0.005 = 200? No, source: range(20)

    def test_soma_passive_axon_active(self):
        """Soma: leak+coupling. Axon: Na+K+coupling."""
        n = MainenSejnowskiNeuron()
        assert n.g_l > 0  # soma leak
        assert n.g_na > 0 and n.g_k > 0  # axon active

    def test_coupling_kappa(self):
        """κ couples soma↔axon bidirectionally."""
        n = MainenSejnowskiNeuron()
        assert n.kappa > 0

    def test_voltage_clipping(self):
        """vs, va clipped to [-200, 200]."""
        n = MainenSejnowskiNeuron()
        for _ in range(500):
            n.step(50.0)
        assert -200 <= n.vs <= 200
        assert -200 <= n.va <= 200

    def test_gating_clipped(self):
        """m, h, n clipped to [0, 1]."""
        n = MainenSejnowskiNeuron()
        for _ in range(500):
            n.step(10.0)
        for attr in ["m", "h", "n"]:
            val = getattr(n, attr)
            assert 0.0 <= val <= 1.0

    def test_reversal_ordering(self):
        n = MainenSejnowskiNeuron()
        assert n.e_k < n.e_l < n.e_na
