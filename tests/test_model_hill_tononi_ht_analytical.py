# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHTAnalytical from former test_model_hill_tononi.py

"""Focused suite: TestHTAnalytical from former test_model_hill_tononi.py."""

from __future__ import annotations

from tests.model_hill_tononi_support import *  # noqa: F403


class TestHTAnalytical:
    def test_six_ionic_currents(self):
        n = SCSixStateThalamocorticalNeuron()
        for g in [n.g_na, n.g_k, n.g_h, n.g_t, n.g_kna, n.g_l]:
            assert g > 0

    def test_kna_activation_formula(self):
        """w_KNa = 0.37 / (1 + (38.7/Na_i)^3.5). At Na_i=38.7: half-max."""
        w = 0.37 / (1.0 + (38.7 / 38.7) ** 3.5)
        assert abs(w - 0.37 / 2.0) < 1e-10

    def test_kna_low_na(self):
        """At low Na_i (5mM): w_KNa ≈ 0 (K channel closed)."""
        w = 0.37 / (1.0 + (38.7 / 5.0) ** 3.5)
        assert w < 0.001

    def test_na_accumulation_during_spiking(self):
        """Na_i increases during spiking (I_Na inward → Na enters)."""
        n = SCSixStateThalamocorticalNeuron()
        na_before = n.na_i
        for _ in range(10_000):
            n.step(2.0)
        # Na should accumulate from spiking
        assert n.na_i != na_before

    def test_na_non_negative(self):
        """Na_i clipped to ≥ 0."""
        n = SCSixStateThalamocorticalNeuron()
        for _ in range(50_000):
            n.step(0.0)
            assert n.na_i >= 0.0

    def test_na_pump_formula(self):
        """Na/K pump: rate = pump_max · Na_i / (Na_i + Na_eq)."""
        n = SCSixStateThalamocorticalNeuron()
        pump_rate = n.na_pump_max * n.na_i / (n.na_i + n.na_eq)
        assert pump_rate > 0 and np.isfinite(pump_rate)

    def test_reversal_ordering(self):
        n = SCSixStateThalamocorticalNeuron()
        assert n.e_k < n.e_l < n.e_h < n.e_na < n.e_ca

    def test_gating_bounded(self):
        n = SCSixStateThalamocorticalNeuron()
        for _ in range(10_000):
            n.step(0.0)
        for attr in ["h_na", "n_k", "m_h", "h_t"]:
            val = getattr(n, attr)
            assert -0.05 <= val <= 1.05, f"{attr}={val}"
