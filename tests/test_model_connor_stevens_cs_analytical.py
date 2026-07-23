# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCSAnalytical from former test_model_connor_stevens.py

"""Focused suite: TestCSAnalytical from former test_model_connor_stevens.py."""

from __future__ import annotations

from tests.model_connor_stevens_support import *  # noqa: F403

class TestCSAnalytical:
    def test_100_substeps_per_call(self):
        """dt=0.01 → 1/0.01 = 100 sub-steps per step() call."""
        n = ConnorStevensNeuron()
        assert int(1.0 / max(n.dt, 0.001)) == 100

    def test_four_ionic_currents(self):
        """I_Na, I_K, I_A, I_L — all conductances positive."""
        n = ConnorStevensNeuron()
        assert n.g_na > 0 and n.g_k > 0 and n.g_a > 0 and n.g_l > 0

    def test_a_current_conductance_dominant(self):
        """g_A=47.7 > g_K=20 — A-current is the signature feature."""
        n = ConnorStevensNeuron()
        assert n.g_a > n.g_k

    def test_reversal_ordering(self):
        """e_a < e_k < e_l < e_na."""
        n = ConnorStevensNeuron()
        assert n.e_a < n.e_k < n.e_l < n.e_na

    def test_gating_variables_bounded(self):
        """All gating variables should stay in [0, 1] range."""
        n = ConnorStevensNeuron()
        for _ in range(500):
            n.step(20.0)
        for attr in ["m", "h", "n", "a", "b"]:
            val = getattr(n, attr)
            assert -0.01 <= val <= 1.01, f"{attr}={val}"

    def test_a_type_delays_spike_onset(self):
        """A-current creates delay at rheobase — Type-I hallmark."""
        # With A-current (default)
        n_with_a = ConnorStevensNeuron()
        spikes_a = _run(n_with_a, current=8.0, steps=200)
        # Without A-current
        n_no_a = ConnorStevensNeuron(g_a=0.0)
        spikes_no_a = _run(n_no_a, current=8.0, steps=200)
        # Without A: should fire more easily
        assert len(spikes_no_a) >= len(spikes_a)
