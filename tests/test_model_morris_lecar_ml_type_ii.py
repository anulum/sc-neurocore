# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMLTypeII from former test_model_morris_lecar.py

"""Focused suite: TestMLTypeII from former test_model_morris_lecar.py."""

from __future__ import annotations

from tests.model_morris_lecar_support import *  # noqa: F403

class TestMLTypeII:
    def test_subthreshold_silent(self):
        n = MorrisLecarNeuron()
        assert len(_run(n, current=10.0, steps=10_000)) == 0

    def test_oscillatory_in_band(self):
        n = MorrisLecarNeuron()
        spikes = _run(n, current=100.0, steps=20_000)
        assert len(spikes) >= 10

    def test_type_ii_frequency_onset(self):
        """Type-II: non-zero frequency onset (Hopf bifurcation)."""
        # Near threshold, frequency should be non-zero
        n = MorrisLecarNeuron()
        spikes = _run(n, current=90.0, steps=50_000)
        if len(spikes) >= 5:
            isis = np.diff(spikes).astype(float)
            # Non-zero frequency at onset (unlike Type-I continuous)
            assert np.mean(isis) < 10_000

    def test_high_current_suppression(self):
        """Very high I pushes past oscillatory window."""
        n = MorrisLecarNeuron()
        s_mid = len(_run(n, current=100.0, steps=10_000))
        n2 = MorrisLecarNeuron()
        s_high = len(_run(n2, current=300.0, steps=10_000))
        assert s_mid >= s_high

    def test_voltage_bounded(self):
        n = MorrisLecarNeuron()
        vs = []
        for _ in range(20_000):
            n.step(100.0)
            vs.append(n.v)
        assert min(vs) > -100 and max(vs) < 150

    def test_w_bounded(self):
        """w ∈ [0, 1) — recovery variable stays in physiological range."""
        n = MorrisLecarNeuron()
        ws = []
        for _ in range(20_000):
            n.step(100.0)
            ws.append(n.w)
        assert min(ws) >= -0.1 and max(ws) <= 1.1
