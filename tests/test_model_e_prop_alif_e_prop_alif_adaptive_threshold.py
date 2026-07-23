# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPropALIFAdaptiveThreshold from former test_model_e_prop_alif.py

"""Focused suite: TestEPropALIFAdaptiveThreshold from former test_model_e_prop_alif.py."""

from __future__ import annotations

from tests.model_e_prop_alif_support import *  # noqa: F403

class TestEPropALIFAdaptiveThreshold:
    """Core: θ(t) = θ_base + β·a(t). a increments on spike, decays with tau_a."""

    def test_a_increments_on_spike(self):
        n = EPropALIFNeuron()
        a_before = n.a
        for _ in range(10000):
            if n.step(0.5) == 1:
                assert n.a > a_before
                break
        else:
            raise AssertionError("No spike")

    def test_a_decays_between_spikes(self):
        n = EPropALIFNeuron()
        n.a = 5.0
        n.step(0.0)  # subthreshold, a decays
        assert n.a < 5.0

    def test_threshold_increases_with_a(self):
        """Effective threshold = θ_base + β·a. Higher a → harder to spike."""
        n = EPropALIFNeuron()
        # After many spikes, a is large → threshold high → ISI long
        spikes = _run(n, current=0.2, steps=5000)
        if len(spikes) >= 10:
            isis = np.diff(spikes)
            assert isis[-1] > isis[0], "ISI should lengthen (adaptation)"

    def test_isi_lengthens(self):
        """Early ISI < late ISI (adaptation effect)."""
        n = EPropALIFNeuron()
        spikes = _run(n, current=0.2, steps=5000)
        if len(spikes) >= 10:
            early = np.mean(np.diff(spikes[:5]))
            late = np.mean(np.diff(spikes[-5:]))
            assert late > early

    def test_no_adaptation_when_beta_zero(self):
        """β=0: threshold is constant → no ISI lengthening."""
        n = EPropALIFNeuron(beta=0.0)
        spikes = _run(n, current=0.2, steps=5000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.05, f"CV = {cv:.4f} with β=0"
