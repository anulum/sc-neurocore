# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWBGammaFrequency from former test_model_wang_buzsaki.py

"""Focused suite: TestWBGammaFrequency from former test_model_wang_buzsaki.py."""

from __future__ import annotations

from tests.model_wang_buzsaki_support import *  # noqa: F403

class TestWBGammaFrequency:
    """The model is designed for gamma-band (30–80 Hz) firing."""

    def test_gamma_band_at_moderate_current(self):
        """At I=0.5–1.0, firing frequency should be in gamma range (30–80 Hz).

        Each step() = 0.5 ms. ISI in steps × 0.5 ms = ISI_ms.
        Freq = 1000 / ISI_ms Hz.
        """
        n = WangBuzsakiNeuron()
        spikes = _run(n, current=1.0, steps=20000)
        assert len(spikes) >= 20
        isis = np.diff(spikes[5:])
        mean_isi_ms = np.mean(isis) * 0.5  # each step = 0.5 ms
        freq_hz = 1000.0 / mean_isi_ms
        assert 30 < freq_hz < 100, f"freq = {freq_hz:.0f} Hz, expected gamma range"

    def test_onset_frequency_near_30hz(self):
        """At threshold current, frequency should start near lower gamma."""
        n = WangBuzsakiNeuron()
        spikes = _run(n, current=0.5, steps=20000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[5:])
            freq = 1000.0 / (np.mean(isis) * 0.5)
            assert freq > 20, f"Onset freq = {freq:.0f} Hz"
