# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRMFI from former test_model_spike_response.py

"""Focused suite: TestSRMFI from former test_model_spike_response.py."""

from __future__ import annotations

from tests.model_spike_response_support import *  # noqa: F403


class TestSRMFI:
    def test_subthreshold_silent(self):
        n = SpikeResponseNeuron()
        assert len(_run(n, current=5.0, steps=10000)) == 0

    def test_suprathreshold_fires(self):
        n = SpikeResponseNeuron()
        assert len(_run(n, current=10.0, steps=10000)) >= 100

    def test_monotonic_fi(self):
        rates = []
        for I in [8.0, 10.0, 15.0, 20.0]:
            n = SpikeResponseNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_fi_cross_current_comparison(self):
        """Verify rate ratios are consistent: stronger input → proportionally more spikes."""
        n10 = SpikeResponseNeuron()
        n20 = SpikeResponseNeuron()
        s10 = len(_run(n10, current=10.0, steps=10000))
        s20 = len(_run(n20, current=20.0, steps=10000))
        # Ratio should be > 1 (monotonic)
        assert s20 > s10
        # Not exactly 2× because ISI depends nonlinearly on η recovery
        ratio = s20 / s10
        assert 1.2 < ratio < 3.0, f"Rate ratio f(20)/f(10) = {ratio:.2f}"
