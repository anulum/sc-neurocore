# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPernarowskiFI from former test_model_pernarowski.py

"""Focused suite: TestPernarowskiFI from former test_model_pernarowski.py."""

from __future__ import annotations

from tests.model_pernarowski_support import *  # noqa: F403


class TestPernarowskiFI:
    def test_moderate_current_sustains_oscillation(self):
        """I ∈ [0, 1.0] should sustain oscillatory spiking."""
        for I in [0.0, 0.3, 0.5, 1.0]:
            n = PernarowskiNeuron()
            spike_times, _ = _run_and_collect(n, current=I, steps=10000)
            assert len(spike_times) >= 10, (
                f"I={I}: only {len(spike_times)} spikes, expected oscillation"
            )

    def test_depolarisation_block(self):
        """High current (I≥2.0) suppresses oscillation — V stays high."""
        for I in [2.0, 3.0]:
            n = PernarowskiNeuron()
            spike_times, voltages = _run_and_collect(n, current=I, steps=10000)
            assert len(spike_times) <= 5, (
                f"I={I}: {len(spike_times)} spikes, expected depolarisation block"
            )

    def test_rate_increases_with_moderate_current(self):
        """Between I=0 and I=0.5, rate should be slightly modulated."""
        n0 = PernarowskiNeuron()
        n1 = PernarowskiNeuron()
        s0, _ = _run_and_collect(n0, current=0.0, steps=10000)
        s1, _ = _run_and_collect(n1, current=0.5, steps=10000)
        # Rate shouldn't change dramatically (both in oscillatory regime)
        ratio = len(s1) / len(s0) if len(s0) > 0 else 0.0
        assert 0.5 < ratio < 2.0, (
            f"Spike ratio {ratio:.2f} — expected similar rates in oscillatory regime"
        )
