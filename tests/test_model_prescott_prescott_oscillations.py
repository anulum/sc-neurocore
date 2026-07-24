# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrescottOscillations from former test_model_prescott.py

"""Focused suite: TestPrescottOscillations from former test_model_prescott.py."""

from __future__ import annotations

from tests.model_prescott_support import *  # noqa: F403


class TestPrescottOscillations:
    def test_spontaneous_oscillation(self):
        """Model oscillates even at I=0 (slow relaxation oscillation)."""
        n = PrescottNeuron()
        spikes = _run(n, current=0.0, steps=100000)
        assert len(spikes) >= 3, f"Expected spontaneous oscillation, got {len(spikes)} spikes"

    def test_slow_isi(self):
        """ISI is on the order of thousands of steps (slow oscillator)."""
        n = PrescottNeuron()
        spikes = _run(n, current=50.0, steps=100000)
        assert len(spikes) >= 3
        isis = np.diff(spikes)
        mean_isi = np.mean(isis)
        assert mean_isi > 1000, f"Mean ISI={mean_isi:.0f}, expected >1000"

    def test_rate_increases_with_current(self):
        """More current → shorter ISI → more spikes."""
        n_low = PrescottNeuron()
        n_high = PrescottNeuron()
        s_low = len(_run(n_low, current=10.0, steps=100000))
        s_high = len(_run(n_high, current=200.0, steps=100000))
        assert s_high > s_low

    def test_voltage_oscillates(self):
        """Voltage should show large-amplitude oscillations."""
        n = PrescottNeuron()
        voltages = []
        for _ in range(50000):
            n.step(50.0)
            voltages.append(n.v)
        v_arr = np.array(voltages)
        v_range = v_arr.max() - v_arr.min()
        assert v_range > 20.0, f"V range = {v_range:.1f}, expected >20 mV"
