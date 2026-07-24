# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerfectIntegratorISI from former test_model_perfect_integrator.py

"""Focused suite: TestPerfectIntegratorISI from former test_model_perfect_integrator.py."""

from __future__ import annotations

from tests.model_perfect_integrator_support import *  # noqa: F403


class TestPerfectIntegratorISI:
    """Inter-spike interval analysis — should be perfectly regular."""

    def test_constant_isi(self):
        """All ISIs identical (deterministic, no adaptation)."""
        n = PerfectIntegratorNeuron()
        times = _collect_spike_times(n, current=5.0, steps=2000)
        assert len(times) >= 10, "Not enough spikes to analyse ISI"
        isis = np.diff(times)
        # All ISIs should be identical (±0 for deterministic model)
        assert np.all(isis == isis[0]), f"ISI variability detected: unique ISIs = {np.unique(isis)}"

    def test_isi_matches_analytical(self):
        """Measured ISI matches C·(θ-V_reset) / (I·dt)."""
        n = PerfectIntegratorNeuron(c_m=2.0, v_threshold=3.0, v_reset=0.5)
        I = 10.0
        times = _collect_spike_times(n, current=I, steps=5000)
        assert len(times) >= 5
        measured_isi = np.median(np.diff(times))
        expected_isi = _analytical_isi_steps(
            I,
            n.c_m,
            n.v_threshold,
            n.v_reset,
            n.dt,
        )
        # Allow 1 step tolerance for floating-point rounding
        assert abs(measured_isi - round(expected_isi)) <= 1

    def test_cv_isi_zero(self):
        """Coefficient of variation of ISI = 0 (no jitter)."""
        n = PerfectIntegratorNeuron()
        times = _collect_spike_times(n, current=5.0, steps=5000)
        isis = np.diff(times).astype(float)
        cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0.0
        assert cv == 0.0, f"CV(ISI) = {cv}, expected 0.0"
