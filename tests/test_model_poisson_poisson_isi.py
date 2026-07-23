# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonISI from former test_model_poisson.py

"""Focused suite: TestPoissonISI from former test_model_poisson.py."""

from __future__ import annotations

from tests.model_poisson_support import *  # noqa: F403

class TestPoissonISI:
    def test_isi_exponentially_distributed(self) -> None:
        """For Poisson process, ISI follows geometric distribution.

        Mean ISI = 1/p where p = λ·dt/1000.
        For rate=200Hz, dt=1ms: p=1-exp(-0.2), mean ISI=1/p steps.
        """
        n = PoissonNeuron(rate_hz=200.0, dt_ms=1.0)
        spike_times = []
        for t in range(200000):
            if n.step() == 1:
                spike_times.append(t)
        isis = np.diff(spike_times).astype(float)
        assert len(isis) >= 1000
        mean_isi = np.mean(isis)
        expected_mean = 1.0 / _poisson_step_probability(200.0, 1.0)
        assert abs(mean_isi - expected_mean) < 0.5, (
            f"mean ISI={mean_isi:.2f}, expected ≈{expected_mean:.1f}"
        )

    def test_cv_isi_near_one(self) -> None:
        """CV(ISI) ≈ 1 for Poisson process (geometric distribution)."""
        n = PoissonNeuron(rate_hz=200.0, dt_ms=1.0)
        spike_times = []
        for t in range(200000):
            if n.step() == 1:
                spike_times.append(t)
        isis = np.diff(spike_times).astype(float)
        cv = np.std(isis) / np.mean(isis)
        # Geometric CV = sqrt(1-p)/p ≈ 1 for small p
        # For p=0.2: CV = sqrt(0.8)/0.2 / (1/0.2) ≈ 0.894
        assert 0.7 < cv < 1.3, f"CV(ISI) = {cv:.3f}, expected ≈1"

    def test_no_refractory_period(self) -> None:
        """Consecutive spikes are possible (ISI=1 allowed)."""
        n = PoissonNeuron(rate_hz=800.0, dt_ms=1.0)  # p=0.8
        spike_times = []
        for t in range(10000):
            if n.step() == 1:
                spike_times.append(t)
        isis = np.diff(spike_times)
        assert 1 in isis, "Expected consecutive spikes (ISI=1) at high rate"
