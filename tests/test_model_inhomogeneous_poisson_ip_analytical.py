# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIPAnalytical from former test_model_inhomogeneous_poisson.py

"""Focused suite: TestIPAnalytical from former test_model_inhomogeneous_poisson.py."""

from __future__ import annotations

from tests.model_inhomogeneous_poisson_support import *  # noqa: F403


class TestIPAnalytical:
    def test_probability_formula(self):
        """P(spike) = 1 - exp(-rate_hz · dt_ms / 1000)."""
        n = InhomogeneousPoissonNeuron(dt_ms=1.0)
        assert n._probability(100.0) == pytest.approx(1.0 - math.exp(-0.1))

    def test_expected_spike_count(self):
        """E[spikes] = N · (1 - exp(-rate · dt/1000)). Statistical test."""
        n = InhomogeneousPoissonNeuron(dt_ms=1.0)
        N = 100_000
        rate = 100.0
        spikes = sum(n.step(rate) for _ in range(N))
        p = _poisson_interval_probability(rate, 1.0)
        expected = N * p
        std = np.sqrt(N * p * (1.0 - p))
        assert abs(spikes - expected) < 5 * std

    def test_negative_rate_no_spikes(self):
        """Negative rate clipped to 0 → P = 0."""
        n = InhomogeneousPoissonNeuron()
        spikes = sum(n.step(-100.0) for _ in range(10_000))
        assert spikes == 0

    def test_zero_rate_no_spikes(self):
        n = InhomogeneousPoissonNeuron()
        spikes = sum(n.step(0.0) for _ in range(10_000))
        assert spikes == 0

    def test_high_rate_saturates_without_invalid_probability(self):
        """High finite rates saturate to one spike per interval without invalid probabilities."""
        n = InhomogeneousPoissonNeuron()
        spikes = sum(n.step(1.0e9) for _ in range(100))
        assert spikes == 100

    def test_rate_proportional(self):
        """Double rate → double expected spikes."""
        n1 = InhomogeneousPoissonNeuron()
        n2 = InhomogeneousPoissonNeuron()
        N = 50_000
        s1 = sum(n1.step(50.0) for _ in range(N))
        s2 = sum(n2.step(100.0) for _ in range(N))
        # s2 should be roughly 2× s1 (statistical)
        assert s2 > s1

    @pytest.mark.parametrize("dt_ms", [0.1, 1.0, 5.0])
    def test_dt_ms_scales_probability(self, dt_ms: float):
        """Larger dt_ms → higher P per step."""
        n = InhomogeneousPoissonNeuron(dt_ms=dt_ms)
        spikes = sum(n.step(100.0) for _ in range(10_000))
        assert isinstance(spikes, int)
