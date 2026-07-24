# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonRate from former test_model_poisson.py

"""Focused suite: TestPoissonRate from former test_model_poisson.py."""

from __future__ import annotations

from tests.model_poisson_support import *  # noqa: F403


class TestPoissonRate:
    def test_mean_rate_matches_lambda(self) -> None:
        """Over many trials, spike rate ≈ 1 - exp(-λ·dt/1000).

        At rate=100Hz, dt=1ms: P(spike) ≈ 0.09516.
        """
        n = PoissonNeuron(rate_hz=100.0, dt_ms=1.0)
        N = 100000
        spikes = sum(n.step() for _ in range(N))
        p = _poisson_step_probability(100.0, 1.0)
        expected = N * p
        # 5σ tolerance for statistical test
        sigma = np.sqrt(N * p * (1.0 - p))
        assert abs(spikes - expected) < 5 * sigma, (
            f"spikes={spikes}, expected={expected:.0f}, 5σ={5 * sigma:.0f}"
        )

    @pytest.mark.parametrize("rate_hz", [50.0, 100.0, 200.0, 500.0])
    def test_rate_proportional_to_lambda(self, rate_hz: float) -> None:
        """Spike count ∝ rate_hz."""
        n = PoissonNeuron(rate_hz=rate_hz, dt_ms=1.0)
        N = 50000
        spikes = sum(n.step() for _ in range(N))
        p = _poisson_step_probability(rate_hz, 1.0)
        expected = N * p
        sigma = np.sqrt(N * p * (1.0 - p))
        assert abs(spikes - expected) < 5 * sigma

    def test_higher_rate_more_spikes(self) -> None:
        """Monotonicity: higher λ → more spikes."""
        n_low = PoissonNeuron(rate_hz=50.0)
        n_high = PoissonNeuron(rate_hz=500.0)
        N = 50000
        s_low = sum(n_low.step() for _ in range(N))
        s_high = sum(n_high.step() for _ in range(N))
        assert s_high > s_low

    def test_zero_rate_no_spikes(self) -> None:
        """λ=0 → P(spike) = 0 → no spikes ever."""
        n = PoissonNeuron(rate_hz=0.0)
        spikes = sum(n.step() for _ in range(100000))
        assert spikes == 0

    def test_rate_override(self) -> None:
        """rate_override parameter overrides stored rate."""
        n = PoissonNeuron(rate_hz=100.0)
        # Override to 0 — no spikes
        spikes = sum(n.step(rate_override=0.0) for _ in range(10000))
        assert spikes == 0

    def test_negative_rate_override_uses_stored(self) -> None:
        """Negative rate_override → use stored rate_hz (API convention)."""
        n = PoissonNeuron(rate_hz=500.0, dt_ms=1.0)
        spikes = sum(n.step(rate_override=-1.0) for _ in range(10000))
        expected = 10000 * 0.5
        assert spikes > expected * 0.5  # should be near 5000
