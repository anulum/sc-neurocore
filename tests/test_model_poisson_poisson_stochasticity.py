# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonStochasticity from former test_model_poisson.py

"""Focused suite: TestPoissonStochasticity from former test_model_poisson.py."""

from __future__ import annotations

from tests.model_poisson_support import *  # noqa: F403


class TestPoissonStochasticity:
    def test_different_runs_differ(self) -> None:
        """Two neurons with distinct seeds produce distinct spike trains."""
        n1 = PoissonNeuron(rate_hz=200.0, seed=1)
        n2 = PoissonNeuron(rate_hz=200.0, seed=2)
        t1 = [n1.step() for _ in range(1000)]
        t2 = [n2.step() for _ in range(1000)]
        # Extremely unlikely to be identical
        assert t1 != t2

    def test_stateless(self) -> None:
        """Spike probability doesn't depend on history (memoryless)."""
        n = PoissonNeuron(rate_hz=200.0, dt_ms=1.0)
        # Run 10k steps, then measure rate for next 10k
        for _ in range(10000):
            n.step()
        spikes_after = sum(n.step() for _ in range(50000))
        p = _poisson_step_probability(200.0, 1.0)
        expected = 50000 * p
        sigma = np.sqrt(50000 * p * (1.0 - p))
        assert abs(spikes_after - expected) < 5 * sigma
