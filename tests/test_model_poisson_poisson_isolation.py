# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonIsolation from former test_model_poisson.py

"""Focused suite: TestPoissonIsolation from former test_model_poisson.py."""

from __future__ import annotations

from tests.model_poisson_support import *  # noqa: F403

class TestPoissonIsolation:
    def test_construction_defaults(self) -> None:
        n = PoissonNeuron()
        assert n.rate_hz == 100.0
        assert n.dt_ms == 1.0

    def test_step_returns_binary(self) -> None:
        n = PoissonNeuron()
        assert n.step() in (0, 1)

    def test_rng_initialised(self) -> None:
        """Internal RNG should be initialised after construction."""
        n = PoissonNeuron()
        assert hasattr(n, "_rng")

    def test_reset_replays_the_seeded_event_stream(self) -> None:
        """Reset restores execution state even though no membrane state exists."""
        n = PoissonNeuron(seed=0xACE1)
        first = [n.step() for _ in range(1000)]
        assert n.rng_state != n.initial_seed
        n.reset()
        assert n.rng_state == n.initial_seed == 0xACE1
        assert [n.step() for _ in range(1000)] == first

    def test_default_seed_is_reproducible_and_none_requests_entropy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitted seeds replay; explicit ``None`` constructs independent streams."""
        default_a = PoissonNeuron(rate_hz=200.0)
        default_b = PoissonNeuron(rate_hz=200.0)
        assert [default_a.step() for _ in range(1000)] == [default_b.step() for _ in range(1000)]

        entropy_words = iter((0, 1))
        monkeypatch.setattr(
            "sc_neurocore.neurons._stochastic_threshold.secrets.randbelow",
            lambda _upper: next(entropy_words),
        )
        entropy_a = PoissonNeuron(rate_hz=200.0, seed=None)
        entropy_b = PoissonNeuron(rate_hz=200.0, seed=None)
        assert (entropy_a.initial_seed, entropy_b.initial_seed) == (1, 2)
