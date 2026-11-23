# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopulationDiversity from former test_meta_plasticity.py

"""Focused suite: TestPopulationDiversity from former test_meta_plasticity.py."""

from __future__ import annotations

from sc_neurocore.meta_plasticity.meta_plasticity import (
    inject_diversity,
    population_diversity,
    RuleEvolver,
)


class TestPopulationDiversity:
    """Diversity of an evolving rule population."""

    def test_identical_population_zero(self) -> None:
        """Identical members have no diversity."""
        ev = RuleEvolver(population_size=4)
        assert population_diversity(ev) < 1e-6

    def test_diverse_population(self) -> None:
        """Different members have positive diversity."""
        ev = RuleEvolver(population_size=4, mutation_rate=1.0, mutation_scale=1.0)
        ev.population[0].stdp.lr = 0.001
        ev.population[1].stdp.lr = 0.05
        ev.population[2].stdp.lr = 0.1
        ev.population[3].stdp.tau_plus = 50.0
        assert population_diversity(ev) > 0

    def test_inject_diversity(self) -> None:
        """Injecting diversity does not lower it."""
        ev = RuleEvolver(population_size=4)
        d_before = population_diversity(ev)
        inject_diversity(ev, n_random=2)
        d_after = population_diversity(ev)
        assert d_after >= d_before

    def test_single_member_population_is_zero_diversity(self) -> None:
        """A single member has no diversity."""
        # A single individual has no pairs to compare, so diversity is 0.
        ev = RuleEvolver(population_size=1)
        assert population_diversity(ev) == 0.0
