# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A population of stochastic neurons is not one neuron repeated

"""Every neuron of a seeded population gets its own stream, reproducibly.

The defect these cases hold closed: ``Population`` built every neuron with
identical keyword arguments, so a model carrying a ``seed`` gave the whole
population one random stream. Measured before the fix, eight
``EscapeRateNeuron`` driven at 60 units for 60 steps produced eight identical
trains of nine spikes each — a group whose variance, synchrony and population
rate all described a single neuron.

The cases are written against behaviour rather than against the derivation:
what matters is that the trains differ, that the same request gives the same
population on any machine, and that a caller asking for entropy still gets it.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.network.population import Population, _seed_default
from sc_neurocore.network.population_seeds import (
    LFSR16_SEED_DOMAIN,
    derive_population_seeds,
)

DRIVE = 60.0
STEPS = 60
PARAMS = {"rho_0": 0.05, "seed": 4242}


def _trains(population: Population) -> np.ndarray:
    """Run a population under a constant drive and return its spike trains.

    Parameters
    ----------
    population : Population
        The population to run.

    Returns
    -------
    numpy.ndarray
        A ``(STEPS, n)`` array of binary spikes, one column per neuron.
    """
    drive = np.full(population.n, DRIVE)
    return np.array([population.step_all(drive) for _ in range(STEPS)])


class TestSeededPopulationsAreGroups:
    def test_eight_seeded_neurons_do_not_produce_one_train_eight_times(self) -> None:
        trains = _trains(Population("EscapeRateNeuron", 8, params=dict(PARAMS)))

        first = trains[:, 0]
        assert not all(np.array_equal(first, trains[:, i]) for i in range(1, 8))

    def test_every_neuron_holds_a_seed_of_its_own(self) -> None:
        population = Population("EscapeRateNeuron", 8, params=dict(PARAMS))

        seeds = [neuron.seed for neuron in population.neurons]
        assert len(set(seeds)) == len(seeds)

    def test_a_population_with_no_seed_named_is_still_a_group(self) -> None:
        # The model's own default is a fixed constant, so leaving the seed
        # unnamed used to share one stream as surely as naming it did.
        population = Population("EscapeRateNeuron", 6, params={"rho_0": 0.05})

        seeds = [neuron.seed for neuron in population.neurons]
        assert len(set(seeds)) == len(seeds)

    def test_the_same_request_builds_the_same_population(self) -> None:
        # Reproducibility is the point of seeding at all: deriving must not
        # trade one defect for an unrepeatable experiment.
        first = Population("EscapeRateNeuron", 6, params=dict(PARAMS))
        second = Population("EscapeRateNeuron", 6, params=dict(PARAMS))

        assert [n.seed for n in first.neurons] == [n.seed for n in second.neurons]
        assert np.array_equal(_trains(first), _trains(second))

    def test_a_different_base_seed_builds_a_different_population(self) -> None:
        first = Population("EscapeRateNeuron", 6, params=dict(PARAMS))
        second = Population("EscapeRateNeuron", 6, params={**PARAMS, "seed": 4243})

        assert [n.seed for n in first.neurons] != [n.seed for n in second.neurons]

    def test_an_explicit_none_still_asks_each_model_for_entropy(self) -> None:
        # `None` is a request for unpredictability. Deriving over it would
        # answer with a reproducible sequence instead.
        population = Population("PoissonNeuron", 4, params={"seed": None})

        seeds = [neuron.initial_seed for neuron in population.neurons]
        assert len(set(seeds)) == len(seeds)

    def test_a_model_without_a_seed_is_built_unchanged(self) -> None:
        population = Population("SCLapicqueLIFNeuron", 3)

        assert population.n == 3
        assert all(not hasattr(neuron, "seed") for neuron in population.neurons)

    def test_an_empty_population_needs_no_seeds(self) -> None:
        assert Population("EscapeRateNeuron", 0).n == 0


class TestTheDerivation:
    def test_it_yields_one_distinct_seed_per_neuron(self) -> None:
        seeds = derive_population_seeds(4242, 512)

        assert len(set(seeds)) == 512

    def test_every_seed_is_inside_the_narrowest_domain_in_this_build(self) -> None:
        # LFSR16 rejects a seed outside [0, 65535] and treats zero as a request
        # for the documented fallback, so a derived zero would silently give two
        # neurons the same stream.
        assert all(1 <= seed <= LFSR16_SEED_DOMAIN for seed in derive_population_seeds(1, 4096))

    def test_it_is_reproducible(self) -> None:
        assert derive_population_seeds(7, 32) == derive_population_seeds(7, 32)

    def test_a_prefix_of_a_larger_population_is_the_smaller_one(self) -> None:
        # Growing a population must not renumber the neurons already in it.
        assert derive_population_seeds(7, 8) == derive_population_seeds(7, 32)[:8]

    def test_it_fills_the_whole_domain_without_repeating(self) -> None:
        seeds = derive_population_seeds(11, LFSR16_SEED_DOMAIN)

        assert len(set(seeds)) == LFSR16_SEED_DOMAIN

    def test_it_refuses_a_population_larger_than_the_domain(self) -> None:
        # Refusing beats handing two neurons the same stream.
        with pytest.raises(ValueError, match="distinct seeds"):
            derive_population_seeds(1, LFSR16_SEED_DOMAIN + 1)

    def test_it_refuses_a_negative_count(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            derive_population_seeds(1, -1)

    def test_no_neurons_needs_no_seeds(self) -> None:
        assert derive_population_seeds(1, 0) == []


class TestTheSeedDefaultLookup:
    def test_a_factory_with_no_seed_parameter_names_no_default(self) -> None:
        assert _seed_default(lambda tau: tau) is None

    def test_a_factory_whose_seed_has_no_default_names_none(self) -> None:
        assert _seed_default(lambda seed: seed) is None

    def test_a_seed_defaulting_to_entropy_is_left_alone(self) -> None:
        # `None` is the model's way of asking for independent entropy, and a
        # population must not turn that into a reproducible sequence.
        assert _seed_default(lambda seed=None: seed) is None

    def test_a_boolean_default_is_not_a_seed(self) -> None:
        assert _seed_default(lambda seed=True: seed) is None

    def test_an_integer_default_is_the_base_a_population_derives_from(self) -> None:
        assert _seed_default(lambda seed=44257: seed) == 44257

    def test_a_builtin_without_a_signature_is_given_no_seed(self) -> None:
        # `inspect` raises ValueError for a builtin type; a factory that cannot
        # be asked about a seed does not receive one.
        assert _seed_default(int) is None

    def test_something_that_is_not_callable_is_given_no_seed(self) -> None:
        # TypeError rather than ValueError, and the same answer.
        assert _seed_default("not a factory") is None
