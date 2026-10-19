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
from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron
from sc_neurocore.neurons.models.poisson import PoissonNeuron
from sc_neurocore.neurons.seed_domain import (
    LFSR16_ALIASES,
    LFSR16_DOMAIN,
    seed_aliases,
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
    """A seeded population behaves as a group of neurons with their own streams."""

    def test_eight_seeded_neurons_do_not_produce_one_train_eight_times(self) -> None:
        """Eight escape-rate neurons under one drive give different spike trains."""
        trains = _trains(Population("EscapeRateNeuron", 8, params=dict(PARAMS)))

        first = trains[:, 0]
        assert not all(np.array_equal(first, trains[:, i]) for i in range(1, 8))

    def test_every_neuron_holds_a_seed_of_its_own(self) -> None:
        """Every neuron of a seeded population carries a different seed."""
        population = Population("EscapeRateNeuron", 8, params=dict(PARAMS))

        seeds = [neuron.seed for neuron in population.neurons]
        assert len(set(seeds)) == len(seeds)

    def test_a_population_with_no_seed_named_is_still_a_group(self) -> None:
        """The model's fixed default seed is derived per neuron as a named one is."""
        # The model's own default is a fixed constant, so leaving the seed
        # unnamed used to share one stream as surely as naming it did.
        population = Population("EscapeRateNeuron", 6, params={"rho_0": 0.05})

        seeds = [neuron.seed for neuron in population.neurons]
        assert len(set(seeds)) == len(seeds)

    def test_the_same_request_builds_the_same_population(self) -> None:
        """The same request gives the same seeds and the same spike trains."""
        # Reproducibility is the point of seeding at all: deriving must not
        # trade one defect for an unrepeatable experiment.
        first = Population("EscapeRateNeuron", 6, params=dict(PARAMS))
        second = Population("EscapeRateNeuron", 6, params=dict(PARAMS))

        assert [n.seed for n in first.neurons] == [n.seed for n in second.neurons]
        assert np.array_equal(_trains(first), _trains(second))

    def test_a_different_base_seed_builds_a_different_population(self) -> None:
        """A different base seed gives different neuron seeds."""
        first = Population("EscapeRateNeuron", 6, params=dict(PARAMS))
        second = Population("EscapeRateNeuron", 6, params={**PARAMS, "seed": 4243})

        assert [n.seed for n in first.neurons] != [n.seed for n in second.neurons]

    def test_an_explicit_none_still_asks_each_model_for_entropy(self) -> None:
        """``seed=None`` is passed through, so every neuron draws its own entropy."""
        # `None` is a request for unpredictability. Deriving over it would
        # answer with a reproducible sequence instead.
        population = Population("PoissonNeuron", 4, params={"seed": None})

        seeds = [neuron.initial_seed for neuron in population.neurons]
        assert len(set(seeds)) == len(seeds)

    def test_a_model_without_a_seed_is_built_unchanged(self) -> None:
        """A model without a seed parameter is built without one."""
        population = Population("SCLapicqueLIFNeuron", 3)

        assert population.n == 3
        assert all(not hasattr(neuron, "seed") for neuron in population.neurons)

    def test_an_empty_population_needs_no_seeds(self) -> None:
        """A population of zero neurons is built without deriving seeds."""
        assert Population("EscapeRateNeuron", 0).n == 0


class TestTheDerivation:
    """``derive_population_seeds`` in the universal domain ``[1, 65535]``."""

    def test_it_yields_one_distinct_seed_per_neuron(self) -> None:
        """512 neurons receive 512 distinct seeds."""
        seeds = derive_population_seeds(4242, 512)

        assert len(set(seeds)) == 512

    def test_every_seed_is_inside_the_narrowest_domain_in_this_build(self) -> None:
        """Every derived seed lies in ``[1, 65535]``, so none is the aliased zero."""
        # LFSR16 rejects a seed outside [0, 65535] and treats zero as a request
        # for the documented fallback, so a derived zero would silently give two
        # neurons the same stream.
        assert all(1 <= seed <= LFSR16_SEED_DOMAIN for seed in derive_population_seeds(1, 4096))

    def test_it_is_reproducible(self) -> None:
        """The same base seed and count give the same list."""
        assert derive_population_seeds(7, 32) == derive_population_seeds(7, 32)

    def test_a_prefix_of_a_larger_population_is_the_smaller_one(self) -> None:
        """The first neurons' seeds do not depend on the population size."""
        # Growing a population must not renumber the neurons already in it.
        assert derive_population_seeds(7, 8) == derive_population_seeds(7, 32)[:8]

    def test_it_fills_the_whole_domain_without_repeating(self) -> None:
        """A population as large as the domain takes every seed once."""
        seeds = derive_population_seeds(11, LFSR16_SEED_DOMAIN)

        assert len(set(seeds)) == LFSR16_SEED_DOMAIN

    def test_it_refuses_a_population_larger_than_the_domain(self) -> None:
        """One neuron more than the domain holds is refused."""
        # Refusing beats handing two neurons the same stream.
        with pytest.raises(ValueError, match="distinct seeds"):
            derive_population_seeds(1, LFSR16_SEED_DOMAIN + 1)

    def test_it_refuses_a_negative_count(self) -> None:
        """A negative count is refused."""
        with pytest.raises(ValueError, match="negative"):
            derive_population_seeds(1, -1)

    def test_no_neurons_needs_no_seeds(self) -> None:
        """Zero neurons give an empty list."""
        assert derive_population_seeds(1, 0) == []


class TestTheSeedDefaultLookup:
    """Which base seed a population derives from when none is named."""

    def test_a_factory_with_no_seed_parameter_names_no_default(self) -> None:
        """A factory without a ``seed`` parameter names no default."""
        assert _seed_default(lambda tau: tau) is None

    def test_a_factory_whose_seed_has_no_default_names_none(self) -> None:
        """A required ``seed`` parameter names no default."""
        assert _seed_default(lambda seed: seed) is None

    def test_a_seed_defaulting_to_entropy_is_left_alone(self) -> None:
        """A ``None`` default asks for entropy and is not derived over."""
        # `None` is the model's way of asking for independent entropy, and a
        # population must not turn that into a reproducible sequence.
        assert _seed_default(lambda seed=None: seed) is None

    def test_a_boolean_default_is_not_a_seed(self) -> None:
        """A boolean default is not taken as a seed."""
        assert _seed_default(lambda seed=True: seed) is None

    def test_an_integer_default_is_the_base_a_population_derives_from(self) -> None:
        """An integer default is the base seed."""
        assert _seed_default(lambda seed=44257: seed) == 44257

    def test_a_builtin_without_a_signature_is_given_no_seed(self) -> None:
        """A builtin without an inspectable signature names no default."""
        # `inspect` raises ValueError for a builtin type; a factory that cannot
        # be asked about a seed does not receive one.
        assert _seed_default(int) is None

    def test_something_that_is_not_callable_is_given_no_seed(self) -> None:
        """A non-callable factory names no default."""
        # TypeError rather than ValueError, and the same answer.
        assert _seed_default("not a factory") is None


class TestDistinctGeneratorStates:
    """RIGOR review 274: LFSR16 seeds 0 and 0xACE1 start one state; a population held both.

    ``Population(PoissonNeuron, 65536, seed=11)`` gave index 47550 seed 0 and index
    23147 seed 44257: two neurons with one stream, inside the first 47551 neurons.
    """

    def test_both_lfsr_models_declare_the_fallback_alias(self) -> None:
        """Seed 0 runs as the hardware fallback, and both LFSR16 models say so."""
        assert dict(LFSR16_ALIASES) == {0: 0xACE1}
        assert seed_aliases(PoissonNeuron) == seed_aliases(EscapeRateNeuron) == {0: 44257}
        assert PoissonNeuron(seed=0).initial_seed == PoissonNeuron(seed=44257).initial_seed

    def test_an_aliased_pair_is_never_handed_out(self) -> None:
        """Without the aliases base 11 hands out 0 and 44257; with them every state is distinct."""
        unaware = derive_population_seeds(11, 47551, LFSR16_DOMAIN)
        assert unaware[47550] == 0 and unaware[23147] == 44257
        seeds = derive_population_seeds(11, 47551, LFSR16_DOMAIN, LFSR16_ALIASES)
        starts = [LFSR16_ALIASES.get(seed, seed) for seed in seeds]
        assert len(set(starts)) == 47551
        # Replay is kept: every seed before the first alias is the seed it was.
        assert seeds[:47550] == unaware[:47550]

    def test_a_population_starts_every_neuron_from_its_own_state(self) -> None:
        """The real population of 47551 Poisson neurons: no two start from one state."""
        population = Population(PoissonNeuron, 47551, params={"seed": 11})
        assert len({neuron.initial_seed for neuron in population.neurons}) == 47551

    def test_the_capacity_is_the_number_of_distinct_states(self) -> None:
        """65535 distinct LFSR16 states fill; a 65536th neuron is refused."""
        seeds = derive_population_seeds(11, 65535, LFSR16_DOMAIN, LFSR16_ALIASES)
        assert len({LFSR16_ALIASES.get(seed, seed) for seed in seeds}) == 65535
        with pytest.raises(ValueError, match="65535 distinct"):
            derive_population_seeds(11, 65536, LFSR16_DOMAIN, LFSR16_ALIASES)
        with pytest.raises(ValueError, match="65535 distinct"):
            Population(PoissonNeuron, 65536, params={"seed": 11})

    def test_a_model_without_aliases_keeps_its_full_domain(self) -> None:
        """No declared alias: the capacity is the domain's size and the seeds are unchanged."""
        assert seed_aliases(None) == {}
        assert derive_population_seeds(5, 64, (0, 63), {}) == derive_population_seeds(
            5, 64, (0, 63)
        )
        assert sorted(derive_population_seeds(5, 64, (0, 63))) == list(range(64))
