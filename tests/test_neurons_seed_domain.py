# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - A model states which seeds it accepts

"""A seeded model declares its domain, and the contract enforces it.

A model's constructor already knew which seeds it took; nothing published it.
So a caller learned a domain by being refused — an out-of-range seed came back
as an invalid model input from the constructor, after the run had been
admitted, rather than as a request the experiment contract refused with the
domain named. And because the narrowest domain was the only value safe for all
models, every fresh seed and every per-neuron seed was drawn from sixteen bits,
including for the models that accept sixty-three.

Every case here fails on that former behaviour. The declarations are checked
against what each constructor actually does, so a declaration cannot drift from
the model it describes.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from sc_neurocore.network import Population
from sc_neurocore.network.population_seeds import derive_population_seeds
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.neurons.seed_domain import (
    LFSR16_DOMAIN,
    NONZERO_SEED_DOMAIN,
    NUMPY_SEED_DOMAIN,
    UNIVERSAL_SEED_DOMAIN,
    SeedOutOfDomain,
    check_seed,
    seed_domain,
)
from sc_neurocore.studio.experiment_spec import ExperimentRejected, resolve_experiment

# Every model in the catalogue that takes a seed, with the domain it declares.
# `GIFPopulationNeuron` is the one whose constructor validates nothing: its
# declaration is what refuses a negative seed, and it does so at the contract.
SEEDED_MODELS: dict[str, tuple[int, int]] = {
    "EscapeRateNeuron": LFSR16_DOMAIN,
    "PoissonNeuron": LFSR16_DOMAIN,
    "GIFPopulationNeuron": NUMPY_SEED_DOMAIN,
    "GLMNeuron": NUMPY_SEED_DOMAIN,
    "QuantumInspiredLIFNeuron": NONZERO_SEED_DOMAIN,
    "SCStochasticRateAdaptationNeuron": NUMPY_SEED_DOMAIN,
    "StochasticLIFNeuron": NUMPY_SEED_DOMAIN,
}

# The model whose constructor accepts a seed its declaration excludes.
UNVALIDATED_MODEL = "GIFPopulationNeuron"


def _model(class_name: str) -> Any:
    """Return a catalogue model class."""
    module = importlib.import_module(f"sc_neurocore.neurons.models.{_CLASS_TO_MODULE[class_name]}")
    return getattr(module, class_name)


class TestDeclarations:
    def test_every_seeded_model_declares_a_domain(self) -> None:
        """A new seeded model must state what it takes, not leave it to be discovered."""
        import inspect

        seeded = sorted(
            name
            for name in _CLASS_TO_MODULE
            if "seed" in inspect.signature(_model(name)).parameters
        )
        assert seeded == sorted(SEEDED_MODELS)

    @pytest.mark.parametrize("class_name", sorted(SEEDED_MODELS))
    def test_the_declaration_is_the_one_recorded(self, class_name: str) -> None:
        """The domain a model publishes is the one this suite was written against."""
        assert seed_domain(_model(class_name)) == SEEDED_MODELS[class_name]

    @pytest.mark.parametrize("class_name", sorted(SEEDED_MODELS))
    def test_the_constructor_accepts_both_ends_of_its_declaration(self, class_name: str) -> None:
        """A declaration wider than the constructor would refuse a valid request."""
        model = _model(class_name)
        low, high = seed_domain(model)
        model(seed=low)
        model(seed=high)

    @pytest.mark.parametrize(
        "class_name", sorted(name for name in SEEDED_MODELS if name != UNVALIDATED_MODEL)
    )
    def test_the_constructor_refuses_below_its_declaration(self, class_name: str) -> None:
        """A declaration narrower than the constructor would admit a bad seed."""
        model = _model(class_name)
        low, _ = seed_domain(model)
        with pytest.raises(ValueError):
            model(seed=low - 1)

    def test_the_one_model_that_validates_nothing_is_recorded_as_such(self) -> None:
        """Its declaration is the only thing standing between a caller and a bad seed."""
        model = _model(UNVALIDATED_MODEL)
        low, _ = seed_domain(model)
        model(seed=low - 1)


class TestSeedDomainAccessor:
    def test_a_model_that_declares_nothing_gets_the_narrowest_domain(self) -> None:
        """An unknown model is given the value that is safe everywhere."""

        class Undeclared:
            """A model with no declared seed domain."""

        assert seed_domain(Undeclared) == UNIVERSAL_SEED_DOMAIN

    def test_no_model_at_all_gets_the_narrowest_domain(self) -> None:
        """A caller with no class to ask still gets a usable answer."""
        assert seed_domain(None) == UNIVERSAL_SEED_DOMAIN

    @pytest.mark.parametrize(
        "declared", ["nonsense", (1,), (1, 2, 3), (1.0, 2.0), (True, 5), (5, True), (9, 1)]
    )
    def test_a_malformed_declaration_is_refused(self, declared: object) -> None:
        """A declaration that is not a pair of ordered integers is not one."""

        class Malformed:
            """A model whose declaration cannot be read."""

            SEED_DOMAIN = declared

        assert seed_domain(Malformed) == UNIVERSAL_SEED_DOMAIN


class TestCheckSeed:
    def test_a_seed_inside_the_domain_is_returned(self) -> None:
        """The check is a pass-through when the domain admits the value."""
        assert check_seed("Probe", 5, (0, 10)) == 5

    @pytest.mark.parametrize("seed", [-1, 11])
    def test_a_seed_outside_the_domain_is_refused_with_both_bounds(self, seed: int) -> None:
        """The refusal has to say what would have been accepted."""
        with pytest.raises(SeedOutOfDomain) as refusal:
            check_seed("Probe", seed, (0, 10))
        assert "[0, 10]" in str(refusal.value)
        assert refusal.value.domain == (0, 10)
        assert refusal.value.seed == seed

    @pytest.mark.parametrize("seed", [True, False])
    def test_a_boolean_is_not_a_seed(self, seed: bool) -> None:
        """`True` is not the seed 1, however it compares."""
        with pytest.raises(SeedOutOfDomain):
            check_seed("Probe", seed, (0, 10))


class TestDerivation:
    def test_derived_seeds_stay_inside_a_wide_domain(self) -> None:
        """A model with a wide domain no longer receives sixteen-bit seeds."""
        low, high = NUMPY_SEED_DOMAIN
        seeds = derive_population_seeds(11, 64, NUMPY_SEED_DOMAIN)
        assert len(set(seeds)) == 64
        assert all(low <= seed <= high for seed in seeds)
        assert max(seeds) > UNIVERSAL_SEED_DOMAIN[1]

    def test_derived_seeds_stay_inside_a_narrow_domain(self) -> None:
        """A sixteen-bit model still receives sixteen-bit seeds."""
        low, high = LFSR16_DOMAIN
        seeds = derive_population_seeds(11, 500, LFSR16_DOMAIN)
        assert len(set(seeds)) == 500
        assert all(low <= seed <= high for seed in seeds)

    def test_the_default_domain_is_the_narrowest(self) -> None:
        """A caller that names no domain gets what every population got before."""
        seeds = derive_population_seeds(7, 32)
        assert all(1 <= seed <= 65535 for seed in seeds)
        assert seeds == derive_population_seeds(7, 32, UNIVERSAL_SEED_DOMAIN)

    def test_a_derivation_is_reproducible_per_domain(self) -> None:
        """The same base, count and domain always give the same list."""
        assert derive_population_seeds(3, 16, NUMPY_SEED_DOMAIN) == derive_population_seeds(
            3, 16, NUMPY_SEED_DOMAIN
        )

    def test_a_wider_domain_gives_different_seeds(self) -> None:
        """The domain is used, not merely accepted."""
        assert derive_population_seeds(3, 16, NUMPY_SEED_DOMAIN) != derive_population_seeds(
            3, 16, LFSR16_DOMAIN
        )

    def test_a_population_larger_than_its_domain_is_refused(self) -> None:
        """Two neurons sharing a stream is the defect this exists to remove."""
        with pytest.raises(ValueError, match=r"\[0, 9\] holds 10"):
            derive_population_seeds(1, 11, (0, 9))

    def test_a_population_exactly_filling_its_domain_is_allowed(self) -> None:
        """The refusal is at the boundary, not before it."""
        seeds = derive_population_seeds(1, 10, (0, 9))
        assert sorted(seeds) == list(range(10))


class TestPopulationUsesTheDeclaration:
    def test_a_narrow_model_receives_narrow_seeds(self) -> None:
        """A sixteen-bit model's population stays inside sixteen bits."""
        population = Population("PoissonNeuron", 8)
        low, high = LFSR16_DOMAIN
        seeds = [neuron.seed for neuron in population.neurons]
        assert len(set(seeds)) == 8
        assert all(low <= seed <= high for seed in seeds)

    def test_a_wide_model_receives_wide_seeds(self) -> None:
        """The defect: a sixty-three-bit model was getting sixteen-bit seeds."""
        population = Population("QuantumInspiredLIFNeuron", 8)
        seeds = [neuron.seed for neuron in population.neurons]
        assert len(set(seeds)) == 8
        assert all(seed >= NONZERO_SEED_DOMAIN[0] for seed in seeds)
        assert max(seeds) > UNIVERSAL_SEED_DOMAIN[1]


class TestTheContractRefusesBeforeConstruction:
    @pytest.mark.parametrize(
        ("class_name", "seed"),
        [
            ("PoissonNeuron", 70000),
            ("EscapeRateNeuron", -1),
            ("QuantumInspiredLIFNeuron", 0),
            ("GIFPopulationNeuron", -1),
        ],
    )
    def test_an_out_of_domain_seed_is_refused_with_the_domain_named(
        self, class_name: str, seed: int
    ) -> None:
        """The refusal names the domain, so a caller does not learn it by guessing."""
        low, high = seed_domain(_model(class_name))
        with pytest.raises(ExperimentRejected) as refusal:
            resolve_experiment({"name": class_name, "duration": 2.0, "current": 1.0, "seed": seed})
        assert f"[{low}, {high}]" in str(refusal.value)

    @pytest.mark.parametrize(
        ("class_name", "seed"),
        [("PoissonNeuron", 65535), ("PoissonNeuron", 0), ("GIFPopulationNeuron", 5_000_000_000)],
    )
    def test_a_seed_inside_the_domain_is_accepted(self, class_name: str, seed: int) -> None:
        """A wide-domain model may now be given a wide seed."""
        spec = resolve_experiment(
            {"name": class_name, "duration": 2.0, "current": 1.0, "seed": seed}
        )
        assert spec.to_public_dict()["randomness"]["seed"] == seed

    def test_a_fresh_trial_draws_inside_the_model_domain(self) -> None:
        """A drawn seed comes from the model's domain, not from the narrowest one."""
        drawn = {
            resolve_experiment(
                {
                    "name": "QuantumInspiredLIFNeuron",
                    "duration": 2.0,
                    "current": 1.0,
                    "trial": "fresh",
                }
            ).to_public_dict()["randomness"]["seed"]
            for _ in range(16)
        }
        low, high = NONZERO_SEED_DOMAIN
        assert all(low <= seed <= high for seed in drawn)
        assert max(drawn) > UNIVERSAL_SEED_DOMAIN[1]
