# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - A generator that advances must be visible to the mutation audit

"""A run's mutation audit must see a random generator advance.

The audit fingerprints every instance attribute before and after a run and
reports the ones that changed without being declared. It fingerprinted an
arbitrary object with ``repr``, and ``repr`` of a NumPy ``Generator`` is the
address-based default: constant across every draw the generator makes. Three
catalogue models advanced their generator on every step and the audit reported
nothing, so their runs were published with complete custody while carrying a
moving, undeclared register that determines the whole stochastic trajectory.

Every case here fails on that former behaviour. The audit now fingerprints a
generator by its generator state, and an object whose ``repr`` is the
address-based default is marked opaque rather than passed off as a value — so
the remaining blind spot is enumerable, and a catalogue-wide case asserts it is
currently empty.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.state_layout import (
    RNG_STATE_BACKING,
    RNG_STATE_VARIABLE,
    attribute_fingerprints,
    declared_state,
    observe_layout,
    undeclared_mutations,
)

# Every catalogue model that holds a random generator, and whether it publishes
# that generator's state through the declared `rng_state` variable. The five
# that do not are DISCOVERED-PRIVATE-REGISTERS-OUTSIDE-LAYOUT and
# DISCOVERED-DESCRIPTORS-WITHOUT-STATE; this pin keeps the set from growing
# while those rows are open.
GENERATOR_MODELS: dict[str, bool] = {
    "EscapeRateNeuron": True,
    "PoissonNeuron": True,
    "GIFPopulationNeuron": False,
    "GLMNeuron": False,
    "GammaRenewalNeuron": False,
    "QuantumInspiredLIFNeuron": False,
    "SCStochasticRateAdaptationNeuron": False,
    "StochasticLIFNeuron": False,
}

RUN_STEPS = 120
DRIVE = 1.0

# A model that only draws under a configuration. StochasticLIFNeuron's noise
# term is gated on `noise_std`, which defaults to zero, so a default instance is
# deterministic and its generator correctly never advances. The audit is asked
# about the configuration where it does.
DRAWING_PARAMETERS: dict[str, dict[str, float]] = {
    "StochasticLIFNeuron": {"noise_std": 0.5},
}


class _HoldsGenerator:
    """A minimal instance carrying one attribute, for fingerprinting."""

    def __init__(self, value: object) -> None:
        self.register = value


class _Opaque:
    """An object with the address-based default ``repr``."""


class _StateBearing:
    """A generator-shaped object publishing an integer state."""

    def __init__(self) -> None:
        self.state = 1

    def draw(self) -> int:
        """Advance the state and return it."""
        self.state = (self.state * 5 + 3) % 65536
        return self.state


def _instance(class_name: str, **parameters: float) -> Any:
    """Return a catalogue model, default-constructed unless parameters are given."""
    module = importlib.import_module(f"sc_neurocore.neurons.models.{_CLASS_TO_MODULE[class_name]}")
    model: Any = getattr(module, class_name)(**parameters)
    return model


def _fingerprint_of(value: object) -> str:
    """Return the audit's fingerprint of one value."""
    return attribute_fingerprints(_HoldsGenerator(value))["register"]


def _run_and_audit(class_name: str) -> tuple[Any, tuple[str, ...]]:
    """Run a model and return it with the attributes the audit reports."""
    neuron = _instance(class_name, **DRAWING_PARAMETERS.get(class_name, {}))
    source, stem, declared = declared_state(class_name)
    layout = observe_layout(
        neuron, source, stem, declared, n_steps=RUN_STEPS, element_budget=1 << 20
    )
    before = attribute_fingerprints(neuron)
    for _ in range(RUN_STEPS):
        neuron.step(DRIVE)
    return neuron, undeclared_mutations(before, attribute_fingerprints(neuron), layout)


class TestFingerprintSeesAGenerator:
    def test_a_numpy_generator_fingerprint_changes_with_a_draw(self) -> None:
        """The defect: a drawn-from generator fingerprinted identically before."""
        generator = np.random.default_rng(7)
        before = _fingerprint_of(generator)
        generator.random()
        assert _fingerprint_of(generator) != before

    def test_a_wrapped_generator_is_seen_through(self) -> None:
        """A model may hold its generator inside a small wrapper; that hides nothing."""

        class Wrapper:
            """A thin adapter over a NumPy generator."""

            def __init__(self) -> None:
                self._rng = np.random.default_rng(11)

        wrapper = Wrapper()
        before = _fingerprint_of(wrapper)
        wrapper._rng.random()
        assert _fingerprint_of(wrapper) != before

    def test_a_state_bearing_generator_fingerprints_by_its_state(self) -> None:
        """The library's own generators publish an integer state; that is the value."""
        generator = _StateBearing()
        before = _fingerprint_of(generator)
        assert before == "state:1"
        generator.draw()
        assert _fingerprint_of(generator) != before

    def test_an_object_with_the_default_repr_is_marked_opaque(self) -> None:
        """A blind spot is recorded as one rather than passed off as a value."""
        assert _fingerprint_of(_Opaque()).startswith("opaque:")

    def test_an_object_with_a_real_repr_still_fingerprints_by_it(self) -> None:
        """A value that describes itself is unchanged by this."""

        class Described:
            """An object whose repr carries its value."""

            def __init__(self, value: int) -> None:
                self.value = value

            def __repr__(self) -> str:
                return f"Described({self.value})"

        assert _fingerprint_of(Described(1)) != _fingerprint_of(Described(2))

    @pytest.mark.parametrize(
        "value", [1.0, True, "text", b"bytes", [1.0, 2.0], (1,), {"a": 1}, {1, 2}]
    )
    def test_ordinary_values_are_unaffected(self, value: object) -> None:
        """Numbers, text and containers keep fingerprinting by their contents."""
        assert not _fingerprint_of(value).startswith(("opaque:", "state:", "bitgenerator:"))

    def test_an_array_still_fingerprints_by_its_contents(self) -> None:
        """The array path is untouched."""
        first = _fingerprint_of(np.zeros(3))
        assert first != _fingerprint_of(np.ones(3))
        assert first.startswith("ndarray")


class TestCatalogueGenerators:
    def test_the_models_holding_a_generator_are_the_pinned_set(self) -> None:
        """A new stochastic model must be considered, not silently added."""
        holders = sorted(
            name
            for name in _CLASS_TO_MODULE
            if any(key in vars(_instance(name)) for key in RNG_STATE_BACKING)
        )
        assert holders == sorted(GENERATOR_MODELS)

    @pytest.mark.parametrize(
        "class_name", sorted(name for name, declared in GENERATOR_MODELS.items() if declared)
    )
    def test_a_declared_generator_is_accounted_for(self, class_name: str) -> None:
        """The private generator behind a declared `rng_state` is not undeclared state."""
        _, _, declared = declared_state(class_name)
        assert RNG_STATE_VARIABLE in {variable.name for variable in declared}
        _, reported = _run_and_audit(class_name)
        assert not set(reported) & set(RNG_STATE_BACKING)

    @pytest.mark.parametrize(
        "class_name", sorted(name for name, declared in GENERATOR_MODELS.items() if not declared)
    )
    def test_an_undeclared_generator_is_reported(self, class_name: str) -> None:
        """A model whose generator nothing declares must say so, not claim custody."""
        _, reported = _run_and_audit(class_name)
        assert set(reported) & set(RNG_STATE_BACKING)

    def test_a_generator_that_is_never_drawn_from_is_not_reported(self) -> None:
        """Holding a generator is not moving one; a deterministic run says so."""
        neuron = _instance("StochasticLIFNeuron")
        assert neuron.noise_std == 0.0
        source, stem, declared = declared_state("StochasticLIFNeuron")
        layout = observe_layout(
            neuron, source, stem, declared, n_steps=RUN_STEPS, element_budget=1 << 20
        )
        before = attribute_fingerprints(neuron)
        for _ in range(RUN_STEPS):
            neuron.step(DRIVE)
        reported = undeclared_mutations(before, attribute_fingerprints(neuron), layout)
        assert not set(reported) & set(RNG_STATE_BACKING)

    def test_no_attribute_anywhere_is_opaque_to_the_audit(self) -> None:
        """The blind spot the opaque marker names is currently empty; keep it so."""
        blind: list[str] = []
        for name in sorted(_CLASS_TO_MODULE):
            try:
                instance = _instance(name)
            except Exception:  # noqa: BLE001 - a model that cannot be built holds nothing
                continue
            blind.extend(
                f"{name}.{attribute}"
                for attribute, fingerprint in attribute_fingerprints(instance).items()
                if fingerprint.startswith("opaque:")
            )
        assert blind == []
