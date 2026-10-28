# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — native-dispatch probe edges for models outside the catalogue

"""The dispatch probe refuses a model it cannot step, and one that draws process randomness.

No catalogue model reaches these branches. Each catalogue model has a
descriptor, builds with no arguments and steps on a float drive. A user model
put in the registry under a catalogue name can fail in each of these ways, so
the tests register one, as ``test_network_rust_dispatch_fidelity`` does. The
probe result is cached per model name. The fixture clears that cache before
and after each test, so no stub result reaches another test.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from sc_neurocore.network import Population
from sc_neurocore.network import rust_dispatch
from sc_neurocore.network.rust_dispatch import (
    draws_process_random,
    evolving_state,
    population_divergence,
)
from sc_neurocore.neurons import models as model_registry

MODEL = "SCLapicqueLIFNeuron"


@pytest.fixture(autouse=True)
def _fresh_probe_cache() -> Iterator[None]:
    """Clear the per-name probe cache around every test."""
    rust_dispatch._probe.cache_clear()
    yield
    rust_dispatch._probe.cache_clear()


def _registered(monkeypatch: pytest.MonkeyPatch, model: type[Any]) -> Population:
    """Return a one-neuron population of *model*, registered under ``MODEL``."""
    population = Population(MODEL, 1)
    monkeypatch.setattr(model_registry, MODEL, model, raising=False)
    population.neurons = [model()]
    return population


class NoStepNeuron:
    """A model that builds but has no callable ``step``."""

    step = None

    def __init__(self) -> None:
        self.v = 0.0


class FloatRefusingNeuron:
    """A model whose ``step`` refuses a float drive."""

    def __init__(self) -> None:
        self.v = 0.0

    def step(self, current: float) -> int:
        """Refuse any drive that is not an array."""
        raise TypeError(f"expected an array drive, got {type(current).__name__}")


class GlobalNoiseNeuron:
    """A model with no state besides ``v`` that draws from NumPy's global generator."""

    def __init__(self) -> None:
        self.v = 0.0

    def step(self, current: float) -> int:
        """Draw noise from the process-wide generator; never move ``v``."""
        np.random.random()
        return 0


def test_a_model_without_a_descriptor_has_no_known_state() -> None:
    """A name the catalogue does not describe has no evolving-state answer and no draws."""
    assert evolving_state("NoSuchCatalogueModel") is None
    assert draws_process_random("NoSuchCatalogueModel") is False


def test_a_model_without_a_callable_step_cannot_be_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered model whose ``step`` is not callable has no evolving-state answer."""
    _registered(monkeypatch, NoStepNeuron)
    assert evolving_state(MODEL) is None
    assert draws_process_random(MODEL) is False


def test_a_model_refusing_the_probe_drive_is_not_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model whose step refuses the float drive is reported as not steppable."""
    population = _registered(monkeypatch, FloatRefusingNeuron)
    assert evolving_state(MODEL) is None
    assert "cannot be stepped" in population_divergence(population)


def test_process_randomness_without_hidden_state_is_a_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that only draws global noise is refused, and the probe leaves the streams intact."""
    population = _registered(monkeypatch, GlobalNoiseNeuron)
    numpy_before = np.random.get_state()[1].copy()
    stdlib_before = random.getstate()
    assert evolving_state(MODEL) == ()
    assert draws_process_random(MODEL) is True
    assert "process-wide random generators" in population_divergence(population)
    assert np.array_equal(np.random.get_state()[1], numpy_before)
    assert random.getstate() == stdlib_before
