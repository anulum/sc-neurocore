# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — quiescence is probed only for declared-pure models

"""Spike gating skips only models whose purity is declared, and reads private slots.

Reproducers of RIGOR review 273: a user neuron drawing from the standard
``random`` module sampled one zero spike, was reported quiescent, and gating
then skipped it, changing the spike train and the caller's random stream; a
user neuron with a private slot ``__ticks`` crashed gated stepping with an
AttributeError. Gating must leave every run equal to the ungated one.
"""

from __future__ import annotations

import random

import numpy as np

from sc_neurocore.network.population import Population
from sc_neurocore.network.quiescence import declares_purity, quiescent_signature


class StdlibRandomNeuron:
    """A user neuron that spikes with probability 0.3 from the standard random module."""

    def __init__(self) -> None:
        self.v = 0.0

    def step(self, current: float) -> int:
        """Draw from the process-wide stdlib generator."""
        return int(random.random() < 0.3)


class DeclaredStdlibRandomNeuron(StdlibRandomNeuron):
    """The same neuron with a false purity declaration: the probe still sees the draw."""

    quiescence_pure = True


class PrivateSlots:
    """A user neuron with a private slot, which Python stores as _PrivateSlots__ticks."""

    __slots__ = ("v", "__ticks")
    quiescence_pure = True

    def __init__(self) -> None:
        self.v = 0.0
        self.__ticks = 0

    def step(self, current: float) -> int:
        """Spike on every second driven step; hold still without input."""
        if current == 0.0:
            return 0
        self.__ticks += 1
        return int(self.__ticks % 2 == 0)


def _trains(model: type, gated: bool, drive: list[float], seed: int) -> tuple[list[int], float]:
    """Step a one-neuron population; return its spikes and the next stdlib draw."""
    random.seed(seed)
    population = Population(model, 1)
    spikes = [
        int(np.asarray(population.step_all(np.array([c]), spike_gating=gated))[0]) for c in drive
    ]
    return spikes, random.random()


def test_an_undeclared_user_model_is_never_probed() -> None:
    """Without a purity declaration the probe refuses, and the stdlib stream is untouched."""
    random.seed(42)
    assert not declares_purity(StdlibRandomNeuron())
    assert quiescent_signature(StdlibRandomNeuron()) is None
    assert random.random() == random.Random(42).random()


def test_a_declared_model_that_draws_from_stdlib_random_is_refused_and_restored() -> None:
    """The probe restores the stdlib generator and refuses the model that moved it."""
    random.seed(42)
    assert quiescent_signature(DeclaredStdlibRandomNeuron()) is None
    assert random.random() == random.Random(42).random()


def test_gating_leaves_a_stdlib_random_run_unchanged() -> None:
    """Gated and ungated runs give the same spikes and leave the same stream."""
    for model in (StdlibRandomNeuron, DeclaredStdlibRandomNeuron):
        assert _trains(model, True, [0.0] * 6, 42) == _trains(model, False, [0.0] * 6, 42)


def test_private_slots_are_read_and_gated_like_public_ones() -> None:
    """A private slot is part of the signature; gated stepping matches ungated."""
    signature = quiescent_signature(PrivateSlots())
    assert signature is not None and "_PrivateSlots__ticks" in dict(signature)
    drive = [0.0, 1.0, 1.0, 0.0, 1.0]
    assert _trains(PrivateSlots, True, drive, 1) == _trains(PrivateSlots, False, drive, 1)


def test_catalogue_models_keep_their_declared_purity() -> None:
    """The catalogue models of sc_neurocore are declared pure and still probed."""
    from sc_neurocore.neurons.models.lapicque import LapicqueNeuron

    assert declares_purity(LapicqueNeuron()) and quiescent_signature(LapicqueNeuron()) is not None
