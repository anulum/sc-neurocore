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


class ReseedingResetNeuron:
    """A declared-pure, quiescent neuron whose ``reset`` reseeds both process generators."""

    quiescence_pure = True

    def __init__(self) -> None:
        self.v = 0.0

    def reset(self) -> None:
        """Return to rest and reseed the process-wide generators, as some models do."""
        self.v = 0.0
        random.seed(7)
        np.random.seed(7)

    def step(self, current: float) -> int:
        """Integrate the input; spike at 1."""
        self.v += current
        if self.v >= 1.0:
            self.v = 0.0
            return 1
        return 0


def test_the_whole_probe_leaves_the_callers_streams_unchanged() -> None:
    """RIGOR review 278 (NEURO-QUIES-RNG-01): the reset ran outside the restoration.

    Population.quiescent_signature copied and reset the first neuron before the
    probe restored anything, so a reset that reseeds moved both caller streams.
    """
    random.seed(42)
    np.random.seed(42)
    population = Population(ReseedingResetNeuron, 2)
    assert population.quiescent_signature() is not None
    assert random.random() == random.Random(42).random()
    assert np.random.random() == np.random.RandomState(42).random_sample()


def test_an_undeclared_model_is_not_even_reset_by_the_probe() -> None:
    """Purity is admitted first: an undeclared model is not copied or reset at all."""

    class UndeclaredReseeding(ReseedingResetNeuron):
        """The same neuron without the purity declaration."""

        quiescence_pure = False

    random.seed(42)
    population = Population(UndeclaredReseeding, 1)
    assert population.quiescent_signature() is None
    assert random.random() == random.Random(42).random()


class CyclicStateNeuron:
    """A declared-pure neuron holding a list that contains itself."""

    quiescence_pure = True

    def __init__(self) -> None:
        self.v = 0.0
        self.history: list[object] = []
        self.history.append(self.history)

    def step(self, current: float) -> int:
        """Integrate the input; spike at 1."""
        self.v += current
        if self.v >= 1.0:
            self.v = 0.0
            return 1
        return 0


def test_a_cyclic_state_is_incomparable_not_a_crash() -> None:
    """RIGOR review 278 (NEURO-QUIES-STATE-CYCLE-01): gated stepping raised RecursionError."""
    assert quiescent_signature(CyclicStateNeuron()) is None
    drive = [0.0, 0.5, 0.0, 0.6, 0.0]
    assert _trains(CyclicStateNeuron, True, drive, 3) == _trains(CyclicStateNeuron, False, drive, 3)
