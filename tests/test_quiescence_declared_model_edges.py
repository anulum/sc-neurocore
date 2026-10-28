# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — quiescence probe edges reached by models declared pure

"""A declared-pure model whose state cannot be read or compared is never quiescent.

The purity gate stops an undeclared model before anything is read, so every
model here declares ``quiescence_pure = True``. That way the probe reaches the
part of the state signature each test is about.
"""

from __future__ import annotations

import numpy as np

from sc_neurocore.network.quiescence import (
    INCOMPARABLE,
    declares_purity,
    is_quiescent,
    quiescent_signature,
    state_signature,
)


class DictAndWeakrefSlots:
    """A slotted model that also reserves ``__dict__`` and ``__weakref__``."""

    __slots__ = ("v", "__dict__", "__weakref__")
    quiescence_pure = True

    def __init__(self) -> None:
        self.v = 0.5
        self.extra = 1

    def step(self, current: float) -> int:
        """Integrate the input current; never spike."""
        self.v += current
        return 0


class UnsetSlot:
    """A slotted model whose second slot is never assigned."""

    __slots__ = ("v", "gain")
    quiescence_pure = True

    def __init__(self) -> None:
        self.v = 0.0

    def step(self, current: float) -> int:
        """Integrate the input current; never spike."""
        self.v += current
        return 0


class ObjectArrayState:
    """A model holding an object array, which has no content signature."""

    quiescence_pure = True

    def __init__(self) -> None:
        self.v = 0.0
        self.handles = np.array([object(), object()], dtype=object)

    def step(self, current: float) -> int:
        """Integrate the input current; never spike."""
        self.v += current
        return 0


class DeeplyNestedState:
    """A model holding a list nested beyond the interpreter's recursion limit."""

    quiescence_pure = True

    def __init__(self, depth: int) -> None:
        self.v = 0.0
        nested: list[object] = []
        for _ in range(depth):
            nested = [nested]
        self.history = nested

    def step(self, current: float) -> int:
        """Integrate the input current; never spike."""
        self.v += current
        return 0


def test_reserved_slots_are_not_state() -> None:
    """``__dict__`` and ``__weakref__`` slots are skipped; the slot and dict attributes are read."""
    neuron = DictAndWeakrefSlots()
    names = [name for name, _ in state_signature(neuron)]
    assert sorted(names) == ["extra", "v"]
    assert quiescent_signature(neuron) is not None


def test_an_unset_slot_makes_the_state_unreadable() -> None:
    """A declared-pure model with an unassigned slot has no quiescent signature."""
    neuron = UnsetSlot()
    assert declares_purity(neuron)
    assert quiescent_signature(neuron) is None


def test_an_unset_slot_is_never_at_a_recorded_signature() -> None:
    """``is_quiescent`` answers False instead of raising when the state cannot be read."""
    recorded = (("v", "number:0/1"), ("gain", "number:0/1"))
    assert is_quiescent(UnsetSlot(), recorded) is False


def test_an_object_array_is_incomparable() -> None:
    """An object array signs as INCOMPARABLE, so the model is never quiescent."""
    neuron = ObjectArrayState()
    assert dict(state_signature(neuron))["handles"] == INCOMPARABLE
    assert quiescent_signature(neuron) is None


def test_state_nested_past_the_recursion_limit_is_incomparable() -> None:
    """A value too deep to read signs as INCOMPARABLE instead of raising RecursionError."""
    neuron = DeeplyNestedState(depth=200_000)
    assert dict(state_signature(neuron))["history"] == INCOMPARABLE
    assert quiescent_signature(neuron) is None
