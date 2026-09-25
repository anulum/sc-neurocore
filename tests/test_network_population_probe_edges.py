# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Populations built from models the probes cannot inspect

"""A population whose model cannot be copied, reset or read is never skipped.

The catalogue's models all copy, reset and expose their state, so the models
here are small user-defined classes, as a user may pass to ``Population``. Each
one fails in exactly one way, and the population must then decline to call it
quiescent rather than guess.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.network.population import Population
from sc_neurocore.network.quiescence import _global_generator_signature_of
from sc_neurocore.network.rust_dispatch import _attribute_map


class UncopyableNeuron:
    """A user model holding a resource that cannot be deep-copied."""

    def __init__(self) -> None:
        self.v = 0.0

    def __deepcopy__(self, memo: dict[int, object]) -> UncopyableNeuron:
        raise TypeError("this neuron holds a handle that cannot be copied")

    def step(self, current: float) -> bool:
        """Integrate one step."""
        self.v += current
        return False


class UnresettableNeuron:
    """A user model whose reset refuses."""

    def __init__(self) -> None:
        self.v = 0.0

    def reset(self) -> None:
        """Refuse, as a model with no defined rest state might."""
        raise RuntimeError("no rest state is defined for this model")

    def step(self, current: float) -> bool:
        """Integrate one step."""
        self.v += current
        return False


class SlottedNeuron:
    """A slotted user model that also allows weak references."""

    __slots__ = ("v", "__weakref__")

    def __init__(self) -> None:
        self.v = 0.25


@pytest.mark.parametrize("model", [UncopyableNeuron, UnresettableNeuron])
def test_a_model_the_probe_cannot_prepare_has_no_quiescent_state(model: type) -> None:
    population = Population(model, 3)
    assert population.quiescent_signature() is None
    # Measured once: the second call answers from the record, not a new probe.
    assert population.quiescent_signature() is None


def test_the_source_lapicque_profile_refuses_parameters_it_does_not_have() -> None:
    with pytest.raises(TypeError, match="source profile received unsupported parameters: tau_x"):
        Population("LapicqueNeuron", 2, params={"v_threshold": 1.0, "tau_x": 3.0})


def test_slots_reserved_for_the_runtime_are_not_model_state() -> None:
    assert _attribute_map(SlottedNeuron()) == {"v": 0.25}


def test_a_generator_state_that_is_not_the_legacy_tuple_is_compared_by_its_text() -> None:
    state = np.random.default_rng(3).bit_generator.state
    assert _global_generator_signature_of(state) == (repr(state),)
