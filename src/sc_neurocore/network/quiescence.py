# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — When skipping a neuron is exactly equivalent to stepping it

"""When a population may skip a neuron without changing what it computes.

Spike gating exists to make compute proportional to the active fraction of a
sparse network. It is only sound where skipping a neuron and stepping it are
the same thing, and that holds in exactly one situation: the neuron sits at a
state the zero-input map returns unchanged, and driving it with zero emits no
spike. Anywhere else, skipping freezes a leak, an adaptation current or a
refractory countdown that the model would have advanced, and the run diverges
from the same run without gating.

The fixed point is not assumed, it is measured. A copy of the model is stepped
with zero input and compared with itself: if any attribute moved, or a spike
came out, the model has no quiescent state and its populations are never
gated. A model that passes gives a signature, and only a neuron whose state
matches that signature exactly may be skipped — exactly, not within a
tolerance, because a neuron a little away from rest is precisely the one whose
relaxation the skip would discard.

Making the test exact costs skips. That is the honest price: the previous
heuristic bought its speed by returning different numbers.
"""

from __future__ import annotations

import copy
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Steppable(Protocol):
    """A neuron a population can drive one timestep with a scalar input."""

    def step(self, current: float) -> int:
        """Advance one timestep under *current* and return the spike flag."""


#: The drive a gated neuron would receive. A neuron is only ever a candidate
#: for skipping when its input is exactly this, so this is the only input the
#: fixed point has to hold for.
QUIESCENT_DRIVE = 0.0

StateSignature = tuple[tuple[str, str], ...]


def _attribute_names(instance: object) -> list[str]:
    """Return every attribute an instance carries, from its dict and its slots."""
    names: list[str] = list(vars(instance)) if hasattr(instance, "__dict__") else []
    for klass in type(instance).__mro__:
        slots = getattr(klass, "__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if (
                isinstance(slot, str)
                and slot not in names
                and slot not in ("__dict__", "__weakref__")
            ):
                names.append(slot)
    return names


#: The signature of a value this module cannot compare by content. A state
#: containing one is not measurable, so the model it belongs to is never
#: reported quiescent and its populations are never gated.
INCOMPARABLE = "incomparable"


def _value_signature(value: object) -> str:
    """Return a value's exact textual signature, or :data:`INCOMPARABLE`.

    A random generator is the case that matters. Its object identity is stable
    while its stream advances on every draw, so a signature built from identity
    would call a stochastic model quiescent and let gating skip it — and a
    skipped neuron does not advance the generator, so the run would diverge
    from the ungated one through the noise rather than through the dynamics.
    Anything not comparable by content is therefore refused outright.
    """
    if isinstance(value, bool):
        return f"bool:{value!r}"
    if isinstance(value, (int, float, np.integer, np.floating)):
        # A real number is its value, not the scalar type it arrived in: a model
        # whose voltage becomes a NumPy float after its first step is at the same
        # state as one that never left Python's, and must still be skippable.
        return f"number:{float(value)!r}"
    if isinstance(value, (str, bytes, type(None))):
        return f"{type(value).__name__}:{value!r}"
    if isinstance(value, np.ndarray):
        return f"ndarray:{value.shape}:{value.dtype}:{value.tobytes().hex()}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}:[{','.join(_value_signature(item) for item in value)}]"
    return INCOMPARABLE


def _global_generator_signature_of(state: object) -> tuple[object, ...]:
    """Return a comparable signature of one process-wide generator state."""
    if not isinstance(state, tuple):
        return (repr(state),)
    return tuple(item.tobytes() if isinstance(item, np.ndarray) else item for item in state)


def _global_generator_signature() -> tuple[object, ...]:
    """Return the state of the process-wide NumPy generator.

    A model that draws from it holds no record of the draw, so its own
    attributes can sit perfectly still while the run's randomness advances.
    """
    return _global_generator_signature_of(np.random.get_state())


def state_signature(neuron: object) -> StateSignature:
    """Return the exact signature of every attribute a neuron carries."""
    return tuple(
        (name, _value_signature(getattr(neuron, name))) for name in _attribute_names(neuron)
    )


def quiescent_signature(neuron: object) -> StateSignature | None:
    """Return the signature of a state the zero-input map leaves unchanged.

    Parameters
    ----------
    neuron : object
        A neuron in the state whose quiescence is in question, usually a freshly
        constructed or reset instance. It is not modified: the probe runs on a
        copy.

    Returns
    -------
    tuple or None
        The signature a neuron must match to be skippable, or ``None`` when this
        model does not hold still — because a copy stepped with zero input
        spiked, moved an attribute, refused the call, or could not be copied at
        all. ``None`` means this model's populations are never gated.

    Examples
    --------
    >>> from sc_neurocore.neurons.models.adex import AdExNeuron
    >>> quiescent_signature(AdExNeuron()) is not None
    True
    >>> resting = AdExNeuron()
    >>> resting.v += 5.0
    >>> quiescent_signature(resting) is None
    True
    """
    try:
        probe = copy.deepcopy(neuron)
    except Exception:
        # A model holding something that cannot be copied cannot be probed, and
        # an unprobed model is never skipped.
        return None
    if not isinstance(probe, Steppable):
        # Not something a population can drive; nothing to hold still.
        return None
    before = state_signature(probe)
    if any(signature == INCOMPARABLE for _, signature in before):
        # Part of this model's state cannot be compared by content, so whether
        # it holds still cannot be established. An unestablished fixed point is
        # not one.
        return None
    # The probe must leave the process exactly as it found it. It already runs
    # on a copy so the neuron is untouched; the process-wide generator is the
    # other thing a step can move, and a probe that consumed a draw would shift
    # every later draw in the run it was measuring.
    entry_state = np.random.get_state()
    entry_stream = _global_generator_signature_of(entry_state)
    try:
        emitted = probe.step(QUIESCENT_DRIVE)
        exit_stream = _global_generator_signature()
    except Exception:
        # A model that refuses a zero drive is not quiescent under one.
        return None
    finally:
        np.random.set_state(entry_state)
    if before != state_signature(probe):
        return None
    if exit_stream != entry_stream:
        # The model drew from the process-wide generator, whose state is not
        # part of any instance. Skipping such a neuron would leave that draw
        # unmade and move every later draw in the process, so it is never
        # quiescent however still its own attributes look.
        return None
    if emitted != 0:
        return None
    return before


def is_quiescent(neuron: object, signature: StateSignature | None) -> bool:
    """Return whether a neuron sits exactly at the measured quiescent state."""
    return signature is not None and state_signature(neuron) == signature


__all__ = [
    "INCOMPARABLE",
    "QUIESCENT_DRIVE",
    "StateSignature",
    "Steppable",
    "is_quiescent",
    "quiescent_signature",
    "state_signature",
]
