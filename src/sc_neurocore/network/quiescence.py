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

Scalar numbers are compared as exact rational values, without rounding integer
registers through binary64. Opaque values anywhere inside a sequence or an
object array disable skipping; pointer bytes are not a state comparison.

A probe can only see the state it knows how to read. A model may also draw
from generators or other state outside its instance (the standard ``random``
module, a file, a clock), and one zero spike sampled from such a model proves
nothing. Only models whose purity is declared are probed: the catalogue
models of :mod:`sc_neurocore`, and a user class that sets
``quiescence_pure = True`` to state that its ``step`` reads and writes nothing
but its own attributes. Every other model is never skipped. The probe still
restores, and refuses a model that moves, both the NumPy and the standard
``random`` process-wide generators.
"""

from __future__ import annotations

import copy
import random
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
    """Return every attribute an instance carries, from its dict and its slots.

    A private slot ``__x`` declared by class ``C`` is stored as ``_C__x`` (Python
    name mangling, with the class name's leading underscores removed).
    """
    names: list[str] = list(vars(instance)) if hasattr(instance, "__dict__") else []
    for klass in type(instance).__mro__:
        slots = getattr(klass, "__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if not isinstance(slot, str) or slot in ("__dict__", "__weakref__"):
                continue
            if slot.startswith("__") and not slot.endswith("__"):
                slot = f"_{klass.__name__.lstrip('_')}{slot}"
            if slot not in names:
                names.append(slot)
    return names


def declares_purity(neuron: object) -> bool:
    """Return whether a model's ``step`` is declared to touch only its own attributes.

    The catalogue models of :mod:`sc_neurocore` are; a user class declares it with a
    class attribute ``quiescence_pure = True``.
    """
    model = type(neuron)
    return model.__module__.startswith("sc_neurocore.") or (
        getattr(model, "quiescence_pure", False) is True
    )


#: The signature of a value this module cannot compare by content. A state
#: containing one is not measurable, so the model it belongs to is never
#: reported quiescent and its populations are never gated.
INCOMPARABLE = "incomparable"


def _value_signature(value: object, _open: frozenset[int] = frozenset()) -> str:
    """Return a value's exact textual signature, or :data:`INCOMPARABLE`.

    A sequence that contains itself (directly or through another) has no
    finite signature and is :data:`INCOMPARABLE`; ``_open`` holds the
    sequences being read.

    A random generator is the case that matters. Its object identity is stable
    while its stream advances on every draw, so a signature built from identity
    would call a stochastic model quiescent and let gating skip it — and a
    skipped neuron does not advance the generator, so the run would diverge
    from the ungated one through the noise rather than through the dynamics.
    Anything not comparable by content is therefore refused outright.
    """
    if isinstance(value, bool):
        return f"bool:{value!r}"
    if isinstance(value, (int, np.integer)):
        return f"number:{int(value)}/1"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return INCOMPARABLE
        numerator, denominator = value.as_integer_ratio()
        return f"number:{numerator}/{denominator}"
    if isinstance(value, (str, bytes, type(None))):
        return f"{type(value).__name__}:{value!r}"
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return INCOMPARABLE
        return f"ndarray:{value.shape}:{value.dtype}:{value.tobytes().hex()}"
    if isinstance(value, (list, tuple)):
        if id(value) in _open:
            return INCOMPARABLE
        inner = _open | {id(value)}
        items = [_value_signature(item, inner) for item in value]
        if INCOMPARABLE in items:
            return INCOMPARABLE
        return f"{type(value).__name__}:[{','.join(items)}]"
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
    """Return the exact signature of every attribute a neuron carries.

    A value nested too deeply to read is :data:`INCOMPARABLE`.
    """
    signature = []
    for name in _attribute_names(neuron):
        try:
            text = _value_signature(getattr(neuron, name))
        except RecursionError:
            text = INCOMPARABLE
        signature.append((name, text))
    return tuple(signature)


class GeneratorTransaction:
    """Restore the NumPy and standard ``random`` process-wide generators on exit.

    Everything a probe does inside — copying a neuron, resetting it, stepping
    it — may draw from or reseed them; the caller's streams are left exactly as
    they were, on success, on refusal and on error.
    """

    def __enter__(self) -> GeneratorTransaction:
        """Record both generator states."""
        self._numpy = np.random.get_state()
        self._stdlib = random.getstate()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Restore both generator states; any exception propagates."""
        np.random.set_state(self._numpy)
        random.setstate(self._stdlib)


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
    >>> from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
    >>> quiescent_signature(LapicqueNeuron()) is not None
    True
    >>> resting = LapicqueNeuron()
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
    if not declares_purity(probe):
        # State outside the instance cannot be read, so its stillness cannot be shown.
        return None
    try:
        before = state_signature(probe)
    except AttributeError:
        # An unset or unreadable attribute: the state cannot be compared.
        return None
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
    entry_stdlib = random.getstate()
    try:
        emitted = probe.step(QUIESCENT_DRIVE)
        exit_stream = _global_generator_signature()
        exit_stdlib = random.getstate()
        after = state_signature(probe)
    except Exception:
        # A model that refuses a zero drive is not quiescent under one.
        return None
    finally:
        np.random.set_state(entry_state)
        random.setstate(entry_stdlib)
    if before != after:
        return None
    if exit_stream != entry_stream or exit_stdlib != entry_stdlib:
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
    if signature is None:
        return False
    try:
        return state_signature(neuron) == signature
    except AttributeError:
        return False


__all__ = [
    "INCOMPARABLE",
    "QUIESCENT_DRIVE",
    "GeneratorTransaction",
    "StateSignature",
    "Steppable",
    "declares_purity",
    "is_quiescent",
    "quiescent_signature",
    "state_signature",
]
