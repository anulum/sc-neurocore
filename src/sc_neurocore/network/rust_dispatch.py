# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Which populations the native network bridge can build faithfully

"""Whether the native network bridge can construct a population as assembled.

The Rust network runner is handed ``(model_name, n)`` and nothing else, so it
builds **default** neurons. A population constructed with parameters, one whose
neurons carry independently derived seeds, or one whose neurons have moved
since construction is therefore a different network from the one the caller
assembled. Dispatching it to the bridge returns results for the defaults while
every Python object goes on reporting the caller's parameters — the run is
wrong and nothing says so.

This module answers the one question that makes the dispatch safe: is every
neuron of this population identical to what the bridge will construct from the
model name alone? The reference is built by the same route the bridge takes —
resolving the model by name and calling it with no arguments — so the answer
describes the bridge rather than a guess about it.

The comparison is deliberately conservative. Anything it cannot decide counts
as a difference, because the cost of a wrong "yes" is a silently incorrect
simulation while the cost of a wrong "no" is a slower, correct one.

Two further limits of the boundary are checked here. The runner returns only
the membrane voltages, so a model that evolves any other state (AdEx's ``w``,
Rulkov's ``x`` and ``y``, a refractory counter, a generator) would leave that
state at its initial value in every Python neuron after a native run: such a
model is found by stepping a default neuron and seeing what besides ``v``
moves. And one name builds a different model on each side: ``LapicqueNeuron``
is the SC LIF compatibility profile in Python but the 1907 source profile in
the engine. Neither is dispatched.

:func:`projection_divergences` names what the runner would drop from a
projection: it receives the CSR arrays and one uniform delay, so per-synapse
delays, weight-threshold pruning and plasticity have no route across.

One residual is stated rather than hidden: the check reads the neurons, so it
sees a parameter changed in place after construction. It cannot see a change
made *after* the check and before the bridge builds its own neurons, and it
says nothing about whether the Rust defaults equal the Python defaults — that
is a parity question, held by the parity suites.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from functools import cache
from typing import Any, Final

import numpy as np

from sc_neurocore.network.population import Population
from sc_neurocore.network.quiescence import GeneratorTransaction
from sc_neurocore.neurons import models as _model_registry
from sc_neurocore.neurons.model_catalogue import load_descriptor

#: Everything the native bridge receives about a population. The rest of what a
#: population knows — its constructor parameters and its evolved state — has no
#: route across the boundary.
BRIDGE_CONSTRUCTOR_ARGUMENTS: Final = ("model_name", "n")

#: How many differing attribute names a reason names before it summarises.
_NAMED_ATTRIBUTE_LIMIT: Final = 4

#: Model names the engine resolves to a different model than Python does.
NATIVE_IDENTITY_DIFFERS: Final = {
    "LapicqueNeuron": (
        "the engine builds the 1907 source profile under this name while Python's "
        "default is the SC LIF compatibility profile; use SCLapicqueLIFNeuron"
    ),
}

#: Constant drives the state probe steps a default neuron through.
_PROBE_DRIVE: Final = (0.0,) * 4 + (1.0,) * 16 + (20.0,) * 16 + (0.0,) * 8


class _Missing:
    """Sentinel for an attribute one side carries and the other does not."""


_MISSING: Final = _Missing()


def _reference_neuron(model_name: str) -> object | None:
    """Return the neuron the bridge builds from a model name, or ``None``.

    Parameters
    ----------
    model_name : str
        The population's model name — the only identifying value the bridge
        receives.

    Returns
    -------
    object or None
        A default-constructed neuron, or ``None`` when the name does not
        resolve or the model cannot be built with no arguments. ``None`` means
        the question cannot be answered, which this module treats as a
        difference.
    """
    model = getattr(_model_registry, model_name, None)
    if model is None or not callable(model):
        return None
    try:
        # Construction may draw from or reseed the process-wide generators; the
        # caller's streams must not move because the dispatch was checked.
        with GeneratorTransaction():
            neuron: object = model()
    except Exception:
        # A model that refuses to build with no arguments cannot be what the
        # bridge builds either; the caller gets the Python path.
        return None
    return neuron


def _is_number(value: object) -> bool:
    """Return whether a value is a real number rather than a boolean flag."""
    return not isinstance(value, bool) and isinstance(value, (int, float, np.integer, np.floating))


def _values_match(left: object, right: object) -> bool:
    """Return whether two attribute values are the same for dispatch purposes.

    Equality is exact: a parameter that differs in the last bit is a different
    model, and a non-finite value never equals itself, so it is never the
    default. Numbers compare by value across the Python and NumPy scalar types,
    because a parameter that has passed through an array is still that
    parameter. Values whose equality cannot be resolved to a plain ``bool`` —
    an RNG object, an unconstrained third-party type — count as different.
    """
    if isinstance(left, _Missing) or isinstance(right, _Missing):
        return False
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        return bool(np.array_equal(left, right))
    if _is_number(left) or _is_number(right):
        return _is_number(left) and _is_number(right) and bool(left == right)
    if isinstance(left, (bool, str, bytes, type(None))):
        return type(left) is type(right) and left == right
    try:
        verdict = left == right
    except Exception:
        return False
    return verdict is True


def _attribute_map(instance: object) -> dict[str, object]:
    """Return every attribute an instance carries, from its dictionary and its slots.

    A model defined with ``__slots__`` has no instance dictionary, so reading
    only ``vars`` would report that it carries nothing — and a comparison of
    nothing against nothing would call every slotted population dispatchable.
    Slots are therefore walked across the whole class hierarchy, and a slot
    that has never been assigned is simply absent, as it is for a dictionary.
    """
    carried: dict[str, object] = dict(vars(instance)) if hasattr(instance, "__dict__") else {}
    for klass in type(instance).__mro__:
        slots = getattr(klass, "__slots__", ())
        names = (slots,) if isinstance(slots, str) else slots
        for name in names:
            if name in ("__dict__", "__weakref__") or name in carried:
                continue
            try:
                carried[name] = getattr(instance, name)
            except AttributeError:
                continue
    return carried


def _differing_attributes(neuron: object, reference: object) -> list[str]:
    """Return the attribute names on which a neuron differs from the reference."""
    carried = _attribute_map(neuron)
    expected = _attribute_map(reference)
    names = sorted(set(carried) | set(expected))
    return [
        name
        for name in names
        if not _values_match(carried.get(name, _MISSING), expected.get(name, _MISSING))
    ]


def _summarise(names: list[str]) -> str:
    """Return a bounded description of the differing attribute names."""
    if len(names) <= _NAMED_ATTRIBUTE_LIMIT:
        return ", ".join(names)
    head = ", ".join(names[:_NAMED_ATTRIBUTE_LIMIT])
    return f"{head} and {len(names) - _NAMED_ATTRIBUTE_LIMIT} more"


def _numpy_global_state() -> tuple[Any, ...]:
    """Return NumPy's global generator state as a comparable tuple (key words as bytes)."""
    state: Any = np.random.get_state(legacy=True)
    return (state[0], state[1].tobytes(), state[2], state[3], state[4])


@cache
def _probe(model_name: str) -> tuple[tuple[str, ...] | None, bool]:
    """Step a default neuron inside a generator transaction; see :func:`evolving_state`.

    Returns
    -------
    tuple
        The evolving state (``None`` when undecidable) and whether stepping
        drew from the process-wide NumPy or standard ``random`` generator.
    """
    descriptor = load_descriptor(model_name)
    if descriptor is None:
        return None, False
    declared = tuple(spec.name for spec in descriptor.state if spec.name != "v")
    stepped = _reference_neuron(model_name)
    fresh = _reference_neuron(model_name)
    step: Any = getattr(stepped, "step", None)
    if stepped is None or fresh is None or not callable(step):
        return None, False
    with GeneratorTransaction():
        numpy_before, stdlib_before = _numpy_global_state(), random.getstate()
        try:
            for drive in _PROBE_DRIVE:
                step(drive)
        except Exception:
            return None, False
        drew = _numpy_global_state() != numpy_before or random.getstate() != stdlib_before
    moved = [name for name in _differing_attributes(stepped, fresh) if name != "v"]
    return declared + tuple(name for name in moved if name not in declared), drew


def draws_process_random(model_name: str) -> bool:
    """Return whether a default neuron of this model draws from a process-wide generator.

    A native run would not advance NumPy's global generator or the standard
    ``random`` one, so every later draw of the caller would differ from a
    Python run; such a model is not dispatched. The probe itself leaves both
    generators exactly as it found them.

    Parameters
    ----------
    model_name : str
        The model the bridge would build.

    Returns
    -------
    bool
        True when stepping a default neuron through the probe drive moved
        either generator.
    """
    return _probe(model_name)[1]


def evolving_state(model_name: str) -> tuple[str, ...] | None:
    """Return the attributes besides ``v`` a default neuron changes as it steps.

    Parameters
    ----------
    model_name : str
        The model the bridge would build.

    Returns
    -------
    tuple of str or None
        The state variables the model's descriptor declares besides ``v``,
        together with the attribute names that differ from a fresh default
        neuron after stepping one through a fixed drive (0, 1, 20 and 0
        again): a drive that never fires leaves an adaptation variable still,
        so the declaration is read as well as the behaviour. ``None`` when the
        model has no descriptor or cannot be built or stepped with a float
        drive, which counts as a difference. The probe leaves the caller's
        process-wide generators exactly as it found them.
    """
    return _probe(model_name)[0]


def population_divergence(population: Population) -> str:
    """Return why the native bridge cannot build this population, or ``""``.

    Parameters
    ----------
    population : Population
        The population a caller assembled.

    Returns
    -------
    str
        Empty when every neuron equals the neuron the bridge builds from the
        model name, so dispatching preserves the network. Otherwise a
        one-line reason naming the population and what differs.

    Examples
    --------
    >>> from sc_neurocore.network import Population
    >>> population_divergence(Population("SCLapicqueLIFNeuron", 3))
    ''
    >>> population_divergence(Population("SCLapicqueLIFNeuron", 3, {"tau": 5.0}))
    "population 'SCLapicqueLIFNeuron' carries tau, which the native bridge cannot receive"
    """
    if population.model_name in NATIVE_IDENTITY_DIFFERS:
        return f"population {population.label!r}: {NATIVE_IDENTITY_DIFFERS[population.model_name]}"
    reference = _reference_neuron(population.model_name)
    if reference is None:
        return (
            f"population {population.label!r} declares model "
            f"{population.model_name!r}, which the native bridge cannot build "
            f"from its name alone"
        )
    if type(reference) is not type(next(iter(population.neurons), reference)):
        return (
            f"population {population.label!r} was built through a different "
            f"constructor than {population.model_name!r} names, which the "
            f"native bridge cannot reach"
        )
    differing: set[str] = set()
    for neuron in population.neurons:
        differing.update(_differing_attributes(neuron, reference))
    if differing:
        return (
            f"population {population.label!r} carries {_summarise(sorted(differing))}, "
            f"which the native bridge cannot receive"
        )
    hidden = evolving_state(population.model_name)
    if hidden is None:
        return (
            f"population {population.label!r}: model {population.model_name!r} cannot be "
            "stepped to find the state the native bridge would not return"
        )
    if hidden:
        return (
            f"population {population.label!r} evolves {_summarise(list(hidden))} besides v; "
            "the native bridge returns only voltages, so that state would stay at its "
            "initial value"
        )
    if draws_process_random(population.model_name):
        return (
            f"population {population.label!r}: model {population.model_name!r} draws from "
            "the process-wide random generators, which a native run would not advance"
        )
    return ""


def projection_divergences(projections: Iterable[Any]) -> list[str]:
    """Return one reason per projection feature the native runner would drop.

    Parameters
    ----------
    projections : iterable of Projection
        Every projection of the network.

    Returns
    -------
    list of str
        Empty when the runner, which receives the CSR arrays and one uniform
        delay, reproduces every projection.
    """
    reasons: list[str] = []
    for index, projection in enumerate(projections):
        name = f"projection {index}"
        if projection.delay_mode == "per_synapse":
            reasons.append(f"{name} has per-synapse delays; the runner takes one uniform delay")
        if float(projection.weight_threshold) != 0.0:
            reasons.append(f"{name} prunes with weight_threshold; the runner does not")
        if projection.plasticity:
            reasons.append(f"{name} is plastic ({projection.plasticity}); the runner is not")
    return reasons


def network_divergences(populations: Iterable[Population]) -> list[str]:
    """Return one reason per population the native bridge cannot build faithfully.

    Parameters
    ----------
    populations : iterable of Population
        Every population of the network.

    Returns
    -------
    list of str
        Empty when the whole network can be dispatched without changing it.
    """
    return [reason for reason in map(population_divergence, populations) if reason]


__all__ = [
    "BRIDGE_CONSTRUCTOR_ARGUMENTS",
    "NATIVE_IDENTITY_DIFFERS",
    "draws_process_random",
    "evolving_state",
    "network_divergences",
    "population_divergence",
    "projection_divergences",
]
