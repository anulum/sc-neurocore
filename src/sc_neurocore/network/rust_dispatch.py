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

One residual is stated rather than hidden: the check reads the neurons, so it
sees a parameter changed in place after construction. It cannot see a change
made *after* the check and before the bridge builds its own neurons, and it
says nothing about whether the Rust defaults equal the Python defaults — that
is a parity question, held by the parity suites.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np

from sc_neurocore.network.population import Population
from sc_neurocore.neurons import models as _model_registry

#: Everything the native bridge receives about a population. The rest of what a
#: population knows — its constructor parameters and its evolved state — has no
#: route across the boundary.
BRIDGE_CONSTRUCTOR_ARGUMENTS: Final = ("model_name", "n")

#: How many differing attribute names a reason names before it summarises.
_NAMED_ATTRIBUTE_LIMIT: Final = 4


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
    >>> population_divergence(Population("AdExNeuron", 3))
    ''
    >>> population_divergence(Population("AdExNeuron", 3, {"v_rest": -60.0}))
    "population 'AdExNeuron' carries v_rest, which the native bridge cannot receive"
    """
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
    if not differing:
        return ""
    return (
        f"population {population.label!r} carries {_summarise(sorted(differing))}, "
        f"which the native bridge cannot receive"
    )


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
    "network_divergences",
    "population_divergence",
]
