# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Simulator registry resolved from packaging entry points

"""Resolve a model's simulator from declared entry points, so models never import their executor.

A model definition calls :func:`resolve_simulator` with its model id and runs the returned
callable. The executor package declares the callable as an entry point in the
``sc_neurocore.simulators`` group, so the dependency points from the executor to this module
and is visible in the installed package metadata.

The resolution is strict, because simulation results are evidence:

- Two registrations of one model id, from any installed distribution, are refused, naming
  both. Entry-point order never picks a winner.
- A registration counts only when its distribution is in :data:`ALLOWED_DISTRIBUTIONS`.
  Another installed distribution cannot take over a model's executor.
- A registration that fails to load is refused, naming the entry point and the error. It is
  never skipped in favour of another backend.
- Without installed metadata for an allowed distribution, for example when the sources run
  from a checkout that was never installed, resolution is refused. No executor module is
  imported by name.

The installed metadata is scanned once per process, on the first resolution, and resolved
simulators are cached for the life of the process. :func:`clear_simulator_cache` empties both,
for example after the installed metadata changed.
"""

from __future__ import annotations

import importlib.metadata
import re
from collections.abc import Callable
from importlib.metadata import EntryPoint
from typing import Any, Final

__all__ = [
    "ALLOWED_DISTRIBUTIONS",
    "SIMULATOR_ENTRY_POINT_GROUP",
    "DuplicateSimulatorRegistration",
    "NoDistributionMetadata",
    "NoSimulatorRegistered",
    "SimulatorLoadRefused",
    "SimulatorRegistryError",
    "clear_simulator_cache",
    "resolve_simulator",
]

SIMULATOR_ENTRY_POINT_GROUP: Final = "sc_neurocore.simulators"
"""Packaging entry-point group in which executors declare their simulators."""

ALLOWED_DISTRIBUTIONS: Final[frozenset[str]] = frozenset({"sc-neurocore"})
"""Normalised names of the distributions whose registrations count."""

_Registration = tuple[str, EntryPoint]
"""A registration: the normalised name of the declaring distribution, and its entry point."""

_CACHE: dict[str, Callable[..., Any]] = {}
_SCAN: list[tuple[bool, dict[str, list[_Registration]]]] = []


class SimulatorRegistryError(RuntimeError):
    """Base class of every refusal to resolve a simulator."""


class NoDistributionMetadata(SimulatorRegistryError):
    """No allowed distribution is installed, so no registration can be read."""


class NoSimulatorRegistered(SimulatorRegistryError):
    """No allowed distribution registers a simulator for the model id."""


class DuplicateSimulatorRegistration(SimulatorRegistryError):
    """More than one entry point registers a simulator for the model id."""


class SimulatorLoadRefused(SimulatorRegistryError):
    """The registered entry point did not load to a callable."""


def _normalise(name: str) -> str:
    """Return a distribution name in the normalised form of the packaging specification."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _describe(distribution: str, entry_point: EntryPoint) -> str:
    """Return ``name = value`` with the declaring distribution, for refusal messages."""
    return f"{entry_point.name} = {entry_point.value} (from {distribution})"


def _scan() -> tuple[bool, dict[str, list[_Registration]]]:
    """Return the installed registrations, reading the metadata on the first call only.

    Distributions are read in path order, and a name already seen is skipped, as
    ``importlib.metadata.entry_points`` does, so the first distribution of a name wins.

    Returns
    -------
    tuple[bool, dict[str, list[_Registration]]]
        Whether an allowed distribution is installed, and the registrations of every model id
        in the ``sc_neurocore.simulators`` group.
    """
    if not _SCAN:
        seen: set[str] = set()
        registrations: dict[str, list[_Registration]] = {}
        for distribution in importlib.metadata.distributions():
            name = _normalise(distribution.metadata["Name"] or "")
            if name in seen:
                continue
            seen.add(name)
            for entry_point in distribution.entry_points.select(group=SIMULATOR_ENTRY_POINT_GROUP):
                registrations.setdefault(entry_point.name, []).append((name, entry_point))
        _SCAN.append((bool(seen & ALLOWED_DISTRIBUTIONS), registrations))
    return _SCAN[0]


def resolve_simulator(model_id: str) -> Callable[..., Any]:
    """Return the simulator registered for ``model_id``.

    Parameters
    ----------
    model_id : str
        The model's registry id, the entry-point name in the ``sc_neurocore.simulators`` group.

    Returns
    -------
    Callable[..., Any]
        The loaded simulator callable, cached for the life of the process.

    Raises
    ------
    NoDistributionMetadata
        If no allowed distribution has installed metadata.
    DuplicateSimulatorRegistration
        If more than one installed entry point, from any distribution, registers ``model_id``.
    NoSimulatorRegistered
        If no allowed distribution registers ``model_id``. A registration from a distribution
        outside the allowlist is named in the message.
    SimulatorLoadRefused
        If the registered entry point fails to load or does not load to a callable.
    """
    cached = _CACHE.get(model_id)
    if cached is not None:
        return cached
    allowed_installed, registrations = _scan()
    if not allowed_installed:
        raise NoDistributionMetadata(
            f"no backend registered for {model_id!r}: no installed distribution metadata "
            f"for {sorted(ALLOWED_DISTRIBUTIONS)}; install the package so its "
            f"{SIMULATOR_ENTRY_POINT_GROUP!r} entry points are declared"
        )
    candidates = registrations.get(model_id, [])
    if len(candidates) > 1:
        named = "; ".join(_describe(*candidate) for candidate in candidates)
        raise DuplicateSimulatorRegistration(
            f"{len(candidates)} simulators are registered for {model_id!r}: {named}"
        )
    allowed = [candidate for candidate in candidates if candidate[0] in ALLOWED_DISTRIBUTIONS]
    if not allowed:
        ignored = f"; ignored {_describe(*candidates[0])}" if candidates else ""
        raise NoSimulatorRegistered(
            f"no backend registered for {model_id!r} by {sorted(ALLOWED_DISTRIBUTIONS)}"
            f"{ignored}; reinstall the package if its entry points changed"
        )
    distribution, entry_point = allowed[0]
    try:
        simulator = entry_point.load()
    except Exception as error:  # every load failure is a refusal, never a skip
        raise SimulatorLoadRefused(
            f"the simulator {_describe(distribution, entry_point)} failed to load: "
            f"{type(error).__name__}: {error}"
        ) from error
    if not callable(simulator):
        raise SimulatorLoadRefused(
            f"the simulator {_describe(distribution, entry_point)} loaded a "
            f"{type(simulator).__name__}, not a callable"
        )
    resolved: Callable[..., Any] = simulator
    _CACHE[model_id] = resolved
    return resolved


def clear_simulator_cache() -> None:
    """Forget the scanned metadata and every resolved simulator, so the next resolution reads again."""
    _CACHE.clear()
    _SCAN.clear()
