# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Auto-discover adapter plugins via importlib.metadata

"""Discover and register adapter classes from Python entry-point metadata.

The adapter registry stores classes, not arbitrary callables.  File-format
import functions therefore enter the registry through thin adapter classes in
``sc_neurocore.adapters.importers`` while third-party packages may expose any
class-compatible adapter target through the ``sc_neurocore.adapters`` entry
point group.
"""

from __future__ import annotations

from collections.abc import Iterable
import importlib
import importlib.metadata
from importlib.metadata import EntryPoint
from typing import Any, Final, cast

from .registry import registry

ADAPTER_ENTRY_POINT_GROUP: Final = "sc_neurocore.adapters"
"""Python packaging entry-point group used for adapter plugin discovery."""

FIRST_PARTY_ADAPTERS: Final[dict[str, str]] = {
    "neuroml": "sc_neurocore.adapters.importers:NeuroMLImporter",
    "sonata": "sc_neurocore.adapters.importers:SONATAImporter",
    "spikeinterface": "sc_neurocore.adapters.importers:SpikeInterfaceImporter",
    "holonomic_dna_storage": "sc_neurocore.adapters.holonomic.dna_storage:DNAEncoder",
    "holonomic_grn": "sc_neurocore.adapters.holonomic.grn:GeneticRegulatoryLayer",
    "holonomic_neuromodulation": (
        "sc_neurocore.adapters.holonomic.neuromodulation:NeuromodulatorSystem"
    ),
}
"""Built-in adapter entry-point targets mirrored in ``pyproject.toml``."""

_DISCOVERY_ERRORS: Final[tuple[type[Exception], ...]] = (
    AttributeError,
    ImportError,
    KeyError,
    TypeError,
    ValueError,
)


def _resolve_target(target: str) -> type:
    module_name, separator, attribute_path = target.partition(":")
    if separator != ":" or not module_name or not attribute_path:
        raise ValueError(f"Adapter entry point target must be 'module:attribute': {target!r}")

    resolved: object = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        resolved = getattr(resolved, attribute)

    if not isinstance(resolved, type):
        raise TypeError(f"Adapter entry point target is not a class: {target!r}")
    return resolved


def _entry_points(group: str) -> tuple[EntryPoint, ...]:
    try:
        selected = importlib.metadata.entry_points(group=group)
    except TypeError:
        selectable = cast(Any, importlib.metadata.entry_points())
        if hasattr(selectable, "select"):
            selected = selectable.select(group=group)
        else:
            selected = selectable.get(group, ())

    return tuple(cast(Iterable[EntryPoint], selected))


def _register_adapter(name: str, adapter_type: type) -> None:
    try:
        registry.register("adapter", name)(adapter_type)
    except KeyError:
        return


def _ensure_holonomic_adapters_loaded() -> None:
    importlib.import_module("sc_neurocore.adapters.holonomic")


def discover_adapters(
    *,
    include_first_party: bool = True,
    include_entry_points: bool = True,
) -> dict[str, type]:
    """Discover adapter classes and register them in the global registry.

    Parameters
    ----------
    include_first_party : bool, default=True
        Register the built-in adapter importers declared by
        :data:`FIRST_PARTY_ADAPTERS`.  This source-level path keeps editable
        checkouts wired even before packaging metadata is installed.
    include_entry_points : bool, default=True
        Load installed third-party plugins from
        :data:`ADAPTER_ENTRY_POINT_GROUP` through ``importlib.metadata``.

    Returns
    -------
    dict[str, type]
        Mapping from registry name to the discovered adapter class. Duplicate
        registry entries are tolerated so repeated discovery is idempotent.
    """
    found: dict[str, type] = {}

    if include_first_party:
        for name, target in FIRST_PARTY_ADAPTERS.items():
            try:
                adapter_type = _resolve_target(target)
            except _DISCOVERY_ERRORS:
                continue
            _register_adapter(name, adapter_type)
            found[name] = adapter_type
        _ensure_holonomic_adapters_loaded()

    if not include_entry_points:
        return found

    for ep in _entry_points(ADAPTER_ENTRY_POINT_GROUP):
        try:
            loaded = ep.load()
            if not isinstance(loaded, type):
                raise TypeError(f"Adapter entry point is not a class: {ep.name!r}")
            name = ep.name
            _register_adapter(name, loaded)
            found[name] = loaded
        except _DISCOVERY_ERRORS:
            continue

    return found
