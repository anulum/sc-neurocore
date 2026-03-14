# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import importlib.metadata

from .registry import registry


def discover_adapters() -> dict[str, type]:
    """Auto-discover adapter plugins via importlib.metadata entry points.

    Looks for entry points in group ``sc_neurocore.adapters``.
    Each entry point should point to a class inheriting BaseStochasticAdapter.
    """
    found = {}
    try:
        eps = importlib.metadata.entry_points(group="sc_neurocore.adapters")
    except TypeError:
        eps = importlib.metadata.entry_points().get("sc_neurocore.adapters", [])

    for ep in eps:
        try:
            cls = ep.load()
            name = ep.name
            registry.register("adapter", name)(cls)
            found[name] = cls
        except (ImportError, KeyError, AttributeError):
            continue

    return found
