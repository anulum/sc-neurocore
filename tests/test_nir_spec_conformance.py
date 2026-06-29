# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR spec conformance / drift detector

"""Conformance of the NIR bridge against the *installed* NIR specification.

This is a drift detector. Instead of hard-coding the list of NIR primitives, it
enumerates the primitives the installed ``nir`` package actually defines and fails
if SC-NeuroCore stops covering one. That turns "a new NIR release added a primitive
we do not handle" from a silent gap into a red test — the early-warning gate that
keeps the bridge in lock-step with upstream NIR.
"""

from __future__ import annotations

import inspect

import pytest

nir = pytest.importorskip("nir")

from nir.ir import NIRNode

from sc_neurocore.nir_bridge.node_map import NODE_MAP

# ``NIRGraph`` is the nested-subgraph container; the parser inlines it directly
# (see nir_bridge/parser.py), so it is handled outside NODE_MAP.
_PARSER_HANDLED: frozenset[str] = frozenset({"NIRGraph"})

# NIR primitives deliberately not lowered, each with the reason. Empty today:
# every nir 1.0.x primitive is supported. A future entry here must be a conscious
# decision (with a reason), never an accidental omission.
_KNOWN_UNSUPPORTED: dict[str, str] = {}


def _installed_nir_primitives() -> dict[str, type]:
    """Return every concrete NIR primitive the installed ``nir`` exposes."""
    primitives: dict[str, type] = {}
    for name in dir(nir):
        obj = getattr(nir, name)
        if inspect.isclass(obj) and issubclass(obj, NIRNode) and obj is not NIRNode:
            primitives[name] = obj
    return primitives


def test_node_map_covers_every_installed_nir_primitive() -> None:
    """Every NIR primitive is mapped, parser-handled, or documented as unsupported."""
    installed = _installed_nir_primitives()
    mapped = {cls.__name__ for cls in NODE_MAP}
    covered = mapped | _PARSER_HANDLED | set(_KNOWN_UNSUPPORTED)
    uncovered = sorted(set(installed) - covered)
    assert not uncovered, (
        f"Installed nir {nir.__version__} exposes primitives SC-NeuroCore does not "
        f"handle: {uncovered}. Map them in NODE_MAP, handle them in the parser, or "
        f"add them to _KNOWN_UNSUPPORTED with a reason before the gap ships."
    )


def test_node_map_has_no_stale_entries() -> None:
    """Every NODE_MAP key is still a real primitive in the installed nir."""
    installed = set(_installed_nir_primitives().values())
    stale = sorted(cls.__name__ for cls in NODE_MAP if cls not in installed)
    assert not stale, (
        f"NODE_MAP maps classes the installed nir {nir.__version__} no longer "
        f"defines: {stale}"
    )


def test_known_unsupported_names_are_real_nir_primitives() -> None:
    """Guard the allow-list against rot: documented-unsupported names must exist."""
    installed = set(_installed_nir_primitives())
    bogus = sorted(set(_KNOWN_UNSUPPORTED) - installed)
    assert not bogus, (
        f"_KNOWN_UNSUPPORTED lists names not present in nir {nir.__version__}: {bogus}"
    )


def test_parser_handled_names_are_real_nir_primitives() -> None:
    """Guard the parser-handled allow-list against rot."""
    installed = set(_installed_nir_primitives())
    missing = sorted(_PARSER_HANDLED - installed)
    assert not missing, (
        f"_PARSER_HANDLED references primitives not present in nir {nir.__version__}: "
        f"{missing}"
    )
