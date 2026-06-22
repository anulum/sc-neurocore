# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Model catalogue completeness gate

"""Catalogue integrity gate for the neuron model library.

Guarantees that the model catalogue cannot silently drift as the library grows:
every model class defined on disk is registered, every registered model loads,
instantiates, and is browsable through the Studio catalogue, and the catalogue
size matches the registry exactly. A new model that is added but not registered,
or a registered model that becomes unbrowsable, fails this gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.models import _load_class, get_model_detail, list_models

_MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "sc_neurocore" / "neurons" / "models"
_MODEL_NAME_SUFFIXES = ("Neuron", "Cell", "Unit", "Model", "Map", "Field")


def _disk_dataclass_models() -> dict[str, str]:
    """Return every on-disk top-level ``@dataclass`` model class -> module stem.

    A class qualifies as a catalogue model when it is decorated with
    ``@dataclass`` and its name ends in a model suffix. Private helper
    dataclasses (underscore-prefixed) are excluded.
    """

    discovered: dict[str, str] = {}
    for path in sorted(_MODELS_DIR.glob("*.py")):
        if path.stem == "__init__":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            is_dataclass = any(
                (isinstance(dec, ast.Name) and dec.id == "dataclass")
                or (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "dataclass"
                )
                for dec in node.decorator_list
            )
            if is_dataclass and node.name.endswith(_MODEL_NAME_SUFFIXES):
                discovered.setdefault(node.name, path.stem)
    return discovered


def test_every_disk_dataclass_model_is_registered() -> None:
    """Tier 0: no model can be defined on disk yet missing from the registry."""

    discovered = _disk_dataclass_models()
    unregistered = sorted(name for name in discovered if name not in _CLASS_TO_MODULE)
    assert unregistered == [], (
        f"model classes defined on disk but not registered in _CLASS_TO_MODULE: {unregistered}"
    )


@pytest.mark.parametrize("name", sorted(_CLASS_TO_MODULE))
def test_registered_model_loads_and_instantiates(name: str) -> None:
    """Tier 0: every registered model loads from its module and instantiates."""

    cls = _load_class(name)
    assert cls.__name__ == name
    instance = cls()
    assert instance is not None


@pytest.mark.parametrize("name", sorted(_CLASS_TO_MODULE))
def test_registered_model_is_browsable(name: str) -> None:
    """Tier 0: every registered model exposes path-free catalogue metadata."""

    detail = get_model_detail(name)
    assert detail is not None, f"{name} is registered but not browsable (get_model_detail is None)"
    assert detail["name"] == name
    assert detail["state_vars"], f"{name} has no state variables"
    assert isinstance(detail["params"], list)
    assert isinstance(detail["dt"], float)


def test_registered_module_points_to_existing_file() -> None:
    """Tier 0: no registry entry points at a module file that does not exist."""

    on_disk = {path.stem for path in _MODELS_DIR.glob("*.py")}
    orphans = sorted(
        f"{name} -> {module}" for name, module in _CLASS_TO_MODULE.items() if module not in on_disk
    )
    assert orphans == [], f"registry entries with missing module files: {orphans}"


def test_catalogue_size_matches_registry() -> None:
    """Tier 0: the browsable catalogue surfaces every registered model exactly."""

    catalogue = {model["name"] for model in list_models()}
    registry = set(_CLASS_TO_MODULE)
    invisible = sorted(registry - catalogue)
    extra = sorted(catalogue - registry)
    assert invisible == [], f"registered models missing from the catalogue: {invisible}"
    assert extra == [], f"catalogue models absent from the registry: {extra}"
