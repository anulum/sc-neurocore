# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Removed engine-loader shims stay documented and replaceable

"""Bind the changelog's removed engine-loader shims to the modules on both sides.

Release 4.0.0 stopped re-exporting a few native engine loaders from public Python
modules. The changelog names each one and the engine module that still provides it.
These tests keep that statement true: the old module no longer offers the name, and
the named bridge module defines it, and lists it in ``__all__`` when the module
declares one.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Old public module -> (engine module that provides the loaders, the removed names).
REMOVED_SHIMS = {
    "sc_neurocore.bridges.quantum_annealing": (
        "sc_neurocore_engine.quantum",
        (
            "get_ising_energy",
            "get_batch_ising_energy",
            "get_simulated_annealing",
            "has_full_quantum_annealing_backend",
        ),
    ),
    "sc_neurocore.optics.photonic_emitter": (
        "sc_neurocore_engine.photonics",
        (
            "get_crosstalk_analyzer",
            "get_crosstalk_bank_analyzer",
            "get_crosstalk_pair_analyzer",
            "has_full_photonic_crosstalk_backend",
        ),
    ),
    "sc_neurocore.studio.models": ("sc_neurocore_engine.studio", ("get_batch_simulate",)),
}


def _release_notes(heading: str) -> str:
    """Return one section of the root changelog.

    Parameters
    ----------
    heading : str
        Section heading after ``## ``, for example ``[4.1.0]``.

    Returns
    -------
    str
        Text between that heading and the next section heading.
    """
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    start = text.index(f"## {heading}")
    return text[start : text.index("\n## [", start + 1)]


def _bridge_exports(engine_module: str) -> tuple[set[str] | None, set[str]]:
    """Read the names a bridge module lists in ``__all__`` and defines as functions.

    The bridge sources are parsed rather than imported, so the check needs no
    native engine build.

    Parameters
    ----------
    engine_module : str
        Dotted module name under ``sc_neurocore_engine``.

    Returns
    -------
    tuple
        The names in ``__all__``, or ``None`` when the module declares none, and the
        names of top-level function definitions.
    """
    path = ROOT / "bridge" / Path(*engine_module.split(".")).with_suffix(".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    listed: set[str] | None = None
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            listed = set(ast.literal_eval(node.value))
    return listed, defined


@pytest.mark.parametrize("old_module", sorted(REMOVED_SHIMS))
def test_old_module_no_longer_offers_the_removed_loaders(old_module: str) -> None:
    """Importing the old public module gives none of the removed loader names.

    Parameters
    ----------
    old_module : str
        Public module that re-exported the loaders before 4.0.0.
    """
    module = importlib.import_module(old_module)
    _engine_module, names = REMOVED_SHIMS[old_module]
    assert [name for name in names if hasattr(module, name)] == []


@pytest.mark.parametrize("old_module", sorted(REMOVED_SHIMS))
def test_engine_bridge_still_provides_each_removed_loader(old_module: str) -> None:
    """The engine module named as the replacement defines every removed name and exports it.

    Parameters
    ----------
    old_module : str
        Public module that re-exported the loaders before 4.0.0.
    """
    engine_module, names = REMOVED_SHIMS[old_module]
    listed, defined = _bridge_exports(engine_module)
    assert set(names) <= defined
    assert listed is None or set(names) <= listed


def test_changelog_names_every_removed_loader_and_its_replacement() -> None:
    """The 4.1.0 notes name each removed loader, its old module and its engine module."""
    notes = " ".join(_release_notes("[4.1.0]").split())
    for old_module, (engine_module, names) in REMOVED_SHIMS.items():
        assert f"`{old_module.removeprefix('sc_neurocore.')}`" in notes
        assert f"`{engine_module}`" in notes
        for name in names:
            assert f"`{name}`" in notes
