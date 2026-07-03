# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tournament selector contract tests

"""Strict contract tests for evolutionary-substrate tournament selection."""

from __future__ import annotations

import importlib
from pathlib import Path
import runpy
from types import ModuleType

import numpy as np
import pytest

import sc_neurocore.evo_substrate.evo_substrate as evo_module
from sc_neurocore.evo_substrate.evo_substrate import Organism, TournamentSelector


def test_optional_rust_evo_import_failure_keeps_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the Python evolutionary substrate available without the Rust extension."""
    real_import_module = importlib.import_module

    def missing_evo_extension(name: str, package: str | None = None) -> ModuleType:
        """Raise only for the optional local evo-substrate extension."""
        if name == "sc_neurocore.evo_substrate.evo_substrate_core":
            raise ImportError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_evo_extension)
    module_path = Path(evo_module.__file__).resolve()
    namespace = runpy.run_path(
        str(module_path),
        run_name="_sc_neurocore_evo_substrate_import_fallback",
    )

    assert namespace["_HAS_RUST_EVO"] is False
    assert namespace["_ec"] is None


def test_tournament_select_rejects_empty_population() -> None:
    """Reject empty tournament populations before RNG sampling."""
    selector = TournamentSelector(tournament_size=3)
    empty_population: list[Organism] = []
    rng = np.random.default_rng(7)

    with pytest.raises(ValueError, match="non-empty population"):
        selector.select(empty_population, rng)


def test_tournament_select_n_propagates_empty_population_rejection() -> None:
    """Reject batched selection from an empty population consistently."""
    selector = TournamentSelector(tournament_size=2)
    empty_population: list[Organism] = []
    rng = np.random.default_rng(11)

    with pytest.raises(ValueError, match="non-empty population"):
        selector.select_n(empty_population, 2, rng)
