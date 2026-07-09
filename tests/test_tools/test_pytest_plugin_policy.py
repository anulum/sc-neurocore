# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pytest plugin policy tests

"""Tests for repository-level pytest plugin policy."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pytest_config_disables_ambient_nengo_plugin() -> None:
    """The committed pytest config disables ambient ``pytest_nengo`` loading."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = pyproject["tool"]["pytest"]["ini_options"]

    assert pytest_options["addopts"] == ["-p", "no:nengo"]


def test_pytest_nengo_plugin_is_not_registered(pytestconfig: pytest.Config) -> None:
    """The active pytest session does not register the ambient Nengo plugin."""
    assert pytestconfig.pluginmanager.get_plugin("nengo") is None


def test_nengo_is_not_imported_by_pytest_collection_policy() -> None:
    """The disabled plugin keeps Nengo out of collection for unrelated tests."""
    assert "nengo" not in sys.modules
