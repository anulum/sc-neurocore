# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed desktop test fixtures

"""Require real sealed desktop bundles only for the installed-application lane."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop.installation import DesktopInstallation


@pytest.fixture(scope="session")
def desktop_bundles() -> tuple[Path, Path]:
    """Require two complete real-runtime inputs for the explicit installation lane.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Prepared desktop bundles with distinct distribution versions and one
        qualified data-runtime family. No source-checkout runtime is substituted.
    """
    raw = os.environ.get("SC_NEUROCORE_DESKTOP_TEST_BUNDLES")
    assert raw, "Set SC_NEUROCORE_DESKTOP_TEST_BUNDLES to the JSON array of two real bundles."
    values: object = json.loads(raw)
    assert isinstance(values, list) and len(values) == 2
    assert all(isinstance(value, str) for value in values)
    first, second = Path(values[0]), Path(values[1])
    assert first.is_absolute() and second.is_absolute() and first != second
    return first, second


@pytest.fixture
def installation_environment(tmp_path: Path) -> dict[str, str]:
    """Select the same disposable user for API, bundled commands and OS launch.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Fresh home and XDG parent directory.

    Returns
    -------
    dict of str to str
        Real process environment with private research directories.
    """
    home = tmp_path / "home with spaces"
    home.mkdir()
    environ = dict(os.environ)
    environ.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data with spaces %"),
        XDG_STATE_HOME=str(tmp_path / "state"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
    )
    return environ


@pytest.fixture
def installation(installation_environment: dict[str, str]) -> Iterator[DesktopInstallation]:
    """Retain the actual copied code and remove it through normal uninstall.

    Parameters
    ----------
    installation_environment : dict of str to str
        Same private environment used by real external processes.

    Yields
    ------
    DesktopInstallation
        Production lifecycle owner whose research profile is preserved.
    """
    instance = DesktopInstallation(installation_environment)
    try:
        yield instance
    finally:
        if instance.root.exists():
            instance.uninstall()
