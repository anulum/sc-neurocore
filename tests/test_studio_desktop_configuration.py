# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop configuration preservation and refusal

"""Preserve actual profile files across replacement and unsupported configuration."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.desktop.configuration import (
    CONFIGURATION_SCHEMA,
    DesktopConfiguration,
    read_configuration,
    write_configuration,
)


def test_configuration_atomically_retains_private_origin_and_project_root(tmp_path: Path) -> None:
    """Read and replace the real retained origin without temporary-file residue.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable configuration directory and project namespace.
    """
    path = tmp_path / "desktop.json"
    assert read_configuration(path) is None
    original = DesktopConfiguration(18031, tmp_path / "projects")
    write_configuration(path, original)
    assert read_configuration(path) == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    replacement = DesktopConfiguration(18032, tmp_path / "restored-projects")
    write_configuration(path, replacement)
    assert read_configuration(path) == replacement
    assert set(tmp_path.iterdir()) == {path}
    assert not replacement.project_root.exists()


def test_configuration_refuses_a_directory_without_changing_its_contents(tmp_path: Path) -> None:
    """Keep an unreadable configuration location and its existing data intact.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable profile containing a directory at the configuration path.
    """
    path = tmp_path / "desktop.json"
    path.mkdir()
    original = path / "original.json"
    original.write_bytes(b'{"seed":"9223372036854775807"}')
    with pytest.raises(AuthoredRefusal, match="configuration cannot be read"):
        read_configuration(path)
    assert original.read_bytes() == b'{"seed":"9223372036854775807"}'
    assert set(path.iterdir()) == {original}


def test_configuration_failed_replacement_cleans_only_its_temporary_file(tmp_path: Path) -> None:
    """Preserve a conflicting directory when atomic replacement fails.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable configuration parent with pre-existing user data.
    """
    path = tmp_path / "desktop.json"
    path.mkdir()
    original = path / "original.json"
    original.write_bytes(b"retained data")
    with pytest.raises(OSError):
        write_configuration(path, DesktopConfiguration(18031, tmp_path / "projects"))
    assert set(tmp_path.iterdir()) == {path}
    assert set(path.iterdir()) == {original}
    assert original.read_bytes() == b"retained data"


def test_configuration_missing_parent_does_not_create_profile_state(tmp_path: Path) -> None:
    """Fail before allocating a temporary file when the profile is unprepared.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable existing ancestor of the absent profile directory.
    """
    path = tmp_path / "absent" / "desktop.json"
    with pytest.raises(FileNotFoundError):
        write_configuration(path, DesktopConfiguration(18031, tmp_path / "projects"))
    assert set(tmp_path.iterdir()) == set()


@pytest.mark.parametrize(
    "document",
    [
        b"invalid JSON",
        b"x" * 16_385,
        json.dumps(
            {"schema_version": "future", "port": 18031, "project_root": "/projects"}
        ).encode(),
        json.dumps(
            {"schema_version": CONFIGURATION_SCHEMA, "port": True, "project_root": "/projects"}
        ).encode(),
        json.dumps(
            {"schema_version": CONFIGURATION_SCHEMA, "port": 0, "project_root": "/projects"}
        ).encode(),
        json.dumps(
            {"schema_version": CONFIGURATION_SCHEMA, "port": 18031, "project_root": "relative"}
        ).encode(),
        json.dumps(
            {
                "schema_version": CONFIGURATION_SCHEMA,
                "port": 18031,
                "project_root": "/projects",
                "unknown": True,
            }
        ).encode(),
    ],
)
def test_invalid_configuration_is_refused_without_rewriting_it(
    tmp_path: Path, document: bytes
) -> None:
    """Refuse malformed or future data instead of silently replacing the user's file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable profile directory.
    document : bytes
        Actual invalid or unsupported configuration contents.
    """
    path = tmp_path / "desktop.json"
    path.write_bytes(document)
    with pytest.raises(AuthoredRefusal):
        read_configuration(path)
    assert path.read_bytes() == document
    assert set(tmp_path.iterdir()) == {path}
