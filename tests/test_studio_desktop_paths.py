# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop user profile path behavior

"""Retain existing projects and isolate actual desktop directories under XDG."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.paths import DesktopPaths
from sc_neurocore.studio.project_paths import PROJECT_ROOT_ENV


def test_desktop_paths_preserve_legacy_and_explicit_project_locations(tmp_path: Path) -> None:
    """Prepare private XDG state without moving a legacy project namespace.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable home and application directories.
    """
    paths = DesktopPaths.from_environment({"HOME": str(tmp_path)})
    assert paths.config == tmp_path / ".config/sc-neurocore/studio"
    assert paths.project_root({}) == paths.data / "projects"
    assert not paths.data.exists()
    paths.legacy_projects.mkdir(parents=True)
    original = paths.legacy_projects / "original.json"
    original.write_text('{"seed":"9007199254740993"}')
    assert paths.project_root({}) == paths.legacy_projects
    stored = tmp_path / "chosen"
    assert paths.project_root({}, stored) == stored
    override = tmp_path / "override"
    assert paths.project_root({PROJECT_ROOT_ENV: str(override)}, stored) == override
    paths.prepare()
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o700
        for path in (paths.config, paths.data, paths.state, paths.cache)
    )
    assert original.read_text() == '{"seed":"9007199254740993"}'
    assert not override.exists()


@pytest.mark.parametrize(
    "key", ["HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"]
)
def test_desktop_paths_refuse_relative_user_roots(tmp_path: Path, key: str) -> None:
    """Refuse relative user roots before preparing any profile state.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable absolute fallback home.
    key : str
        User base directory being configured incorrectly.
    """
    with pytest.raises(DesktopStartRefused, match="absolute directory"):
        DesktopPaths.from_environment({"HOME": str(tmp_path), key: "relative"})
    assert list(tmp_path.iterdir()) == []


def test_desktop_paths_refuse_a_file_as_home_without_replacing_it(tmp_path: Path) -> None:
    """Refuse a non-directory home before creating application directories.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable parent containing an incorrectly configured home file.
    """
    home = tmp_path / "home"
    home.write_bytes(b"existing user data")
    with pytest.raises(DesktopStartRefused, match="HOME must name a directory"):
        DesktopPaths.from_environment({"HOME": str(home)})
    assert home.read_bytes() == b"existing user data"
    assert set(tmp_path.iterdir()) == {home}


@pytest.mark.parametrize(
    "key", ["HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"]
)
def test_desktop_paths_refuse_symlink_cycles_without_creating_state(
    tmp_path: Path, key: str
) -> None:
    """Refuse an unresolvable profile base without replacing its symbolic link.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable home and parent of a self-referential symbolic link.
    key : str
        User base directory configured to the actual filesystem cycle.
    """
    cycle = tmp_path / "cycle"
    cycle.symlink_to(cycle.name)
    with pytest.raises(DesktopStartRefused, match=f"{key} could not be resolved"):
        DesktopPaths.from_environment({"HOME": str(tmp_path), key: str(cycle)})
    assert cycle.is_symlink()
    assert cycle.readlink() == Path(cycle.name)
    assert set(tmp_path.iterdir()) == {cycle}


def test_desktop_paths_refuse_unreadable_legacy_projects_without_falling_back(
    tmp_path: Path,
) -> None:
    """Preserve an unresolvable legacy namespace instead of opening an empty one.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable home containing the actual legacy directory symlink cycle.
    """
    paths = DesktopPaths.from_environment({"HOME": str(tmp_path)})
    paths.legacy_projects.parent.mkdir(parents=True)
    paths.legacy_projects.symlink_to(paths.legacy_projects.name)
    with pytest.raises(DesktopStartRefused, match="project directory cannot be read"):
        paths.project_root({})
    assert paths.legacy_projects.is_symlink()
    assert paths.legacy_projects.readlink() == Path(paths.legacy_projects.name)
    assert not paths.data.exists()
