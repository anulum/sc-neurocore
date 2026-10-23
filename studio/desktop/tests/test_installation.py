# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real desktop installation lifecycle integration

"""Install complete application versions and preserve actual user profile files."""

from __future__ import annotations

import fcntl
import os
import stat
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.configuration import DesktopConfiguration, write_configuration
from sc_neurocore.studio.desktop.installation import DesktopInstallation

pytest_plugins = ["studio.desktop.tests.installation_fixtures"]


def test_upgrade_rollback_and_uninstall_keep_the_research_profile(
    installation: DesktopInstallation, desktop_bundles: tuple[Path, Path]
) -> None:
    """Exercise real independent copies, atomic selection and retained user files.

    Parameters
    ----------
    installation : DesktopInstallation
        Fresh actual per-user lifecycle controller.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Two complete runtimes with distinct distribution versions.
    """
    first = installation.install(desktop_bundles[0])
    root = installation.root
    selected = root / "current"
    assert os.readlink(selected) == "versions/" + first.build_id
    assert (selected / "runtime/bin/python3.12").stat().st_ino != (
        desktop_bundles[0] / "runtime/bin/python3.12"
    ).stat().st_ino
    project = installation.profile.data / "projects" / "retained.json"
    project.parent.mkdir()
    project.write_bytes(b'{"state":{"seed":"9223372036854775807"}}')
    configuration = installation.profile.config / "desktop.json"
    write_configuration(configuration, DesktopConfiguration(18031, project.parent))
    before = {path: path.read_bytes() for path in [project, configuration]}
    assert installation.install(desktop_bundles[0]) == first
    second = installation.install(desktop_bundles[1])
    assert second.build_id != first.build_id
    assert os.readlink(selected) == "versions/" + second.build_id
    assert set(path.name for path in (root / "versions").iterdir()) == {
        first.build_id,
        second.build_id,
    }
    assert installation.rollback(first.build_id) == first
    assert os.readlink(selected) == "versions/" + first.build_id
    assert {path: path.read_bytes() for path in before} == before
    registration = installation.registration.paths
    installation.uninstall()
    assert not root.exists()
    assert all(not path.exists() for path in registration)
    assert {path: path.read_bytes() for path in before} == before
    assert (installation.profile.state / "application-data.json").is_file()


def test_existing_foreign_launcher_is_preserved(
    installation: DesktopInstallation, desktop_bundles: tuple[Path, Path]
) -> None:
    """Refuse an occupied user command before creating an active application version.

    Parameters
    ----------
    installation : DesktopInstallation
        Fresh profile with no installation ownership record.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Actual complete application to attempt to install.
    """
    path = installation.registration.paths[0]
    path.parent.mkdir(parents=True)
    original = b"user-owned command\n"
    path.write_bytes(original)
    with pytest.raises(DesktopStartRefused, match="existing launch file"):
        installation.install(desktop_bundles[0])
    assert path.read_bytes() == original
    assert not installation.root.exists()


def test_profile_lock_prevents_lifecycle_changes(
    installation: DesktopInstallation, desktop_bundles: tuple[Path, Path]
) -> None:
    """Respect real kernel profile-lock contention without signalling its owner.

    Parameters
    ----------
    installation : DesktopInstallation
        Actual installed profile in a disposable user namespace.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete real runtime versions.
    """
    first = installation.install(desktop_bundles[0])
    with (installation.profile.state / "backend.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(DesktopStartRefused, match="Close Studio"):
            installation.install(desktop_bundles[1])
        with pytest.raises(DesktopStartRefused, match="Close Studio"):
            installation.rollback(first.build_id)
        with pytest.raises(DesktopStartRefused, match="Close Studio"):
            installation.uninstall()
    assert os.readlink(installation.root / "current") == "versions/" + first.build_id


def test_uninstall_refuses_modified_application_files(
    installation: DesktopInstallation, desktop_bundles: tuple[Path, Path]
) -> None:
    """Preserve a changed installed resource and its launchers until the user resolves it.

    Parameters
    ----------
    installation : DesktopInstallation
        Real installed application owner.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Actual input runtime retained independently of installation.
    """
    installation.install(desktop_bundles[0])
    icon = installation.root / "current/share/icons/snn-studio.svg"
    original = icon.read_bytes()
    icon.write_bytes(b"user-modified resource")
    try:
        with pytest.raises(DesktopStartRefused, match="contents have changed"):
            installation.uninstall()
        assert icon.read_bytes() == b"user-modified resource"
        assert all(path.exists() for path in installation.registration.paths)
    finally:
        icon.write_bytes(original)


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        ("application-symlink", "application directory cannot be a symlink"),
        ("application-file", "record must be a regular file"),
        ("missing-owner", "record must be a regular file"),
        ("future-owner", "not a managed Studio installation"),
        ("owner-symlink", "record must be a regular file"),
        ("owner-fifo", "record must be a regular file"),
    ],
)
def test_uninstall_preserves_unowned_locations(
    installation_environment: dict[str, str],
    tmp_path: Path,
    layout: str,
    message: str,
) -> None:
    """Refuse actual foreign installation locations before opening or deleting data.

    Parameters
    ----------
    installation_environment : dict of str to str
        Private user namespace with no managed application installed.
    tmp_path : pathlib.Path
        Outside data that must remain unaffected by the lifecycle request.
    layout : str
        Actual unsupported filesystem ownership shape.
    message : str
        Deliberately authored refusal for this shape.
    """
    installation = DesktopInstallation(installation_environment)
    installation.profile.prepare()
    outside = tmp_path / "outside-research"
    outside.mkdir()
    user_file = outside / "retained.json"
    user_file.write_bytes(b'{"seed":"9223372036854775807"}')
    root = installation.root
    marker: Path | None = None
    if layout == "application-symlink":
        root.symlink_to(outside)
    elif layout == "application-file":
        root.write_bytes(b"user-owned file")
    else:
        root.mkdir()
        marker = root / "user-owned.json"
        marker.write_bytes(b"retained application-directory data")
        owner = root / "owner.json"
        if layout == "future-owner":
            owner.write_bytes(b'{"schema_version":"future"}')
        elif layout == "owner-symlink":
            owner.symlink_to(user_file)
        elif layout == "owner-fifo":
            os.mkfifo(owner)
    with pytest.raises(DesktopStartRefused, match=message):
        installation.uninstall()
    assert user_file.read_bytes() == b'{"seed":"9223372036854775807"}'
    if marker is not None:
        assert marker.read_bytes() == b"retained application-directory data"
    if layout == "application-symlink":
        assert root.is_symlink() and root.readlink() == outside
    elif layout == "application-file":
        assert root.read_bytes() == b"user-owned file"
    elif layout == "future-owner":
        assert (root / "owner.json").read_bytes() == b'{"schema_version":"future"}'
    elif layout == "owner-symlink":
        assert (root / "owner.json").is_symlink()
        assert (root / "owner.json").readlink() == user_file
    elif layout == "owner-fifo":
        assert stat.S_ISFIFO((root / "owner.json").lstat().st_mode)
