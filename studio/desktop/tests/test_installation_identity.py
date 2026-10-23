# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained application identity admission

"""Refuse relocated or resealed installed code while preserving the user's profile."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.bundle import create_bundle_manifest
from sc_neurocore.studio.desktop.installation import DesktopInstallation
from sc_neurocore.studio.workspace_schema import write_atomic

pytest_plugins = ["studio.desktop.tests.installation_fixtures"]


def test_absent_application_refuses_removal_and_unknown_rollback(
    installation: DesktopInstallation,
) -> None:
    """Keep absent and empty application locations without inventing a version.

    Parameters
    ----------
    installation : DesktopInstallation
        Fresh real user profile with no installed application.
    """
    for build_id in ["short", "g" * 64]:
        with pytest.raises(DesktopStartRefused, match="complete installed build identity"):
            installation.rollback(build_id)
    operations: tuple[Callable[[], object], ...] = (
        installation.uninstall,
        lambda: installation.rollback("0" * 64),
    )
    for operation in operations:
        with pytest.raises(DesktopStartRefused, match="not installed for this profile"):
            operation()
    assert not installation.root.exists()
    installation.root.mkdir()
    try:
        with pytest.raises(DesktopStartRefused, match="not installed for this profile"):
            installation.uninstall()
        assert list(installation.root.iterdir()) == []
    finally:
        installation.root.rmdir()
    assert all(not path.exists() for path in installation.registration.paths)


def test_resealed_installed_version_preserves_its_original_selection_and_research(
    installation: DesktopInstallation,
    desktop_bundles: tuple[Path, Path],
) -> None:
    """Refuse valid payload metadata stored under a different installed identity.

    Parameters
    ----------
    installation : DesktopInstallation
        Complete real installation whose owned manifest is deliberately resealed.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Actual wheel-backed runtime used for the initial and refused installations.
    """
    manifest = installation.install(desktop_bundles[0])
    root = installation.root
    current = root / "current"
    selected = current.readlink()
    target = root / "versions" / manifest.build_id
    seal = target / "bundle.json"
    original = seal.read_text()
    project = installation.profile.data / "retained.json"
    research = b'{"seed":"9223372036854775807"}'
    project.write_bytes(research)
    launchers = {path: path.read_bytes() for path in installation.registration.paths}
    changed = create_bundle_manifest(
        target, version="3.16.0+deliberately.resealed", compatibility_key=manifest.compatibility_key
    )
    assert changed.build_id != manifest.build_id
    try:
        with pytest.raises(DesktopStartRefused, match="selected Studio version identity"):
            installation.install(desktop_bundles[0])
        assert current.readlink() == selected
        current.unlink()
        with pytest.raises(DesktopStartRefused, match="installed bundle identity does not match"):
            installation.install(desktop_bundles[0])
        with pytest.raises(DesktopStartRefused, match="retained Studio version identity"):
            installation.rollback(manifest.build_id)
        with pytest.raises(DesktopStartRefused, match="unexpected identity"):
            installation.uninstall()
        missing = ("0" if manifest.build_id[0] != "0" else "1") * 64
        with pytest.raises(DesktopStartRefused, match="That Studio version is not installed"):
            installation.rollback(missing)
        assert project.read_bytes() == research
        assert {path: path.read_bytes() for path in launchers} == launchers
        assert not current.exists() and not current.is_symlink()
    finally:
        write_atomic(seal, original)
        if not current.is_symlink():
            current.symlink_to(selected)
    assert installation.rollback(manifest.build_id) == manifest
    assert project.read_bytes() == research


def test_missing_data_stamp_does_not_admit_a_different_selected_runtime_family(
    installation: DesktopInstallation,
    desktop_bundles: tuple[Path, Path],
) -> None:
    """Use the actual selected payload's contract when its separate stamp is missing.

    Parameters
    ----------
    installation : DesktopInstallation
        Real installed application with retained data and command launchers.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete actual payload offered as an incompatible upgrade.
    """
    first = installation.install(desktop_bundles[0])
    root = installation.root
    current = root / "current"
    selected = current.readlink()
    original_target = root / "versions" / first.build_id
    original_seal = (original_target / "bundle.json").read_text()
    stamp = installation.profile.state / "application-data.json"
    original_stamp = stamp.read_text()
    project = installation.profile.data / "retained.json"
    research = b'{"seed":"9223372036854775807"}'
    project.write_bytes(research)
    launchers = {path: path.read_bytes() for path in installation.registration.paths}
    other_key = ("0" if first.compatibility_key[0] != "0" else "1") * 64
    changed = create_bundle_manifest(
        original_target, version=first.version, compatibility_key=other_key
    )
    moved = root / "versions" / changed.build_id
    original_target.rename(moved)
    current.unlink()
    current.symlink_to(Path("versions") / changed.build_id)
    stamp.unlink()
    try:
        with pytest.raises(DesktopStartRefused, match="does not share the installed data contract"):
            installation.install(desktop_bundles[1])
        assert current.readlink() == Path("versions") / changed.build_id
        assert not stamp.exists()
        assert project.read_bytes() == research
        assert {path: path.read_bytes() for path in launchers} == launchers
        assert list((root / "versions").iterdir()) == [moved]
    finally:
        write_atomic(moved / "bundle.json", original_seal)
        moved.rename(original_target)
        current.unlink()
        current.symlink_to(selected)
        write_atomic(stamp, original_stamp)
    assert installation.rollback(first.build_id) == first
    assert project.read_bytes() == research
