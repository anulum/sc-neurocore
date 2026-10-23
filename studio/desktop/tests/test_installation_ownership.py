# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed application ownership and selector safety

"""Preserve real installed payloads and user files after ownership or layout changes."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.installation import DesktopInstallation

pytest_plugins = ["studio.desktop.tests.installation_fixtures"]


def test_uninstall_preserves_user_changes_to_managed_application_records(
    installation: DesktopInstallation,
    desktop_bundles: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Refuse changed records and selectors without removing research or launchers.

    Parameters
    ----------
    installation : DesktopInstallation
        Actual independently copied application and private research profile.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete wheel-backed runtimes used by normal installation.
    tmp_path : pathlib.Path
        Retained outside data referenced by unsupported application links.
    """
    manifest = installation.install(desktop_bundles[0])
    root = installation.root
    project = installation.profile.data / "retained-research.json"
    research = b'{"seed":"9223372036854775807"}'
    project.write_bytes(research)
    launchers = {path: path.read_bytes() for path in installation.registration.paths}
    for path in [root / "research-notes.json", root / "versions/user-record.json"]:
        contents = b'{"owner":"retained user record"}'
        path.write_bytes(contents)
        try:
            with pytest.raises(DesktopStartRefused, match="unowned"):
                installation.uninstall()
            assert path.read_bytes() == contents
            assert project.read_bytes() == research
            assert {name: name.read_bytes() for name in launchers} == launchers
        finally:
            path.unlink()

    owner = root / "owner.json"
    original_owner = owner.read_bytes()
    owner.write_bytes(original_owner.ljust(8193, b" "))
    try:
        with pytest.raises(DesktopStartRefused, match="record exceeds its size limit"):
            installation.uninstall()
        assert owner.read_bytes() == original_owner.ljust(8193, b" ")
        assert project.read_bytes() == research
    finally:
        owner.write_bytes(original_owner)

    stage = root / "versions/.stage-user-research"
    stage.mkdir()
    stage_data = stage / "retained.json"
    stage_data.write_bytes(research)
    stage_owner = stage / "owner.json"
    stage_owner.write_bytes(b'{"schema_version":"user-owned-stage"}')
    try:
        with pytest.raises(DesktopStartRefused, match="no Studio owner"):
            installation.uninstall()
        assert stage_data.read_bytes() == research
    finally:
        stage_data.unlink()
        stage_owner.unlink()
        stage.rmdir()

    outside = tmp_path / "outside-research.json"
    outside.write_bytes(research)
    pending = root / ".next"
    pending.write_bytes(research)
    try:
        with pytest.raises(DesktopStartRefused, match="unowned file"):
            installation.uninstall()
        assert pending.read_bytes() == research
    finally:
        pending.unlink()
    pending.symlink_to(outside)
    try:
        with pytest.raises(DesktopStartRefused, match="not owned by Studio"):
            installation.uninstall()
        assert pending.is_symlink() and pending.readlink() == outside
        assert outside.read_bytes() == research
    finally:
        pending.unlink()

    current = root / "current"
    selected = current.readlink()
    current.unlink()
    current.write_bytes(research)
    try:
        with pytest.raises(DesktopStartRefused, match="not a managed link"):
            installation.uninstall()
        assert current.read_bytes() == research
    finally:
        current.unlink()
        current.symlink_to(selected)
    current.unlink()
    current.symlink_to(outside)
    try:
        with pytest.raises(DesktopStartRefused, match="leaves its version directory"):
            installation.uninstall()
        assert current.is_symlink() and current.readlink() == outside
        assert outside.read_bytes() == research
    finally:
        current.unlink()
        current.symlink_to(selected)

    pending.symlink_to(selected)
    assert installation.install(desktop_bundles[0]) == manifest
    assert not pending.exists() and not pending.is_symlink()
    assert current.readlink() == selected
    assert project.read_bytes() == research
    assert {name: name.read_bytes() for name in launchers} == launchers


def test_rollback_preserves_a_manually_relocated_version_store(
    installation: DesktopInstallation,
    desktop_bundles: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Refuse missing, regular-file and linked version stores without touching their data.

    Parameters
    ----------
    installation : DesktopInstallation
        Real installed application whose version store is deliberately relocated.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Full runtime installed before changing the persistent directory layout.
    tmp_path : pathlib.Path
        Independent location retaining the actual complete version store.
    """
    manifest = installation.install(desktop_bundles[0])
    versions = installation.root / "versions"
    retained = tmp_path / "retained-version-store"
    current = installation.root / "current"
    selected = current.readlink()
    current.unlink()
    versions.rename(retained)
    seal = retained / manifest.build_id / "bundle.json"
    original_seal = seal.read_bytes()
    try:
        with pytest.raises(DesktopStartRefused, match="not installed"):
            installation.rollback(manifest.build_id)
        assert not versions.exists()
        for kind in ["file", "symlink"]:
            if kind == "file":
                versions.write_bytes(b"retained user-selected location")
            else:
                versions.symlink_to(retained, target_is_directory=True)
            try:
                with pytest.raises(DesktopStartRefused, match="version directory"):
                    installation.rollback(manifest.build_id)
                assert seal.read_bytes() == original_seal
                if kind == "file":
                    assert versions.read_bytes() == b"retained user-selected location"
                else:
                    assert versions.is_symlink() and versions.readlink() == retained
            finally:
                versions.unlink()
    finally:
        retained.rename(versions)
        current.symlink_to(selected)
    assert installation.rollback(manifest.build_id) == manifest


def test_install_admits_only_its_exact_retained_bundle_path(
    installation: DesktopInstallation,
    installation_environment: dict[str, str],
    desktop_bundles: tuple[Path, Path],
) -> None:
    """Permit exact retained reinstallation and refuse overlapping application roots.

    Parameters
    ----------
    installation : DesktopInstallation
        Actual independently copied and selected application.
    installation_environment : dict of str to str
        Private profile used to demonstrate an unsafe application location.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete source bundle whose resources must remain outside installation data.
    """
    manifest = installation.install(desktop_bundles[0])
    current = installation.root / "current"
    assert installation.install(current) == manifest
    target = installation.root / "versions" / manifest.build_id
    misplaced = installation.root / "borrowed-payload"
    target.rename(misplaced)
    original = (misplaced / "bundle.json").read_bytes()
    try:
        with pytest.raises(DesktopStartRefused, match="separate directories"):
            installation.install(misplaced)
        assert (misplaced / "bundle.json").read_bytes() == original
    finally:
        misplaced.rename(target)

    source = desktop_bundles[0]
    data = source / "misplaced-user-data"
    overlapping = DesktopInstallation({**installation_environment, "XDG_DATA_HOME": str(data)})
    with pytest.raises(DesktopStartRefused, match="separate directories"):
        overlapping.install(source)
    assert not data.exists()
    assert installation.rollback(manifest.build_id) == manifest
