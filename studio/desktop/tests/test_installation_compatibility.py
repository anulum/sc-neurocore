# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop retained-data compatibility refusals

"""Keep actual research profiles and selected versions when migration is unqualified."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.bundle import verify_bundle
from sc_neurocore.studio.desktop.installation import DesktopInstallation

pytest_plugins = ["studio.desktop.tests.installation_fixtures"]


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (b"not JSON", "retained application data contract is invalid"),
        (b"[]", "separately qualified data migration"),
        (
            b'{"schema_version":"studio.desktop.installed-data.v2","compatibility_key":"future"}',
            "separately qualified data migration",
        ),
        (None, "retained application data contract is invalid"),
    ],
    ids=["malformed-json", "unexpected-type", "future-schema", "oversized-valid-json"],
)
def test_install_preserves_an_unsupported_retained_data_record(
    installation: DesktopInstallation,
    desktop_bundles: tuple[Path, Path],
    document: bytes | None,
    message: str,
) -> None:
    """Refuse malformed, oversized and future data before copying any application.

    Parameters
    ----------
    installation : DesktopInstallation
        Fresh private profile with retained data but no installed application.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete actual application whose data admission must refuse.
    document : bytes or None
        Unsupported persisted record, or an otherwise valid record padded past
        the storage limit when None.
    message : str
        Deliberately authored refusal for this persisted record.
    """
    installation.profile.prepare()
    if document is None:
        bundle = verify_bundle(desktop_bundles[0])
        document = (
            json.dumps(
                {
                    "schema_version": "studio.desktop.installed-data.v1",
                    "compatibility_key": bundle.compatibility_key,
                }
            )
            .encode()
            .ljust(8193, b" ")
        )
    stamp = installation.profile.state / "application-data.json"
    stamp.write_bytes(document)
    project = installation.profile.data / "retained-project.json"
    project.write_bytes(b'{"seed":"9223372036854775807"}')
    with pytest.raises(DesktopStartRefused, match=message):
        installation.install(desktop_bundles[0])
    assert stamp.read_bytes() == document
    assert project.read_bytes() == b'{"seed":"9223372036854775807"}'
    assert not installation.root.exists()
    assert all(not path.exists() for path in installation.registration.paths)


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_install_preserves_a_nonregular_retained_data_record(
    installation: DesktopInstallation,
    desktop_bundles: tuple[Path, Path],
    tmp_path: Path,
    kind: str,
) -> None:
    """Refuse real symlink and pipe records without reading or replacing their data.

    Parameters
    ----------
    installation : DesktopInstallation
        Private lifecycle controller with no installed code.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete runtime supplied to the refused public installation request.
    tmp_path : pathlib.Path
        Outside user data selected by the deliberately unsupported symlink.
    kind : str
        Actual non-regular filesystem entry placed at the data-record path.
    """
    installation.profile.prepare()
    outside = tmp_path / "outside-user-contract.json"
    original = b'{"owner":"retained outside data"}'
    outside.write_bytes(original)
    stamp = installation.profile.state / "application-data.json"
    if kind == "symlink":
        stamp.symlink_to(outside)
    else:
        os.mkfifo(stamp)
    message = "cannot be a symlink" if kind == "symlink" else "data contract is invalid"
    with pytest.raises(DesktopStartRefused, match=message):
        installation.install(desktop_bundles[0])
    assert outside.read_bytes() == original
    if kind == "symlink":
        assert stamp.is_symlink() and stamp.readlink() == outside
    else:
        assert stat.S_ISFIFO(stamp.lstat().st_mode)
    assert not installation.root.exists()
    assert all(not path.exists() for path in installation.registration.paths)


def test_install_keeps_a_future_desktop_configuration(
    installation: DesktopInstallation, desktop_bundles: tuple[Path, Path]
) -> None:
    """Reject a future profile through the lifecycle entry without starting a migration.

    Parameters
    ----------
    installation : DesktopInstallation
        Existing private profile owned by a newer configuration schema.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete actual application whose older profile contract must refuse.
    """
    installation.profile.prepare()
    configuration = installation.profile.config / "desktop.json"
    original = json.dumps(
        {
            "schema_version": "studio.desktop.configuration.v2",
            "port": 18031,
            "project_root": str(installation.profile.data / "projects"),
        }
    ).encode()
    configuration.write_bytes(original)
    with pytest.raises(DesktopStartRefused, match="configuration version is not supported"):
        installation.install(desktop_bundles[0])
    assert configuration.read_bytes() == original
    assert not installation.root.exists()
    assert not (installation.profile.state / "application-data.json").exists()


def test_upgrade_and_rollback_preserve_a_foreign_data_family(
    installation: DesktopInstallation, desktop_bundles: tuple[Path, Path]
) -> None:
    """Keep the real selected version and research when the retained data family differs.

    Parameters
    ----------
    installation : DesktopInstallation
        Actual independently copied application and retained user profile.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Two complete research runtime versions from one original data family.
    """
    first = installation.install(desktop_bundles[0])
    current = installation.root / "current"
    selected = current.readlink()
    launchers = {path: path.read_bytes() for path in installation.registration.paths}
    project = installation.profile.data / "retained-project.json"
    project.write_bytes(b'{"seed":"9223372036854775807"}')
    stamp = installation.profile.state / "application-data.json"
    original = stamp.read_bytes()
    other_key = ("0" if first.compatibility_key[0] != "0" else "1") * 64
    foreign = json.dumps(
        {"schema_version": "studio.desktop.installed-data.v1", "compatibility_key": other_key}
    ).encode()
    stamp.write_bytes(foreign)
    try:
        with pytest.raises(DesktopStartRefused, match="separately qualified data migration"):
            installation.rollback(first.build_id)
        with pytest.raises(DesktopStartRefused, match="separately qualified data migration"):
            installation.install(desktop_bundles[1])
        assert current.readlink() == selected
        assert {path.name for path in (installation.root / "versions").iterdir()} == {
            first.build_id
        }
        assert {path: path.read_bytes() for path in launchers} == launchers
        assert stamp.read_bytes() == foreign
        assert project.read_bytes() == b'{"seed":"9223372036854775807"}'
    finally:
        stamp.write_bytes(original)
