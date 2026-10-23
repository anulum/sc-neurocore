# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop payload filesystem admission

"""Refuse incomplete inventories and unsupported permissions in real runtime payloads."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.bundle import create_bundle_manifest, verify_bundle
from sc_neurocore.studio.workspace_schema import write_atomic

pytest_plugins = ["studio.desktop.tests.installation_fixtures"]


def test_sealing_cannot_omit_an_unreadable_payload_directory(
    desktop_bundles: tuple[Path, Path],
) -> None:
    """Keep the prior seal when real runtime resources cannot be enumerated.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete application whose resource directory becomes unreadable.
    """
    root = desktop_bundles[0]
    manifest = verify_bundle(root)
    original = (root / "bundle.json").read_text()
    directory = root / "runtime/share/retained-protected-resources"
    directory.mkdir(parents=True)
    resource = directory / "experiment.json"
    contents = b'{"seed":"9223372036854775807"}'
    resource.write_bytes(contents)
    directory.chmod(0)
    try:
        with pytest.raises(PermissionError):
            create_bundle_manifest(
                root, version=manifest.version, compatibility_key=manifest.compatibility_key
            )
        assert (root / "bundle.json").read_text() == original
    finally:
        directory.chmod(0o755)
        assert resource.read_bytes() == contents
        resource.unlink()
        directory.rmdir()
        write_atomic(root / "bundle.json", original)


@pytest.mark.parametrize("mode", [0o4755, 0o2755, 0o1755], ids=["setuid", "setgid", "sticky"])
def test_sealing_refuses_special_permission_bits(
    desktop_bundles: tuple[Path, Path], mode: int
) -> None:
    """Refuse permissions which a verified portable payload cannot represent.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Full runtime whose original seal remains unchanged by the refusal.
    mode : int
        Real filesystem mode carrying one unsupported special permission bit.
    """
    root = desktop_bundles[0]
    manifest = verify_bundle(root)
    original = (root / "bundle.json").read_text()
    resource = root / "runtime/share/retained-permission-resource.json"
    contents = b'{"owner":"retained research data"}'
    resource.write_bytes(contents)
    resource.chmod(mode)
    try:
        with pytest.raises(DesktopStartRefused, match="special permission bits"):
            create_bundle_manifest(
                root, version=manifest.version, compatibility_key=manifest.compatibility_key
            )
        assert resource.read_bytes() == contents
        assert stat.S_IMODE(resource.stat().st_mode) == mode
        assert (root / "bundle.json").read_text() == original
    finally:
        resource.unlink()
        write_atomic(root / "bundle.json", original)
