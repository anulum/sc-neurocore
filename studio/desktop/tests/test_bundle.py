# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual desktop bundle integrity integration

"""Verify and refuse complete real-runtime payloads through their public manifest API."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.bundle import create_bundle_manifest, verify_bundle
from sc_neurocore.studio.workspace_schema import write_atomic

pytest_plugins = ["studio.desktop.tests.installation_fixtures"]


def test_real_bundle_sealing_is_reproducible(desktop_bundles: tuple[Path, Path]) -> None:
    """Reseal the actual complete payload without changing its content identity.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Two real self-contained desktop runtime versions.
    """
    root = desktop_bundles[0]
    original = verify_bundle(root)
    recreated = create_bundle_manifest(
        root, version=original.version, compatibility_key=original.compatibility_key
    )
    assert original == recreated == verify_bundle(root)
    assert len(original.entries) > 0
    assert original.build_id != verify_bundle(desktop_bundles[1]).build_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "future"),
        ("platform", "different"),
        ("version", None),
        ("version", "invalid"),
        ("compatibility_key", "invalid"),
        ("entries", None),
        ("entries", [None]),
        ("entries", [{"path": "install.sh", "kind": "file", "mode": True, "value": ""}]),
        ("entries", [{"path": "../outside"}]),
        ("build_id", "0" * 64),
    ],
)
def test_modified_manifest_is_refused_without_rewriting_it(
    desktop_bundles: tuple[Path, Path], field: str, value: object
) -> None:
    """Reject malformed and future metadata before treating it as an installed payload.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Actual runtime whose original manifest is restored after the case.
    field : str
        Manifest property changed by an external writer.
    value : object
        Invalid or unsupported JSON property value.
    """
    root = desktop_bundles[0]
    path = root / "bundle.json"
    original = path.read_text()
    data = json.loads(original)
    data[field] = value
    changed = json.dumps(data)
    write_atomic(path, changed)
    try:
        with pytest.raises(DesktopStartRefused):
            verify_bundle(root)
        assert path.read_text() == changed
    finally:
        write_atomic(path, original)


def test_extra_user_file_is_not_silently_adopted(desktop_bundles: tuple[Path, Path]) -> None:
    """Keep an unexpected real file intact while refusing the changed payload.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete real application input retained after the case.
    """
    root = desktop_bundles[0]
    foreign = root / "user-experiment.json"
    original = b'{"seed":"9223372036854775807"}'
    foreign.write_bytes(original)
    try:
        with pytest.raises(DesktopStartRefused, match="contents have changed"):
            verify_bundle(root)
        assert foreign.read_bytes() == original
    finally:
        foreign.unlink()


def test_external_link_and_special_file_are_refused(
    desktop_bundles: tuple[Path, Path], tmp_path: Path
) -> None:
    """Refuse actual escaping symlinks and FIFOs without following or opening them.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete real runtime bundle whose data remains untouched.
    tmp_path : pathlib.Path
        External directory that must never become part of the payload.
    """
    root = desktop_bundles[0]
    link = root / "outside-link"
    link.symlink_to(tmp_path)
    try:
        with pytest.raises(DesktopStartRefused, match="links must stay inside"):
            verify_bundle(root)
    finally:
        link.unlink()
    fifo = root / "unexpected.pipe"
    os.mkfifo(fifo)
    try:
        with pytest.raises(DesktopStartRefused, match="special filesystem"):
            verify_bundle(root)
    finally:
        fifo.unlink()


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("not JSON", "not valid JSON"),
        ("[]", "unsupported fields"),
        (None, "exceeds its size limit"),
    ],
    ids=["invalid-json", "wrong-document-type", "oversized-valid-json"],
)
def test_unreadable_manifest_preserves_the_original_payload(
    desktop_bundles: tuple[Path, Path], document: str | None, message: str
) -> None:
    """Refuse an unusable manifest without rewriting it or its complete runtime.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Full runtime whose sealed manifest is restored after each public request.
    document : str or None
        Invalid content, or a valid original document padded past the size limit.
    message : str
        Expected authored refusal for the actual persisted input.
    """
    root = desktop_bundles[0]
    path = root / "bundle.json"
    original = path.read_text()
    changed = original.ljust(16_777_217) if document is None else document
    write_atomic(path, changed)
    try:
        with pytest.raises(DesktopStartRefused, match=message):
            verify_bundle(root)
        assert path.read_text() == changed
        assert (root / "runtime/bin/python3.12").is_file()
    finally:
        write_atomic(path, original)


@pytest.mark.parametrize("kind", ["missing", "directory", "fifo", "symlink"])
def test_nonregular_manifest_is_preserved(desktop_bundles: tuple[Path, Path], kind: str) -> None:
    """Keep an unsupported manifest entry intact instead of opening or replacing it.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete runtime retained while its manifest path is temporarily unavailable.
    kind : str
        Real filesystem layout encountered in place of the sealed manifest.
    """
    root = desktop_bundles[0]
    path = root / "bundle.json"
    retained = root.parent / (root.name + ".retained-manifest")
    assert not retained.exists()
    original = path.read_bytes()
    path.rename(retained)
    try:
        if kind == "directory":
            path.mkdir()
        elif kind == "fifo":
            os.mkfifo(path)
        elif kind == "symlink":
            path.symlink_to(retained)
        with pytest.raises(DesktopStartRefused, match="manifest must be a regular file"):
            verify_bundle(root)
        if kind == "directory":
            assert path.is_dir()
        elif kind == "fifo":
            assert stat.S_ISFIFO(path.lstat().st_mode)
        elif kind == "symlink":
            assert path.is_symlink() and path.readlink() == retained
        else:
            assert not path.exists()
        assert retained.read_bytes() == original
    finally:
        if path.is_dir() and not path.is_symlink():
            path.rmdir()
        else:
            path.unlink(missing_ok=True)
        retained.rename(path)


@pytest.mark.parametrize("layout", ["missing", "not-executable"])
def test_sealing_refuses_an_unlaunchable_complete_runtime(
    desktop_bundles: tuple[Path, Path], layout: str
) -> None:
    """Refuse an extraction with a missing or non-executable installation bootstrap.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Full real runtime retaining every scientific, UI and native resource.
    layout : str
        Missing launcher or permissions lost during package extraction.
    """
    root = desktop_bundles[0]
    manifest_path = root / "bundle.json"
    original = manifest_path.read_bytes()
    metadata = json.loads(original)
    launcher = root / "install.sh"
    retained = root.parent / (root.name + ".retained-bootstrap")
    assert not retained.exists()
    identity = launcher.stat()
    contents = launcher.read_bytes()
    launcher.rename(retained)
    try:
        if layout == "not-executable":
            launcher.write_bytes(contents)
            launcher.chmod(0o644)
        message = "missing its runtime, UI or launcher" if layout == "missing" else "not executable"
        with pytest.raises(DesktopStartRefused, match=message):
            create_bundle_manifest(
                root, version=metadata["version"], compatibility_key=metadata["compatibility_key"]
            )
        assert manifest_path.read_bytes() == original
        assert retained.read_bytes() == contents
        assert retained.stat().st_ino == identity.st_ino
        assert stat.S_IMODE(retained.stat().st_mode) == stat.S_IMODE(identity.st_mode)
    finally:
        launcher.unlink(missing_ok=True)
        retained.rename(launcher)


@pytest.mark.parametrize("kind", ["dangling", "cycle"])
def test_broken_internal_link_is_refused_without_following_it(
    desktop_bundles: tuple[Path, Path], kind: str
) -> None:
    """Refuse real incomplete or cyclic extraction links while retaining their bytes.

    Parameters
    ----------
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete research runtime whose original manifest remains untouched.
    kind : str
        Missing internal target or a symbolic link back to itself.
    """
    root = desktop_bundles[0]
    link = root / "0-incomplete-runtime-link"
    target = "missing-runtime-file" if kind == "dangling" else link.name
    original = (root / "bundle.json").read_bytes()
    link.symlink_to(target)
    try:
        with pytest.raises(DesktopStartRefused, match="links must resolve to an internal entry"):
            verify_bundle(root)
        assert link.is_symlink() and os.readlink(link) == target
        assert (root / "bundle.json").read_bytes() == original
    finally:
        link.unlink()


def test_missing_bundle_directory_is_an_authored_refusal(tmp_path: Path) -> None:
    """Refuse a user's nonexistent package path without creating a substitute.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Existing user-selected parent containing no extracted bundle.
    """
    missing = tmp_path / "missing-extracted-bundle"
    with pytest.raises(DesktopStartRefused, match="must be an existing directory"):
        verify_bundle(missing)
    assert not missing.exists()
