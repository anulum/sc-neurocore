# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Self-contained desktop bundle inventory

"""Bind a complete Linux desktop payload to its content and declared data compatibility."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.workspace_schema import write_atomic

BUNDLE_SCHEMA = "studio.desktop.bundle.v1"
_MAX_MANIFEST_BYTES = 16_777_216
_SITE = "runtime/lib/python3.12/site-packages/"
_REQUIRED = {
    "runtime/bin/python3.12",
    _SITE + "sc_neurocore/studio/desktop/__main__.py",
    _SITE + "sc_neurocore/studio/desktop/install.py",
    _SITE + "sc_neurocore/studio/frontend_dist/index.html",
    _SITE + "PySide6/Qt/libexec/QtWebEngineProcess",
    "share/icons/snn-studio.svg",
    "install.sh",
}


@dataclass(frozen=True, slots=True)
class BundleEntry:
    """One directory, regular file or internally confined symbolic link.

    Attributes
    ----------
    path : str
        Canonical relative POSIX path inside the bundle.
    kind : str
        ``directory``, ``file`` or ``symlink``.
    mode : int
        POSIX permission bits, including executable permissions.
    value : str
        File SHA-256, relative symlink target, or empty directory value.
    """

    path: str
    kind: str
    mode: int
    value: str


@dataclass(frozen=True, slots=True)
class BundleManifest:
    """A versioned payload with an explicit, qualified data-compatibility identity.

    Attributes
    ----------
    version : str
        Distribution version; independent of the embedded library version.
    compatibility_key : str
        SHA-256 identity of the qualified data contract. Equal keys permit
        activation and rollback; creating a manifest does not qualify a key.
    entries : tuple of BundleEntry
        Complete ordered payload inventory, excluding the manifest itself.
    """

    version: str
    compatibility_key: str
    entries: tuple[BundleEntry, ...]

    @property
    def build_id(self) -> str:
        """Return the SHA-256 identity of all manifest metadata and payload entries."""
        return hashlib.sha256(_encoded(self)).hexdigest()


def create_bundle_manifest(root: Path, *, version: str, compatibility_key: str) -> BundleManifest:
    """Inventory and atomically seal a real self-contained Linux desktop payload.

    Parameters
    ----------
    root : pathlib.Path
        Existing directory containing the full runtime, UI, bootstrap and icon.
    version : str
        Distribution version, optionally with a prerelease or build suffix.
    compatibility_key : str
        Qualified data-contract digest supplied by the bundle producer.

    Returns
    -------
    BundleManifest
        Exact content inventory also written to ``bundle.json``.

    Raises
    ------
    DesktopStartRefused
        Metadata, platform, required resources or filesystem entries are invalid.
    OSError
        Reading the payload or writing the manifest fails.

    Notes
    -----
    This checksum contract provides integrity, not publisher authentication,
    scientific acceptance or a claim that a compatibility key was qualified.
    """
    _metadata(version, compatibility_key)
    root = _root(root)
    manifest = BundleManifest(version, compatibility_key, _inventory(root))
    _required(manifest)
    data = json.loads(_encoded(manifest))
    data["build_id"] = manifest.build_id
    write_atomic(root / "bundle.json", json.dumps(data, sort_keys=True, indent=2) + "\n")
    return manifest


def verify_bundle(root: Path) -> BundleManifest:
    """Verify every payload entry before installation, activation or removal.

    Parameters
    ----------
    root : pathlib.Path
        Existing bundle directory, including its bounded ``bundle.json``.

    Returns
    -------
    BundleManifest
        Parsed metadata whose exact inventory matches the actual filesystem.

    Raises
    ------
    DesktopStartRefused
        The manifest or actual payload is missing, modified or unsupported.
    OSError
        A filesystem entry cannot be inspected.
    """
    root = _root(root)
    if (root / "bundle.json").is_symlink() or not (root / "bundle.json").is_file():
        raise DesktopStartRefused("The desktop bundle manifest must be a regular file.")
    with (root / "bundle.json").open("rb") as stream:
        raw = stream.read(_MAX_MANIFEST_BYTES + 1)
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise DesktopStartRefused("The desktop bundle manifest exceeds its size limit.")
    try:
        document: object = json.loads(raw)
    except (ValueError, UnicodeError):
        raise DesktopStartRefused("The desktop bundle manifest is not valid JSON.") from None
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "platform",
        "version",
        "compatibility_key",
        "entries",
        "build_id",
    }:
        raise DesktopStartRefused("The desktop bundle manifest has unsupported fields.")
    if document["schema_version"] != BUNDLE_SCHEMA or document["platform"] != "linux-x86_64":
        raise DesktopStartRefused("The desktop bundle schema or platform is not supported.")
    version, key, entries = document["version"], document["compatibility_key"], document["entries"]
    _metadata(version, key)
    if not isinstance(version, str) or not isinstance(key, str) or not isinstance(entries, list):
        raise DesktopStartRefused("The desktop bundle metadata is invalid.")
    parsed: list[BundleEntry] = []
    for row in entries:
        if not isinstance(row, dict) or set(row) != {"path", "kind", "mode", "value"}:
            raise DesktopStartRefused("The desktop bundle inventory is malformed.")
        path, kind, mode, value = row["path"], row["kind"], row["mode"], row["value"]
        if (
            not isinstance(path, str)
            or not isinstance(kind, str)
            or type(mode) is not int
            or not 0 <= mode <= 0o777
            or not isinstance(value, str)
            or kind not in {"directory", "file", "symlink"}
            or str(PurePosixPath(path)) != path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or path in {".", "bundle.json"}
        ):
            raise DesktopStartRefused("The desktop bundle inventory contains invalid entries.")
        parsed.append(BundleEntry(path, kind, mode, value))
    manifest = BundleManifest(version, key, tuple(parsed))
    if document["build_id"] != manifest.build_id:
        raise DesktopStartRefused("The desktop bundle manifest identity does not match.")
    _required(manifest)
    if manifest.entries != _inventory(root):
        raise DesktopStartRefused("The desktop bundle contents have changed.")
    return manifest


def _metadata(version: object, key: object) -> None:
    if (
        not isinstance(version, str)
        or len(version) > 80
        or re.fullmatch(r"\d+\.\d+\.\d+(?:[+-][A-Za-z0-9.-]+)?", version) is None
        or not isinstance(key, str)
        or re.fullmatch(r"[0-9a-f]{64}", key) is None
    ):
        raise DesktopStartRefused(
            "The desktop bundle version or compatibility identity is invalid."
        )


def _root(root: Path) -> Path:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise DesktopStartRefused("This desktop bundle requires Linux x86_64.")
    root = root.resolve()
    if not root.is_dir():
        raise DesktopStartRefused("The desktop bundle must be an existing directory.")
    return root


def _required(manifest: BundleManifest) -> None:
    files = {entry.path: entry for entry in manifest.entries if entry.kind == "file"}
    if not files.keys() >= _REQUIRED:
        raise DesktopStartRefused("The desktop bundle is missing its runtime, UI or launcher.")
    if any(
        not files[name].mode & 0o111
        for name in (
            "runtime/bin/python3.12",
            _SITE + "PySide6/Qt/libexec/QtWebEngineProcess",
            "install.sh",
        )
    ):
        raise DesktopStartRefused("The desktop bundle launcher is not executable.")


def _encoded(manifest: BundleManifest) -> bytes:
    return json.dumps(
        {
            "schema_version": BUNDLE_SCHEMA,
            "platform": "linux-x86_64",
            "version": manifest.version,
            "compatibility_key": manifest.compatibility_key,
            "entries": [
                {"path": item.path, "kind": item.kind, "mode": item.mode, "value": item.value}
                for item in manifest.entries
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _inventory(root: Path) -> tuple[BundleEntry, ...]:
    entries: list[BundleEntry] = []
    paths: list[Path] = []
    directories = [root]
    while directories:
        for path in directories.pop().iterdir():
            paths.append(path)
            if not path.is_symlink() and path.is_dir():
                directories.append(path)
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if relative == "bundle.json":
            continue
        mode = path.lstat().st_mode
        if stat.S_IMODE(mode) & 0o7000:
            raise DesktopStartRefused("Desktop bundles cannot contain special permission bits.")
        if stat.S_ISLNK(mode):
            target = os.readlink(path)
            try:
                confined = path.resolve(strict=True).is_relative_to(root)
            except (OSError, RuntimeError):
                raise DesktopStartRefused(
                    "Desktop bundle links must resolve to an internal entry."
                ) from None
            if Path(target).is_absolute() or not confined:
                raise DesktopStartRefused("Desktop bundle links must stay inside the bundle.")
            kind, value = "symlink", target
        elif stat.S_ISDIR(mode):
            kind, value = "directory", ""
        elif stat.S_ISREG(mode):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1_048_576):
                    digest.update(chunk)
            kind, value = "file", digest.hexdigest()
        else:
            raise DesktopStartRefused("Desktop bundles cannot contain special filesystem entries.")
        entries.append(BundleEntry(relative, kind, stat.S_IMODE(mode), value))
    return tuple(entries)
