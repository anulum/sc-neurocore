# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lossless retention of complete native proof artifacts

"""Retain one lossless representation and restore bounded native artifact views."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import stat
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import BinaryIO

from tools.function_synthesis_json import decode_native_json

_MANIFEST = "retained-artifacts.json"
_STORE = ".retained-artifacts"
_METADATA = frozenset({"binding.json", "measurement.json", "execution.txt"})
_SCHEMA = "sc-neurocore.lossless-proof-artifacts.v1"
_MAX_BYTES = 1024**3
_MAX_ENTRIES = 50_000
_MAX_MANIFEST = 16 * 1024**2


def _digest(path: Path) -> str:
    """Hash a regular artifact through its complete byte stream."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Retained proof data must be regular unlinked files")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inventory(directory: Path) -> set[str]:
    """List every regular artifact while refusing linked or special entries."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Retained proof data requires a regular directory")
    files = set()
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Retained proof data contains symbolic links or special entries")
        if path.is_file():
            files.add(path.relative_to(directory).as_posix())
    return files


@dataclass(frozen=True, slots=True)
class _Artifact:
    """Bind the original bytes and mode to one stored lossless representation."""

    bytes: int
    sha256: str
    mode: int
    stored_bytes: int
    stored_sha256: str


def _record(value: object) -> _Artifact:
    """Require finite byte counts, ordinary mode bits and exact SHA-256 fields."""
    if not isinstance(value, dict) or set(value) != {
        "bytes",
        "sha256",
        "mode",
        "stored_bytes",
        "stored_sha256",
    }:
        raise ValueError("Retained proof artifact record is malformed")
    size, stored, mode = value["bytes"], value["stored_bytes"], value["mode"]
    original, encoded = value["sha256"], value["stored_sha256"]
    if (
        type(size) is not int
        or not 0 <= size <= _MAX_BYTES
        or type(stored) is not int
        or not 0 < stored <= _MAX_BYTES
        or type(mode) is not int
        or not 0 <= mode <= 0o777
        or not isinstance(original, str)
        or re.fullmatch(r"[0-9a-f]{64}", original) is None
        or not isinstance(encoded, str)
        or re.fullmatch(r"[0-9a-f]{64}", encoded) is None
    ):
        raise ValueError("Retained proof artifact identity is invalid")
    return _Artifact(size, original, mode, stored, encoded)


def _read_manifest(directory: Path) -> tuple[dict[str, _Artifact], dict[str, str]]:
    """Validate the complete stored layout before restoring any native file."""
    path = directory / _MANIFEST
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_MANIFEST:
        raise ValueError("Retained proof manifest exceeds its finite budget")
    value = decode_native_json(path.read_bytes())
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "metadata_sha256", "artifacts"}
        or value["schema_version"] != _SCHEMA
        or not isinstance(value["metadata_sha256"], dict)
        or set(value["metadata_sha256"]) != _METADATA
        or not isinstance(value["artifacts"], dict)
        or not 0 < len(value["artifacts"]) <= _MAX_ENTRIES
    ):
        raise ValueError("Retained proof manifest is malformed")
    metadata: dict[str, str] = {}
    for name, identity in value["metadata_sha256"].items():
        if identity != _digest(directory / name):
            raise ValueError("Retained proof metadata changed")
        metadata[name] = identity
    records = {}
    for name, record in value["artifacts"].items():
        relative = PurePosixPath(name)
        if (
            not name
            or relative.is_absolute()
            or relative.as_posix() != name
            or any(part in {".", ".."} for part in relative.parts)
            or name in _METADATA | {_MANIFEST}
            or relative.parts[0] == _STORE
        ):
            raise ValueError("Retained proof path is not a canonical local artifact")
        records[name] = _record(record)
    if (
        sum(record.bytes for record in records.values()) > _MAX_BYTES
        or sum(record.stored_bytes for record in records.values()) > _MAX_BYTES
    ):
        raise ValueError("Retained proof data exceeds the finite restoration budget")
    expected = _METADATA | {_MANIFEST} | {f"{_STORE}/{name}.gz" for name in records}
    if _inventory(directory) != expected:
        raise ValueError("Retained proof layout has missing or unexpected files")
    return records, metadata


def _restore(path: Path, record: _Artifact, destination: BinaryIO | None) -> None:
    """Stream a verified original without exceeding its declared restoration size."""
    if path.stat().st_size != record.stored_bytes or _digest(path) != record.stored_sha256:
        raise ValueError("Stored proof artifact changed")
    size = 0
    identity = hashlib.sha256()
    try:
        with gzip.open(path, "rb") as stream:
            while chunk := stream.read(65536):
                size += len(chunk)
                if size > record.bytes:
                    raise ValueError("Stored proof artifact exceeds its declared original size")
                identity.update(chunk)
                if destination is not None:
                    destination.write(chunk)
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError("Stored proof artifact cannot be restored") from error
    if (
        size != record.bytes
        or identity.hexdigest() != record.sha256
        or _digest(path) != record.stored_sha256
    ):
        raise ValueError("Restored proof artifact differs from its original bytes")


def retain_artifacts(directory: Path, *, byte_budget: int = _MAX_BYTES) -> None:
    """Store every qualified native artifact losslessly without retaining duplicate trees.

    Parameters
    ----------
    directory : pathlib.Path
        Completed native acquisition with binding, measurement and timestamp.
        Call the normal native proof reader before and after retention.
    byte_budget : int, optional
        Positive maximum total original native artifact bytes, at most 1 GiB.
        A caller may lower this limit to reserve space for other acquisitions;
        the default and the retained format's absolute limit remain 1 GiB.

    Raises
    ------
    OSError, ValueError
        For linked, changing, already-retained or oversized data. Failed packing
        remains an unpublished attempt; original bytes survive in plain or
        verified lossless form, and are never removed before round-trip checks.

    """
    if type(byte_budget) is not int or not 0 < byte_budget <= _MAX_BYTES:
        raise ValueError(
            "Native retention requires an integer byte budget between one byte and 1 GiB"
        )
    files = _inventory(directory)
    if (
        not files.issuperset(_METADATA)
        or (directory / _MANIFEST).exists()
        or (directory / _STORE).exists()
    ):
        raise ValueError("Retaining proof artifacts requires a fresh complete acquisition")
    names = sorted(files - _METADATA)
    if not 0 < len(names) <= _MAX_ENTRIES:
        raise ValueError("Retained proof artifact count is outside its finite budget")
    if sum((directory / name).stat().st_size for name in names) > byte_budget:
        raise ValueError("Retained proof data exceeds the finite restoration budget")
    metadata = {name: _digest(directory / name) for name in sorted(_METADATA)}
    records = {}
    for name in names:
        source = directory / name
        size, identity = source.stat().st_size, _digest(source)
        mode = stat.S_IMODE(source.stat().st_mode)
        target = directory / _STORE / (name + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        with (
            source.open("rb") as original,
            target.open("xb") as stored,
            gzip.GzipFile(filename="", fileobj=stored, mode="wb", mtime=0) as encoded,
        ):
            shutil.copyfileobj(original, encoded, length=65536)
        record = _record(
            {
                "bytes": size,
                "sha256": identity,
                "mode": mode,
                "stored_bytes": target.stat().st_size,
                "stored_sha256": _digest(target),
            }
        )
        _restore(target, record, None)
        if _digest(source) != identity:
            raise ValueError("Native proof artifact changed during retention")
        records[name] = asdict(record)
    if metadata != {name: _digest(directory / name) for name in _METADATA}:
        raise ValueError("Native proof metadata changed during retention")
    manifest = {"schema_version": _SCHEMA, "metadata_sha256": metadata, "artifacts": records}
    with (directory / _MANIFEST).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    for name in names:
        (directory / name).unlink()


@contextmanager
def materialize_artifacts(directory: Path) -> Iterator[Path]:
    """Restore a temporary exact view for the unchanged native proof validators.

    Parameters
    ----------
    directory : pathlib.Path
        Retained acquisition, or a legacy regular acquisition without storage
        metadata. Missing or extra stored artifacts are refused.

    Yields
    ------
    pathlib.Path
        Complete original filenames and bytes, including proof records and
        acquisition time. A restored temporary view is removed on exit.

    Raises
    ------
    OSError, ValueError
        For corrupt storage, unqualified paths, changed metadata or bounded-out
        restoration. This format validates custody; the caller must still run
        all source, toolchain, geometry and complete native proof checks.

    """
    _inventory(directory)
    if not (directory / _MANIFEST).exists():
        if (directory / _STORE).exists():
            raise ValueError("Retained proof storage is incomplete")
        yield directory
        return
    records, metadata = _read_manifest(directory)
    with TemporaryDirectory(prefix=".function-proof-view-", dir=directory.parent) as temporary:
        view = Path(temporary)
        for name in _METADATA:
            shutil.copy2(directory / name, view / name)
        for name, record in records.items():
            target = view / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                _restore(directory / _STORE / (name + ".gz"), record, stream)
            target.chmod(record.mode)
        if metadata != {name: _digest(view / name) for name in _METADATA}:
            raise ValueError("Restored proof metadata differs from its acquisition")
        if _read_manifest(directory) != (records, metadata):
            raise ValueError("Retained proof storage changed during restoration")
        yield view
