# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual native image and counter-input validation

"""Keep real native file reads and explicit observation validation separate from runtime hooks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class Image:
    """Keep one explicitly admitted backing image and its diagnostic role."""

    path: Path
    sha256: str
    backend: str
    model: str

    def verify_backing(self, identity: tuple[int, int]) -> None:
        """Read the real backing file again before any snapshot or mapping acceptance."""
        info = self.path.stat()
        if (info.st_dev, info.st_ino) != identity:
            raise RuntimeError("Image backing identity changed before snapshot")
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise RuntimeError("Image bytes changed before snapshot")


def read_images(path: Path, root: Path) -> dict[Path, Image]:
    """Refuse incomplete, foreign or ambiguous image declarations before any load."""
    raw: object = json.loads(path.read_text())
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("Exactly four actual family images are required")
    result: dict[Path, Image] = {}
    roles: set[tuple[str, str]] = set()
    for untyped in raw:
        if not isinstance(untyped, dict):
            raise ValueError("Image entry must be a mapping")
        row = cast(dict[str, object], untyped)
        if set(row) != {"path", "sha256", "backend", "model"}:
            raise ValueError("Unexpected image declaration fields")
        filename, expected, backend, model = (
            row["path"],
            row["sha256"],
            row["backend"],
            row["model"],
        )
        if not all(isinstance(value, str) for value in (filename, expected, backend, model)):
            raise ValueError("Image declarations require strings")
        filename, expected, backend, model = cast(
            tuple[str, str, str, str], (filename, expected, backend, model)
        )
        source = Path(filename)
        if not source.is_absolute() or source.is_symlink():
            raise ValueError("Declare the absolute actual image, not an alias")
        source = source.resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError("Image is outside the explicit gate allocation")
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError("Image digest is not a lowercase SHA256")
        if backend not in {"go", "mojo"} or model not in {"mat", "sc_resetting_mat"}:
            raise ValueError("Unknown diagnostic image role")
        if source in result or (backend, model) in roles:
            raise ValueError("Duplicate image or role")
        roles.add((backend, model))
        result[source] = Image(source, expected, backend, model)
    return result


def validate_audited_image(
    image: Image,
    before: os.stat_result,
    after: os.stat_result,
    digest: str,
    loaded_identity: tuple[int, int] | None,
) -> tuple[int, int]:
    """Conserve actual descriptor metadata, full digest and any earlier observed backing inode."""
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    changed = any(getattr(before, field) != getattr(after, field) for field in identity_fields)
    if digest != image.sha256 or changed:
        raise RuntimeError("Diagnostic image bytes or metadata changed before load")
    identity = (before.st_dev, before.st_ino)
    if loaded_identity is not None and loaded_identity != identity:
        raise RuntimeError("Diagnostic image backing identity changed")
    return identity


def executable_image_maps(identity: tuple[int, int], maps: str) -> list[str]:
    """Require an executable mapping of the actual backing device and inode in raw map data."""
    device, inode = identity
    retained = []
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        major, minor = fields[3].split(":")
        if (
            int(fields[4]) == inode
            and (int(major, 16), int(minor, 16)) == (os.major(device), os.minor(device))
            and "x" in fields[1]
        ):
            retained.append(line)
    if not retained:
        raise RuntimeError("Loaded image lacks its expected executable mapping")
    return retained


def validate_invalid_snapshot_status(status: int) -> None:
    """Require the actual native writer to refuse its deliberately invalid destination."""
    if status != 1:
        raise RuntimeError("Invalid native snapshot path was accepted")


def validate_snapshot_output(destination: Path, status: int) -> None:
    """Read real snapshot metadata and counter files only after an actual successful status."""
    if status != 0:
        raise RuntimeError("Native Go counter snapshot refused")
    if not list(destination.glob("covmeta.*")) or not list(destination.glob("covcounters.*")):
        raise RuntimeError("Native snapshot omitted metadata or counters")


def original_abi_launch_arguments(arguments: tuple[object, ...], phase: str) -> list[str] | None:
    """Validate actual audit argument geometry before retaining an original ABI child launch."""
    if len(arguments) < 2 or not isinstance(arguments[1], list):
        return None
    argv: object = arguments[1]
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        return None
    strings = cast(list[str], argv)
    if len(strings) < 4 or strings[1:3] != ["-B", "-c"]:
        return None
    if "from tests.test_mat_family_c_abi import" not in strings[3]:
        return None
    if phase != "tests" or len(arguments) != 4 or arguments[3] is not None:
        raise RuntimeError("Original ABI launch must inherit the declared test environment")
    return strings


def validate_existing_observer(
    actual_job: Path, actual_phase: str, requested_job: Path, requested_phase: str
) -> None:
    """Preserve the current process observer's actual job and phase across repeated registration."""
    if actual_job != requested_job or actual_phase != requested_phase:
        raise RuntimeError("An observer from another gate phase is already active")
