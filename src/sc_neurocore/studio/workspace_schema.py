# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio workspace revision schema and durable writes

"""The shape of a saved workspace, and how it reaches the disk intact.

A workspace used to be one file that every save truncated and rewrote in
place. A write that failed part-way — a full disk, a quota, a killed process —
left a shorter file of valid-looking bytes and destroyed the revision that was
already there. Measured with a real ``RLIMIT_FSIZE`` refusal: a 156-byte
workspace became 65,536 bytes of unterminated JSON, and loading it raised.

So a revision is written to a sibling temporary file, flushed, ``fsync``ed and
then ``os.replace``d into place. Replace is atomic on POSIX: a reader sees the
old bytes or the new ones, never half of either, and a failure before the
replace leaves the previous revision untouched.

Revisions themselves are immutable. A save adds one; nothing rewrites an
existing revision file, so the history a user can go back to is exactly the
history they wrote.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

WORKSPACE_SCHEMA_VERSION = "studio.workspace.v1"
SCHEMA_VERSION = 1

#: Project payload version every revision carries. The evidence bundle
#: contract requires it alongside the name, timestamp and state, so a revision
#: is a valid project payload as well as a workspace document.
PROJECT_PAYLOAD_VERSION = "0.3.0"

#: Blocks of Studio state a revision carries. Named rather than inferred, so a
#: save that silently forgot the network graph or the hardware profile is a
#: test failure and not an unnoticed loss.
WORKSPACE_STATE_BLOCKS = (
    "experiment",
    "graph",
    "candidates",
    "analysis_refs",
    "run_refs",
    "hardware_profile",
)


class WorkspaceSchemaError(ValueError):
    """Raised when a stored workspace cannot be read as one."""


def migrate_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Bring one stored workspace document forward to this schema.

    Parameters
    ----------
    document : mapping
        The parsed contents of a revision file, or a legacy single-file
        project payload.

    Returns
    -------
    dict
        The document at the current schema version.

    Raises
    ------
    WorkspaceSchemaError
        The document is not an object, is missing its state, or was written by
        a newer schema. A newer document is refused rather than downgraded:
        dropping fields a future build added would lose a user's work quietly.
    """
    if not isinstance(document, Mapping):
        raise WorkspaceSchemaError("a workspace document must be an object")
    stored = document.get("schema_version")
    if stored is None:
        # A legacy project payload: one flat file with name/saved_at/version
        # and a state object. It becomes revision content unchanged, so the
        # save that follows it is the first immutable revision.
        state = document.get("state")
        if not isinstance(state, Mapping):
            raise WorkspaceSchemaError("a legacy project payload must carry a 'state' object")
        return {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "schema_number": SCHEMA_VERSION,
            "migrated_from": "studio.project-save.v0",
            "name": document.get("name"),
            "saved_at": document.get("saved_at"),
            "version": document.get("version") or PROJECT_PAYLOAD_VERSION,
            "state": dict(state),
        }
    if stored != WORKSPACE_SCHEMA_VERSION:
        raise WorkspaceSchemaError(
            f"unsupported workspace schema {stored!r}; this build reads "
            f"{WORKSPACE_SCHEMA_VERSION}. Upgrade the package rather than downgrading the "
            "workspace."
        )
    number = int(document.get("schema_number", SCHEMA_VERSION))
    if number > SCHEMA_VERSION:
        raise WorkspaceSchemaError(
            f"the workspace was written by schema number {number}; this build understands "
            f"{SCHEMA_VERSION}."
        )
    if not isinstance(document.get("state"), Mapping):
        raise WorkspaceSchemaError("a workspace document must carry a 'state' object")
    return dict(document)


def dump_canonical(document: Mapping[str, Any]) -> str:
    """Serialise a workspace document deterministically."""
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def write_atomic(path: Path, payload: str) -> None:
    """Write one file so a reader never sees a partial one.

    The payload goes to a temporary file beside the target, is flushed and
    ``fsync``ed, and is then moved into place with ``os.replace``. The
    directory is ``fsync``ed too, so the rename itself survives a power loss.
    A failure anywhere before the replace leaves the existing file untouched
    and removes the temporary one.

    Parameters
    ----------
    path : pathlib.Path
        Target file. Its parent directory must exist.
    payload : str
        Complete file contents.
    """
    directory = path.parent
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    except OSError:  # pragma: no cover - directory fsync is unsupported on some filesystems
        pass
    finally:
        os.close(directory_fd)


def read_document(path: Path) -> dict[str, Any]:
    """Read and migrate one stored workspace document.

    Raises
    ------
    WorkspaceSchemaError
        The file is unreadable, is not JSON, or is not a workspace this build
        understands. The message never contains the path.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkspaceSchemaError(
            f"the workspace revision could not be read ({exc.strerror})"
        ) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkspaceSchemaError(f"the workspace revision is not valid JSON ({exc.msg})") from exc
    return migrate_document(parsed)


__all__ = [
    "PROJECT_PAYLOAD_VERSION",
    "SCHEMA_VERSION",
    "WORKSPACE_SCHEMA_VERSION",
    "WORKSPACE_STATE_BLOCKS",
    "WorkspaceSchemaError",
    "dump_canonical",
    "migrate_document",
    "read_document",
    "write_atomic",
]
