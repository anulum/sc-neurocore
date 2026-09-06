# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio versioned workspace store

"""Saving a workspace adds a revision; it never overwrites one.

Three failures this replaces, each reproduced through the old public API
before it was written:

* **Lost update.** Two editors loaded the same workspace and both saved; the
  second save silently replaced the first, with nothing to notice it by. A
  save now states the revision it was made from, and a save from a stale
  revision is refused as a conflict instead of winning.
* **Unrecoverable delete.** ``delete_project`` called ``os.remove``. Deletion
  now moves the workspace aside, where it can be restored or purged
  deliberately.
* **Torn file.** A save truncated the live file and wrote into it. Under a
  real ``RLIMIT_FSIZE`` refusal a 156-byte workspace became 65,536 bytes of
  unterminated JSON and loading it raised — the previous revision was gone.
  Revisions are written to a sibling temporary file and moved into place.

A workspace directory holds ``revisions/<n>.json`` (immutable), ``head.json``
(which revision is current) and ``trash/`` (deleted workspaces). Forking copies
a revision into a new workspace, so a user can branch without touching the one
they branched from.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sc_neurocore.studio.workspace_lifecycle import (
    delete_workspace,
    deleted_workspaces,
    export_workspace,
    fork_workspace,
    import_workspace,
    restore_workspace,
)
from sc_neurocore.studio.workspace_schema import (
    PROJECT_PAYLOAD_VERSION,
    SCHEMA_VERSION,
    WORKSPACE_SCHEMA_VERSION,
    WorkspaceSchemaError,
    dump_canonical,
    read_document,
    write_atomic,
)

REVISIONS_DIR = "revisions"
#: Suffix of the pre-revision one-file-per-workspace layout.
LEGACY_SUFFIX = ".json"
TRASH_DIR = ".trash"
HEAD_FILE = "head.json"


class WorkspaceConflict(ValueError):
    """Raised when a save is made from a revision that is no longer current.

    Attributes
    ----------
    expected : int or None
        The revision the caller believed was current.
    actual : int
        The revision that is current.
    """

    def __init__(self, *, expected: int | None, actual: int) -> None:
        super().__init__(
            f"the workspace moved to revision {actual} while you were editing revision "
            f"{expected}; reload and reapply your change."
        )
        self.expected = expected
        self.actual = actual

    def to_public_detail(self) -> dict[str, object]:
        """Return the path-free public error detail."""
        return {
            "actual_revision": self.actual,
            "error": "workspace_conflict",
            "expected_revision": self.expected,
            "reason": str(self),
        }


@dataclass(frozen=True, slots=True)
class WorkspaceRevision:
    """One immutable saved revision of a workspace.

    Attributes
    ----------
    name : str
        Workspace name.
    revision : int
        Monotonic revision number, starting at 1.
    saved_at : float
        Unix timestamp the revision was written.
    state_sha256 : str
        Digest of the canonical state JSON.
    parent : int or None
        The revision this one was saved from.
    """

    name: str
    revision: int
    saved_at: float
    state_sha256: str
    parent: int | None

    def to_public_dict(self) -> dict[str, object]:
        """Return a path-free JSON representation of this revision."""
        return {
            "name": self.name,
            "parent": self.parent,
            "revision": self.revision,
            "saved_at": self.saved_at,
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "state_sha256": self.state_sha256,
        }


def state_digest(state: Mapping[str, Any]) -> str:
    """Return the digest of one workspace state's canonical JSON."""
    return hashlib.sha256(dump_canonical(dict(state)).encode("utf-8")).hexdigest()


class WorkspaceStore:
    """A directory of workspaces, each an append-only list of revisions.

    Parameters
    ----------
    root : pathlib.Path
        Directory holding one subdirectory per workspace.
    clock : callable, optional
        Returns the current Unix timestamp; defaults to :func:`time.time`.
    """

    def __init__(self, *, root: Path, clock: Any = None) -> None:
        self._root = root
        self._clock = clock or time.time
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the directory this store keeps workspaces in."""
        return self._root

    def workspace_dir(self, name: str) -> Path:
        """Return the directory holding one workspace's revisions and head."""
        return self._root / name

    def now(self) -> float:
        """Return this store's clock, so collaborators stamp the same time."""
        return float(self._clock())

    def _revision_path(self, name: str, revision: int) -> Path:
        return self.workspace_dir(name) / REVISIONS_DIR / f"{revision}.json"

    def _head_path(self, name: str) -> Path:
        return self.workspace_dir(name) / HEAD_FILE

    def _legacy_path(self, name: str) -> Path:
        """Return where the pre-revision store kept this workspace."""
        return self._root / f"{name}{LEGACY_SUFFIX}"

    def adopt_legacy(self, name: str) -> bool:
        """Bring a pre-revision workspace file into the revision layout.

        Before revisions, a workspace was one flat ``<name>.json`` in the
        project root. Those files are invisible to a store that reads
        ``<name>/head.json``, so an existing installation would have found its
        saved work simply gone. The flat file becomes revision 1 and is left
        on disk untouched: an adoption that writes is reversible by deleting
        the directory, one that deleted would not be.

        A file that is not a readable workspace is skipped, not adopted and
        not raised over: it stays on disk and the name behaves as unused.

        Returns
        -------
        bool
            Whether an adoption happened. Adopting twice is a no-op, because
            the second call finds a head.
        """
        legacy = self._legacy_path(name)
        if self._head_path(name).is_file() or not legacy.is_file():
            return False
        try:
            document = read_document(legacy)
        except WorkspaceSchemaError:
            # A file in the project root that is not a readable workspace is
            # left exactly where it is and adopted by nobody. Raising here
            # would let one unreadable file take down the whole listing.
            return False
        saved_at = float(document.get("saved_at") or self._clock())
        revision = {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "schema_number": SCHEMA_VERSION,
            "name": name,
            "revision": 1,
            "parent": None,
            "saved_at": saved_at,
            "version": document.get("version") or PROJECT_PAYLOAD_VERSION,
            "adopted_from": "studio.project-save.v0",
            "state": dict(document.get("state") or {}),
        }
        (self.workspace_dir(name) / REVISIONS_DIR).mkdir(parents=True, exist_ok=True)
        write_atomic(self._revision_path(name, 1), dump_canonical(revision) + "\n")
        write_atomic(
            self._head_path(name),
            dump_canonical(
                {
                    "schema_version": WORKSPACE_SCHEMA_VERSION,
                    "schema_number": SCHEMA_VERSION,
                    "name": name,
                    "state": {"revision": 1, "saved_at": saved_at},
                }
            )
            + "\n",
        )
        return True

    def exists(self, name: str) -> bool:
        """Return whether a workspace has at least one revision."""
        self.adopt_legacy(name)
        return self._head_path(name).is_file()

    def head_revision(self, name: str) -> int | None:
        """Return the current revision number, or ``None`` for a new workspace."""
        self.adopt_legacy(name)
        path = self._head_path(name)
        if not path.is_file():
            return None
        document = read_document(path)
        return int(document["state"]["revision"])

    def save(
        self,
        name: str,
        state: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> WorkspaceRevision:
        """Append one revision, refusing a save made from a stale one.

        Parameters
        ----------
        name : str
            Workspace name, already validated by the caller.
        state : mapping
            Complete Studio state to persist.
        expected_revision : int, optional
            The revision the caller edited. ``None`` means "this workspace is
            new"; saving with ``None`` over an existing workspace is a
            conflict, because the caller did not know it was there.

        Returns
        -------
        WorkspaceRevision
            The revision that was written.

        Raises
        ------
        WorkspaceConflict
            The workspace has moved on since ``expected_revision``.
        """
        current = self.head_revision(name)
        if current != expected_revision:
            raise WorkspaceConflict(expected=expected_revision, actual=current or 0)
        revision = (current or 0) + 1
        saved_at = float(self._clock())
        document = {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "schema_number": SCHEMA_VERSION,
            "name": name,
            "revision": revision,
            "parent": current,
            "saved_at": saved_at,
            # A revision is also a project payload: the evidence bundle
            # contract reads name, saved_at, version and state from it.
            "version": PROJECT_PAYLOAD_VERSION,
            "state": dict(state),
        }
        directory = self.workspace_dir(name) / REVISIONS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        write_atomic(self._revision_path(name, revision), dump_canonical(document) + "\n")
        head = {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "schema_number": SCHEMA_VERSION,
            "name": name,
            "state": {"revision": revision, "saved_at": saved_at},
        }
        write_atomic(self._head_path(name), dump_canonical(head) + "\n")
        return WorkspaceRevision(
            name=name,
            revision=revision,
            saved_at=saved_at,
            state_sha256=state_digest(state),
            parent=current,
        )

    def load(self, name: str, *, revision: int | None = None) -> dict[str, Any]:
        """Return one revision's document, defaulting to the current one.

        Raises
        ------
        KeyError
            The workspace or the requested revision does not exist.
        WorkspaceSchemaError
            The stored revision cannot be read as a workspace.
        """
        self.adopt_legacy(name)
        target = revision if revision is not None else self.head_revision(name)
        if target is None:
            raise KeyError(name)
        path = self._revision_path(name, target)
        if not path.is_file():
            raise KeyError(f"{name}@{target}")
        return read_document(path)

    def revisions(self, name: str) -> tuple[WorkspaceRevision, ...]:
        """Return every stored revision of one workspace, oldest first."""
        self.adopt_legacy(name)
        directory = self.workspace_dir(name) / REVISIONS_DIR
        if not directory.is_dir():
            return ()
        numbers = sorted(int(path.stem) for path in directory.glob("*.json") if path.stem.isdigit())
        found: list[WorkspaceRevision] = []
        for number in numbers:
            try:
                document = read_document(self._revision_path(name, number))
            except WorkspaceSchemaError:
                # A revision that cannot be read is reported by its absence
                # from this list, never by pretending the workspace is empty.
                continue
            found.append(
                WorkspaceRevision(
                    name=name,
                    revision=number,
                    saved_at=float(document.get("saved_at") or 0.0),
                    state_sha256=state_digest(document.get("state") or {}),
                    parent=document.get("parent"),
                )
            )
        return tuple(found)

    def list_workspaces(self) -> tuple[dict[str, object], ...]:
        """Return one summary per workspace, by name."""
        summaries: list[dict[str, object]] = []
        for path in sorted(self._root.glob(f"*{LEGACY_SUFFIX}")):
            if path.is_file():
                self.adopt_legacy(path.name[: -len(LEGACY_SUFFIX)])
        for directory in sorted(self._root.iterdir()):
            if not directory.is_dir() or directory.name == TRASH_DIR:
                continue
            name = directory.name
            head = self.head_revision(name) if self.exists(name) else None
            if head is None:
                continue
            summaries.append(
                {
                    "name": name,
                    "revision": head,
                    "revision_count": len(self.revisions(name)),
                }
            )
        return tuple(summaries)

    def fork(self, name: str, new_name: str, *, revision: int | None = None) -> WorkspaceRevision:
        """Copy one revision into a new workspace; see ``workspace_lifecycle``."""
        return fork_workspace(self, name, new_name, revision=revision)

    def delete(self, name: str) -> Path:
        """Move a workspace aside so it can be restored."""
        return delete_workspace(self, name)

    def deleted(self) -> tuple[dict[str, object], ...]:
        """Return the workspaces waiting in the trash, newest first."""
        return deleted_workspaces(self)

    def restore(self, token: str) -> str:
        """Bring one deleted workspace back under its original name."""
        return restore_workspace(self, token)

    def export_document(self, name: str, *, revision: int | None = None) -> dict[str, Any]:
        """Return one revision as a self-contained document."""
        return export_workspace(self, name, revision=revision)

    def import_document(self, name: str, document: Mapping[str, Any]) -> WorkspaceRevision:
        """Create a workspace from an exported document."""
        return import_workspace(self, name, document)


__all__ = [
    "HEAD_FILE",
    "LEGACY_SUFFIX",
    "REVISIONS_DIR",
    "TRASH_DIR",
    "WorkspaceConflict",
    "WorkspaceRevision",
    "WorkspaceStore",
    "state_digest",
]
