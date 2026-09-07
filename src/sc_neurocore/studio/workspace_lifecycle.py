# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio workspace lifecycle

"""Forking, deleting, restoring and moving a workspace between installations.

Saving a revision is one responsibility and lives in
:mod:`sc_neurocore.studio.workspace_store`; what happens to a whole workspace
is this one. The rule they share: nothing here overwrites a workspace that
already exists, and nothing removes one irrecoverably. Deleting moves the
directory aside, restoring brings it back only onto a free name, and forking
writes a new workspace rather than touching the one it copied.

Each of those checks is only worth as much as its atomicity, so deleting and
restoring hold the workspace the same way a save does: a delete that ran
between a save's head check and its write, or two restores racing onto one
free name, would defeat the check by walking through it. Forking and importing
need no lock of their own — they create their workspace through
:meth:`~sc_neurocore.studio.workspace_store.WorkspaceStore.save`, which holds
the destination.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sc_neurocore.studio.workspace_schema import (
    WORKSPACE_STATE_BLOCKS,
    migrate_document,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from sc_neurocore.studio.workspace_store import WorkspaceRevision, WorkspaceStore

TRASH_DIR = ".trash"


def _conflict(expected: int | None, actual: int) -> Exception:
    """Build a workspace conflict without importing the store at module scope.

    The store imports these helpers, so binding its exception here at import
    time would close a cycle.
    """
    from sc_neurocore.studio.workspace_store import WorkspaceConflict

    return WorkspaceConflict(expected=expected, actual=actual)


def fork_workspace(
    store: WorkspaceStore, name: str, new_name: str, *, revision: int | None = None
) -> WorkspaceRevision:
    """Copy one revision into a new workspace as its first revision.

    Raises
    ------
    KeyError
        The source workspace or revision does not exist.
    WorkspaceConflict
        The destination workspace already exists; forking never
        overwrites one.
    """
    document = store.load(name, revision=revision)
    return store.save(new_name, document["state"], expected_revision=None)


def branch_conflicting_edit(
    store: WorkspaceStore,
    name: str,
    state: Mapping[str, Any],
    *,
    base_revision: int,
    branch_name: str | None = None,
) -> WorkspaceRevision:
    """Keep an edit that lost a save conflict, instead of asking for it again.

    A save from a stale revision is refused so it cannot overwrite the other
    editor's work, which is correct — but the refused edit then exists only in
    the editor's browser, and the message asks them to reapply it. This stores
    it as the first revision of its own workspace, so both edits survive and
    either can be compared, exported or merged by hand afterwards.

    The branch records the workspace and revision it diverged from in its name,
    which is what a reader needs to reconcile the two later.

    Parameters
    ----------
    store : WorkspaceStore
        Store holding the workspace whose save was refused.
    name : str
        Workspace the edit was made against.
    state : Mapping[str, Any]
        The refused state, exactly as the editor had it.
    base_revision : int
        The revision the editor was working from.
    branch_name : str, optional
        Name for the branch. Defaults to ``"<name> (from revision <n>)"``.

    Returns
    -------
    WorkspaceRevision
        The first revision of the branch.

    Raises
    ------
    KeyError
        The source workspace or ``base_revision`` does not exist, so the edit
        does not describe a divergence from anything.
    WorkspaceConflict
        A workspace of that name already exists; branching never overwrites one.
    """
    store.load(name, revision=base_revision)
    target = branch_name or f"{name} (from revision {base_revision})"
    return store.save(target, state, expected_revision=None)


def delete_workspace(store: WorkspaceStore, name: str) -> Path:
    """Move a workspace aside so it can be restored.

    Returns
    -------
    pathlib.Path
        Where the workspace now lives.

    Raises
    ------
    KeyError
        The workspace does not exist.
    """
    with store.lock(name):
        source = store.workspace_dir(name)
        if not source.is_dir():
            raise KeyError(name)
        trash = store.root / TRASH_DIR
        trash.mkdir(parents=True, exist_ok=True)
        destination = trash / f"{name}.{int(store.now() * 1000)}"
        shutil.move(str(source), str(destination))
        return destination


def deleted_workspaces(store: WorkspaceStore) -> tuple[dict[str, object], ...]:
    """Return the workspaces waiting in the trash, newest first."""
    trash = store.root / TRASH_DIR
    if not trash.is_dir():
        return ()
    entries: list[dict[str, object]] = []
    for directory in trash.iterdir():
        if not directory.is_dir():
            continue
        name, _, stamp = directory.name.rpartition(".")
        entries.append(
            {
                "deleted_at": int(stamp) / 1000 if stamp.isdigit() else None,
                "name": name or directory.name,
                "token": directory.name,
            }
        )
    return tuple(sorted(entries, key=lambda entry: str(entry["token"]), reverse=True))


def restore_workspace(store: WorkspaceStore, token: str) -> str:
    """Bring one deleted workspace back under its original name.

    Raises
    ------
    KeyError
        No deleted workspace carries that token.
    WorkspaceConflict
        A live workspace already holds the name; restoring would overwrite
        it, which is the loss this store exists to prevent.
    """
    source = store.root / TRASH_DIR / token
    if not source.is_dir():
        raise KeyError(token)
    name, _, _stamp = token.rpartition(".")
    name = name or token
    with store.lock(name):
        destination = store.workspace_dir(name)
        if destination.exists():
            raise _conflict(None, store.head_revision(name) or 0)
        shutil.move(str(source), str(destination))
        return name


def export_workspace(
    store: WorkspaceStore, name: str, *, revision: int | None = None
) -> dict[str, Any]:
    """Return one revision as a self-contained document."""
    document = store.load(name, revision=revision)
    blocks = document.get("state") or {}
    return {
        **document,
        "exported_at": store.now(),
        "state_blocks_present": [block for block in WORKSPACE_STATE_BLOCKS if block in blocks],
    }


def import_workspace(
    store: WorkspaceStore, name: str, document: Mapping[str, Any]
) -> WorkspaceRevision:
    """Create a workspace from an exported document.

    Raises
    ------
    WorkspaceSchemaError
        The document is not a workspace this build reads.
    WorkspaceConflict
        The destination already exists.
    """
    migrated = migrate_document(document)
    return store.save(name, migrated["state"], expected_revision=None)


__all__ = [
    "TRASH_DIR",
    "delete_workspace",
    "deleted_workspaces",
    "export_workspace",
    "fork_workspace",
    "import_workspace",
    "restore_workspace",
]
