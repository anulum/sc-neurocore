# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Review comments bound to immutable workspace revisions

"""Comment on one saved revision of a workspace, and know it is still that revision.

A revision is immutable, so a comment names the revision it is about and the
digest of that revision's state when it was written. Reading the comments
recomputes each revision's digest: a comment whose revision has gone, or whose
revision no longer has the digest it was written against, is marked, never
silently shown as if it still applied. Comments are appended to one line-per-
comment file beside the workspace's revisions and are never rewritten; a reply
names the comment it answers, which must be on the same revision.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass
from typing import Any

from sc_neurocore.studio.workspace_schema import WorkspaceSchemaError
from sc_neurocore.studio.workspace_store import WorkspaceStore, state_digest

REVIEW_SCHEMA_VERSION = "studio.workspace-review.v1"
REVIEW_FILE = "review.jsonl"
MAX_COMMENT_CHARS = 4000


@dataclass(frozen=True)
class ReviewComment:
    """One comment on one revision."""

    comment_id: str
    revision: int
    state_sha256: str
    author: str
    created_at: float
    body: str
    reply_to: str | None


def _path(store: WorkspaceStore, name: str) -> Any:
    return store.workspace_dir(name) / REVIEW_FILE


def _read(store: WorkspaceStore, name: str) -> list[ReviewComment]:
    path = _path(store, name)
    if not path.is_file():
        return []
    comments: list[ReviewComment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            comments.append(
                ReviewComment(**{k: entry[k] for k in ReviewComment.__dataclass_fields__})
            )
    return comments


def add_comment(
    store: WorkspaceStore,
    name: str,
    revision: int,
    *,
    author: str,
    body: str,
    reply_to: str | None = None,
) -> ReviewComment:
    """Append a comment on ``name`` at ``revision``.

    Raises
    ------
    KeyError
        The workspace or the revision does not exist.
    ValueError
        The body is empty or longer than :data:`MAX_COMMENT_CHARS`, or
        ``reply_to`` is not a comment on the same revision.
    """
    text = body.strip()
    if not text or len(text) > MAX_COMMENT_CHARS:
        raise ValueError(f"a comment holds 1 to {MAX_COMMENT_CHARS} characters")
    with store.lock(name):
        document = store.load(name, revision=revision)
        if reply_to is not None and not any(
            comment.comment_id == reply_to and comment.revision == revision
            for comment in _read(store, name)
        ):
            raise ValueError(f"{reply_to} is not a comment on revision {revision}")
        comment = ReviewComment(
            comment_id=secrets.token_hex(8),
            revision=revision,
            state_sha256=state_digest(document.get("state") or {}),
            author=author,
            created_at=store.now(),
            body=text,
            reply_to=reply_to,
        )
        path = _path(store, name)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(comment), sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return comment


def list_comments(
    store: WorkspaceStore, name: str, *, revision: int | None = None
) -> dict[str, Any]:
    """Return a workspace's comments, each checked against its revision.

    Raises
    ------
    KeyError
        The workspace does not exist.
    """
    if not store.exists(name):
        raise KeyError(name)
    with store.lock(name):
        digests: dict[int, str | None] = {}
        entries: list[dict[str, Any]] = []
        for comment in _read(store, name):
            if revision is not None and comment.revision != revision:
                continue
            if comment.revision not in digests:
                try:
                    document = store.load(name, revision=comment.revision)
                    digests[comment.revision] = state_digest(document.get("state") or {})
                except (KeyError, WorkspaceSchemaError):
                    digests[comment.revision] = None
            current = digests[comment.revision]
            entries.append(
                {
                    **asdict(comment),
                    "revision_status": (
                        "missing"
                        if current is None
                        else "matches"
                        if current == comment.state_sha256
                        else "changed"
                    ),
                }
            )
    return {"schema_version": REVIEW_SCHEMA_VERSION, "workspace": name, "comments": entries}


__all__ = [
    "MAX_COMMENT_CHARS",
    "REVIEW_FILE",
    "REVIEW_SCHEMA_VERSION",
    "ReviewComment",
    "add_comment",
    "list_comments",
]
