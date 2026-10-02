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
import math
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.workspace_store import WorkspaceStore, state_digest

REVIEW_SCHEMA_VERSION = "studio.workspace-review.v1"
REVIEW_FILE = "review.jsonl"
MAX_COMMENT_CHARS = 4000


class WorkspaceReviewRefused(AuthoredRefusal):
    """An authored explanation of invalid comment text or a reply target."""


class WorkspaceReviewMissing(KeyError):
    """The requested workspace or revision is absent, rather than unreadable."""


class WorkspaceReviewStorageError(RuntimeError):
    """Stored review or revision data cannot support a trustworthy comment."""


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


def _path(store: WorkspaceStore, name: str) -> Path:
    """Locate the append-only review file beside its workspace revisions."""
    return store.workspace_dir(name) / REVIEW_FILE


def _read_comment(line: str) -> ReviewComment:
    """Validate persisted fields before they become a public review record."""
    try:
        record: object = json.loads(line)
    except json.JSONDecodeError as exc:
        raise WorkspaceReviewStorageError("Unreadable workspace review record") from exc
    if not isinstance(record, dict):
        raise WorkspaceReviewStorageError("Invalid workspace review record")
    comment_id: object = record.get("comment_id")
    revision: object = record.get("revision")
    digest: object = record.get("state_sha256")
    author: object = record.get("author")
    created_at: object = record.get("created_at")
    body: object = record.get("body")
    reply_to: object = record.get("reply_to")
    if (
        not isinstance(comment_id, str)
        or not comment_id
        or not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < 1
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or not isinstance(author, str)
        or not isinstance(created_at, (int, float))
        or isinstance(created_at, bool)
        or not math.isfinite(created_at)
        or not isinstance(body, str)
        or not body.strip()
        or len(body) > MAX_COMMENT_CHARS
        or (reply_to is not None and not isinstance(reply_to, str))
        or "reply_to" not in record
    ):
        raise WorkspaceReviewStorageError("Invalid workspace review record")
    try:
        for text in (comment_id, author, body, reply_to or ""):
            text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise WorkspaceReviewStorageError("Invalid workspace review text") from exc
    return ReviewComment(comment_id, revision, digest, author, float(created_at), body, reply_to)


def _read(store: WorkspaceStore, name: str) -> list[ReviewComment]:
    """Read actual review records; a missing file alone means no comments."""
    path = _path(store, name)
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError) as exc:
        raise WorkspaceReviewStorageError("Unreadable workspace review file") from exc
    comments: list[ReviewComment] = []
    for line in content.splitlines():
        if line.strip():
            comments.append(_read_comment(line))
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
    WorkspaceReviewMissing
        The workspace or the revision does not exist.
    WorkspaceReviewRefused
        The body is empty or longer than :data:`MAX_COMMENT_CHARS`, comment
        text or author is not UTF-8, or the reply is not on the same revision.
    WorkspaceReviewStorageError
        Persisted review or revision data is unreadable; nothing is appended.
    """
    text = body.strip()
    if not text or len(text) > MAX_COMMENT_CHARS:
        raise WorkspaceReviewRefused(f"a comment holds 1 to {MAX_COMMENT_CHARS} characters")
    try:
        text.encode("utf-8")
        author.encode("utf-8")
    except UnicodeEncodeError:
        raise WorkspaceReviewRefused("comment text and author must be valid UTF-8") from None
    with store.lock(name):
        try:
            document = store.load(name, revision=revision)
            digest = state_digest(document.get("state") or {})
        except KeyError as exc:
            raise WorkspaceReviewMissing(name) from exc
        except (ValueError, TypeError, OverflowError) as exc:
            raise WorkspaceReviewStorageError("Unreadable workspace review revision") from exc
        comments = _read(store, name)
        if reply_to is not None and not any(
            comment.comment_id == reply_to and comment.revision == revision for comment in comments
        ):
            display_id = reply_to.encode("utf-8", errors="backslashreplace").decode("utf-8")
            raise WorkspaceReviewRefused(f"{display_id} is not a comment on revision {revision}")
        comment = ReviewComment(
            comment_id=secrets.token_hex(8),
            revision=revision,
            state_sha256=digest,
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
    WorkspaceReviewMissing
        The workspace does not exist.
    WorkspaceReviewStorageError
        Existing review records or their saved revisions cannot be read safely.
    """
    if not store.exists(name):
        raise WorkspaceReviewMissing(name)
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
                except KeyError:
                    digests[comment.revision] = None
                except (ValueError, TypeError, OverflowError) as exc:
                    raise WorkspaceReviewStorageError(
                        "Unreadable workspace review revision"
                    ) from exc
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
    "WorkspaceReviewMissing",
    "WorkspaceReviewRefused",
    "WorkspaceReviewStorageError",
    "add_comment",
    "list_comments",
]
