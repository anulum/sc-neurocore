# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — benchmark variant source digests

"""Check the source digests a benchmark record holds once per variant, each at that variant's own revision.

A record that compares a parent revision with a candidate hashes each variant's sources inside
``variants.<i>.source_hashes``. The evidence gate's ``source_hashes`` map points at one place per source path, and the
record's commit holds one version of each source, so these digests cannot be checked like the others: a parent's
digests describe the parent revision. A gate that sets ``variant_source_hashes`` to :data:`VARIANT_REVISION_MODE`
checks each variant where it ran:

* a clean variant at its ``git_head``, a parent (no ``git_head``) at its ``source_ref``;
* a variant measured on a dirty tree (``surface_dirty`` true) at the record's commit, the only commit that can hold the
  bytes it hashed; under the gate's ``worktree`` policy, against the files on disk.

The failures are ``unresolvable_variant_revision`` (the revision is not in the history), ``stale_variant_source_hash``
(a clean variant's digest is not of its revision's bytes), ``dirty_variant_not_committed`` (the record's commit did
not take a dirty variant's bytes), ``missing_historical_source``, ``invalid_source_path``, ``missing_variants`` and
``variant_without_source_hashes``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from pathlib import Path
from typing import Any

#: The ``variant_source_hashes`` value that asks the gate to check every variant at its own revision.
VARIANT_REVISION_MODE = "variant_revision"


def variant_revision(variant: Mapping[str, Any]) -> str | None:
    """Return the revision a clean variant ran: its ``git_head``, else a parent's ``source_ref``, else ``None``.

    Parameters
    ----------
    variant : Mapping
        One entry of a record's ``variants``.
    """
    for field in ("git_head", "source_ref"):
        value = variant.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def variant_source_hash_failures(
    payload: Any,
    *,
    repo_root: Path,
    record_commit: str | None,
    git_bytes: Callable[..., bytes | None],
    safe_path: Callable[[str], bool],
) -> list[tuple[str, str]]:
    """Check every variant's source digests at its own revision and return the failures as ``(reason, place)``.

    Parameters
    ----------
    payload : Any
        The benchmark record.
    repo_root : Path
        The repository root.
    record_commit : str or None
        The commit that wrote the record, where a dirty variant is checked; ``None`` under the ``worktree`` policy,
        where a dirty variant is compared with the files on disk.
    git_bytes : callable
        Runs ``git -C repo_root <args>`` and returns its output, or ``None`` when it fails.
    safe_path : callable
        Whether a source path is a plain path inside the repository.

    Returns
    -------
    list of tuple of (str, str)
        Each failure's reason and its place in the record, in order.
    """
    variants = payload.get("variants") if isinstance(payload, dict) else None
    if not isinstance(variants, list) or not variants:
        return [("missing_variants", "variants")]
    failures: list[tuple[str, str]] = []
    for index, variant in enumerate(variants):
        place = f"variants.{index}"
        hashes = variant.get("source_hashes") if isinstance(variant, dict) else None
        if not isinstance(variant, dict) or not isinstance(hashes, dict) or not hashes:
            failures.append(("variant_without_source_hashes", place))
            continue
        dirty = variant.get("surface_dirty") is True
        revision = (
            record_commit if dirty else _resolve(repo_root, variant_revision(variant), git_bytes)
        )
        if revision is None and not dirty:
            failures.append(("unresolvable_variant_revision", place))
            continue
        for path, recorded in sorted(hashes.items()):
            where = f"{place}.source_hashes.{path}"
            if not safe_path(path):
                failures.append(("invalid_source_path", where))
                continue
            data = _source_bytes(repo_root, revision, path, git_bytes)
            if data is None:
                failures.append(("missing_historical_source", where))
            elif hashlib.sha256(data).hexdigest() != recorded:
                failures.append(
                    ("dirty_variant_not_committed" if dirty else "stale_variant_source_hash", where)
                )
    return failures


def _resolve(
    repo_root: Path, name: str | None, git_bytes: Callable[..., bytes | None]
) -> str | None:
    """Return the commit ``name`` resolves to in the repository, or ``None``.

    Parameters
    ----------
    repo_root : Path
        The repository root.
    name : str or None
        A revision name, full or abbreviated.
    git_bytes : callable
        The Git runner.
    """
    if name is None:
        return None
    resolved = git_bytes(repo_root, "rev-parse", "--verify", "--quiet", f"{name}^{{commit}}")
    if not resolved:
        return None
    return resolved.decode("ascii", errors="ignore").strip() or None


def _source_bytes(
    repo_root: Path, revision: str | None, path: str, git_bytes: Callable[..., bytes | None]
) -> bytes | None:
    """Return ``path``'s bytes at ``revision``, or on disk when there is no revision; ``None`` when absent.

    Parameters
    ----------
    repo_root : Path
        The repository root.
    revision : str or None
        The commit to read, or ``None`` for the working files.
    path : str
        The source path, relative to the root.
    git_bytes : callable
        The Git runner.
    """
    if revision is not None:
        return git_bytes(repo_root, "show", f"{revision}:{path}")
    source = repo_root / path
    return source.read_bytes() if source.is_file() else None


__all__ = ["VARIANT_REVISION_MODE", "variant_revision", "variant_source_hash_failures"]
