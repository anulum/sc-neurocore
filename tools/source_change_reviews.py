# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — benchmark source-change reviews

"""Recorded reasons for not measuring a benchmark record again, and the check that holds them.

A benchmark record binds the bytes of its sources. When a bound source changes, the coverage census calls the record
``outdated``; it describes earlier bytes, and the rule is that no stale number is shipped. Most changes need a new
measurement. A change of documentation or comments only cannot move a measured number, and for that case a review in
:data:`REVIEWS_FILE` records the reason instead::

    schema_version = 1

    [[review]]
    record = "bench_kl_refine.json"           # below benchmarks/results
    path = "engine/src/partition.rs"          # the bound source that changed
    source_sha256 = "<64 hex>"                # the source's current bytes, as reviewed
    reason = "Only Rust comments changed; ..."

The census accepts a review only when the source still has the reviewed bytes, and when its code form
(:func:`tools.source_code_equivalence.same_code`) equals that of the bytes the record measured. Those bytes are found
in the repository's history by the record's digest. A review of a code change, of other bytes, of a language without
a code form, or outside a repository is therefore rejected, so a reason never stands in for a measurement that is
owed. The file is read strictly: a malformed or duplicated review is refused, not skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the repository runs 3.12; kept for the declared floor
    import tomli as tomllib

from tools.source_code_equivalence import same_code

#: The reviews file, relative to the repository root.
REVIEWS_FILE = Path("benchmarks/source_change_reviews.toml")

#: The fields every review has, and nothing else.
_FIELDS = ("record", "path", "source_sha256", "reason")

#: A SHA-256 digest as the records write it.
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Review:
    """One recorded reason for not measuring a record again after a source changed.

    Attributes
    ----------
    record : str
        The record, below ``benchmarks/results``.
    path : str
        The bound source that changed, relative to the repository root.
    source_sha256 : str
        The SHA-256 of the source's bytes the review was made on.
    reason : str
        Why the change cannot move the record's numbers.
    """

    record: str
    path: str
    source_sha256: str
    reason: str


def load_reviews(repo_root: Path) -> dict[tuple[str, str], Review]:
    """Read the reviews file, keyed by ``(record, path)``.

    Parameters
    ----------
    repo_root : Path
        The repository root.

    Returns
    -------
    dict
        Every review; empty when the file does not exist.

    Raises
    ------
    ValueError
        If the schema version is not 1, a review lacks a field or has another, a field is not a nonempty string, a
        digest is not 64 lowercase hexadecimal characters, a record or path is absolute or has a ``..`` part, or a
        ``(record, path)`` is reviewed twice.
    """
    path = repo_root / REVIEWS_FILE
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"{REVIEWS_FILE}: schema_version must be 1")
    reviews: dict[tuple[str, str], Review] = {}
    for index, entry in enumerate(data.get("review", [])):
        if (
            not isinstance(entry, dict)
            or set(entry) != set(_FIELDS)
            or not all(isinstance(entry[field], str) and entry[field] for field in _FIELDS)
        ):
            raise ValueError(
                f"{REVIEWS_FILE}: review {index} needs exactly {', '.join(_FIELDS)}, nonempty"
            )
        if not _DIGEST.match(entry["source_sha256"]):
            raise ValueError(
                f"{REVIEWS_FILE}: review {index}'s source_sha256 is not a SHA-256 digest"
            )
        for field in ("record", "path"):
            parts = PurePosixPath(entry[field])
            if parts.is_absolute() or ".." in parts.parts:
                raise ValueError(
                    f"{REVIEWS_FILE}: review {index}'s {field} must be a relative path inside the repository"
                )
        key = (entry["record"], entry["path"])
        if key in reviews:
            raise ValueError(f"{REVIEWS_FILE}: {key[0]}:{key[1]} is reviewed twice")
        reviews[key] = Review(**entry)
    return reviews


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    """Run Git in ``repo_root`` and return the finished process."""
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, check=False)


def measured_bytes(repo_root: Path, path: str, digest: str) -> bytes | None:
    """Return the version of ``path`` in the repository's history whose SHA-256 is ``digest``.

    Parameters
    ----------
    repo_root : Path
        The repository root; it must be the top of a Git work tree.
    path : str
        The source path, relative to the root.
    digest : str
        The SHA-256 the record holds for it.

    Returns
    -------
    bytes or None
        The measured bytes, or ``None`` outside a work tree's top or when no commit holds them.
    """
    top = _git(repo_root, "rev-parse", "--show-toplevel")
    if top.returncode != 0 or Path(top.stdout.decode().strip()).resolve() != repo_root.resolve():
        return None
    for commit in _git(repo_root, "log", "--format=%H", "--", path).stdout.decode().split():
        shown = _git(repo_root, "show", f"{commit}:{path}")
        if shown.returncode == 0 and hashlib.sha256(shown.stdout).hexdigest() == digest:
            return shown.stdout
    return None


def review_accepted(review: Review, repo_root: Path, recorded_digest: str) -> bool:
    """Say whether a review stands: the reviewed bytes are current and their code is the measured code.

    Parameters
    ----------
    review : Review
        The review.
    repo_root : Path
        The repository root.
    recorded_digest : str
        The digest the record holds for the reviewed source.

    Returns
    -------
    bool
        True only when the source has the reviewed bytes, the measured bytes are found in history, and both have
        the same code form.
    """
    source = repo_root / review.path
    if not source.is_file():
        return False
    current = source.read_bytes()
    if hashlib.sha256(current).hexdigest() != review.source_sha256:
        return False
    measured = measured_bytes(repo_root, review.path, recorded_digest)
    return measured is not None and same_code(review.path, measured, current)


__all__ = ["REVIEWS_FILE", "Review", "load_reviews", "measured_bytes", "review_accepted"]
