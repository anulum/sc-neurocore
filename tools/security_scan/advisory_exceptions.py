# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact advisory exceptions with installed vendor-source evidence

"""Accept named Grype matches only while their exact installed vendor evidence still holds.

Two reasons exist. ``fixed_code_present`` accepts a finding whose installed file
equals the immutable fixed vendor source. ``vendor_fix_unreleased`` accepts, for at
most two weeks per review, a finding whose fix the vendor has not released: the
installed file must equal the immutable source of the installed release, and the
vendor channel must list no release newer than the installed version.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tomllib
from typing import Any
import urllib.request

from packaging.version import InvalidVersion, Version
from tools.security_scan.locked_dependency_inventory import inventory_locks

EXCEPTION_FILE = Path("tools/security_scan/advisory_exceptions.toml")
CONDA_FORGE_PYTHON_INDEX = "https://api.anaconda.org/package/conda-forge/python"
_SOURCE_LIMIT = 65536
_INDEX_LIMIT = 32 * 1024 * 1024
UNRELEASED_FIX_REVIEW_WINDOW = timedelta(days=14)
# Each reason admits exactly one advisory and the immutable CPython file that proves it.
_REASONS = {
    "fixed_code_present": (
        "CVE-2025-15367",
        r"https://raw\.githubusercontent\.com/python/cpython/[0-9a-f]{40}/Lib/poplib\.py",
    ),
    "vendor_fix_unreleased": (
        "CVE-2026-103995",
        r"https://raw\.githubusercontent\.com/python/cpython/[0-9a-f]{40}/Lib/http/client\.py",
    ),
}
ReadUrl = Callable[[str, int], bytes]
_FIELDS = frozenset(
    {
        "lock",
        "package",
        "version",
        "advisory",
        "artifact_sha256",
        "reason",
        "fixed_version",
        "proof_path",
        "proof_sha256",
        "source",
        "review_by",
        "rationale",
    }
)


def read_url(url: str, limit: int) -> bytes:
    """Read at most ``limit`` bytes from a pinned HTTPS address.

    Parameters
    ----------
    url : str
        Address fixed by this module or by a validated exception record.
    limit : int
        Number of bytes to read; callers pass one more than they accept.

    Returns
    -------
    bytes
        The bytes read, which a caller rejects when they reach ``limit``.
    """
    # Every address is an HTTPS constant or passed the record's source pattern.
    with urllib.request.urlopen(url, timeout=30) as response:  # nosec B310
        return bytes(response.read(limit))


def newer_vendor_releases(index: bytes, installed: str) -> list[str]:
    """Return the final releases in a channel package index that are newer than ``installed``.

    Parameters
    ----------
    index : bytes
        JSON package document whose ``versions`` member lists every published version.
    installed : str
        Version of the installed package.

    Returns
    -------
    list of str
        Newer final releases in ascending order; prereleases are not installable fixes.

    Raises
    ------
    ValueError
        The document has no version list or a version cannot be ordered.
    """
    payload = json.loads(index)
    versions = payload.get("versions") if isinstance(payload, dict) else None
    if not isinstance(versions, list) or not versions:
        raise ValueError("Vendor channel index has no version list.")
    current = Version(installed)
    newer: list[Version] = []
    for value in versions:
        if not isinstance(value, str):
            raise ValueError("Vendor channel index version is not text.")
        try:
            version = Version(value)
        except InvalidVersion as exc:
            raise ValueError("Vendor channel index version cannot be ordered.") from exc
        if not version.is_prerelease and not version.is_devrelease and version > current:
            newer.append(version)
    return [str(version) for version in sorted(newer)]


def load_exceptions(raw: bytes) -> tuple[dict[str, str], ...]:
    """Read exact, dated Python vendor-source exception records.

    Parameters
    ----------
    raw : bytes
        Complete UTF-8 TOML table, including an empty exception list.

    Returns
    -------
    tuple of dict
        Validated records keyed by lock, package, version and advisory.

    Raises
    ------
    ValueError
        Scope, evidence, review date or record cardinality is invalid.
        Wildcards and non-Python version comparisons are unsupported.
    """
    payload = tomllib.loads(raw.decode("utf-8"))
    rows = payload.get("exception")
    if set(payload) != {"exception"} or not isinstance(rows, list):
        raise ValueError("Advisory exception table is invalid.")
    records: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != _FIELDS
            or any(
                not isinstance(value, str) or not value.strip() or "*" in value
                for value in row.values()
            )
        ):
            raise ValueError("Advisory exception fields are invalid.")
        if row["package"] != "python" or row["reason"] not in _REASONS:
            raise ValueError("Advisory exception reason or package is unsupported.")
        for key in ("lock", "proof_path"):
            path = PurePosixPath(row[key])
            if path.is_absolute() or ".." in path.parts or path.as_posix() != row[key]:
                raise ValueError("Advisory exception path must be relative.")
        if any(
            re.fullmatch("[0-9a-f]{64}", row[key]) is None
            for key in ("artifact_sha256", "proof_sha256")
        ):
            raise ValueError("Advisory exception digest is invalid.")
        advisory, source = _REASONS[row["reason"]]
        if re.fullmatch(source, row["source"]) is None or row["advisory"] != advisory:
            raise ValueError("Advisory exception vendor source is unsupported.")
        today = datetime.now(timezone.utc).date()
        review_by = datetime.fromisoformat(row["review_by"]).date()
        if review_by < today:
            raise ValueError("Advisory exception requires renewed review.")
        unfixed = Version(row["version"]) < Version(row["fixed_version"])
        if row["reason"] == "fixed_code_present" and unfixed:
            raise ValueError("Advisory exception version predates the vendor fix.")
        if row["reason"] == "vendor_fix_unreleased":
            if not unfixed:
                raise ValueError("Advisory exception version already contains the vendor fix.")
            if review_by > today + UNRELEASED_FIX_REVIEW_WINDOW:
                raise ValueError("Advisory exception review date exceeds two weeks.")
        scope = tuple(row[field] for field in ("lock", "package", "version", "advisory"))
        if scope in seen:
            raise ValueError("Advisory exception scope is repeated.")
        seen.add(scope)
        records.append(row)
    return tuple(records)


def conda_lock_decision(
    repo_root: Path,
    relative: str,
    matches: list[dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
    prefix: Path,
    read: ReadUrl = read_url,
) -> dict[str, Any]:
    """Keep every finding visible and verify named vendor-source evidence at use.

    Parameters
    ----------
    repo_root : Path
        Checkout containing the reviewed exception table.
    relative : str
        Audited lock path within the checkout.
    matches : list of dict
        All findings from the validated native Grype report.
    artifacts : dict of str to dict
        Exact Syft package identities already bound to the lock.
    prefix : Path
        Installed environment whose physical vendor source is read.
    read : callable, optional
        Reader of the pinned vendor addresses, bounded by a byte limit.

    Returns
    -------
    dict
        Accepted and blocked findings, unused records and the blocking result.

    Raises
    ------
    OSError, ValueError
        The table, installed proof, immutable vendor source or channel index is
        unavailable.
    """
    raw = (repo_root / EXCEPTION_FILE).read_bytes()
    records = load_exceptions(raw)
    known = {lock.path for lock in inventory_locks(repo_root) if lock.ecosystem == "pixi"}
    if not {record["lock"] for record in records}.issubset(known):
        raise ValueError("Advisory exception names an unaudited lock.")
    accepted: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    used: set[int] = set()
    for match in matches:
        artifact, advisory = match["artifact"], match["vulnerability"]
        chosen = [
            (number, record)
            for number, record in enumerate(records)
            if (record["lock"], record["package"], record["version"], record["advisory"])
            == (relative, artifact["name"], artifact["version"], advisory["id"])
        ]
        if not chosen:
            blocked.append(match)
            continue
        number, record = chosen[0]
        used.add(number)
        native = artifacts[artifact["id"]]
        path = prefix / record["proof_path"]
        fix = advisory.get("fix")
        holds = (
            advisory.get("namespace") == "nvd:cpe"
            and isinstance(fix, dict)
            and fix.get("versions") == [record["fixed_version"]]
            and native["metadata"]["sha256"] == record["artifact_sha256"]
            and path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(prefix.resolve())
        )
        digest = ""
        if holds:
            with path.open("rb") as stream:
                installed = stream.read(_SOURCE_LIMIT + 1)
            digest = hashlib.sha256(installed).hexdigest()
            holds = len(installed) <= _SOURCE_LIMIT and digest == record["proof_sha256"]
            if holds:
                upstream = read(record["source"], _SOURCE_LIMIT + 1)
                holds = len(upstream) <= _SOURCE_LIMIT and upstream == installed
        evidence: dict[str, Any] = {
            "holds": holds,
            "source": record["source"],
            "proof_sha256": digest,
        }
        if record["reason"] == "vendor_fix_unreleased":
            newer: list[str] = []
            if holds:
                index = read(CONDA_FORGE_PYTHON_INDEX, _INDEX_LIMIT + 1)
                if len(index) > _INDEX_LIMIT:
                    raise ValueError("Vendor channel index exceeds its size limit.")
                newer = newer_vendor_releases(index, record["version"])
                holds = not newer
            evidence.update(holds=holds, channel=CONDA_FORGE_PYTHON_INDEX, newer_releases=newer)
        if holds:
            accepted.append({"finding": match, "reason": record["reason"], "evidence": evidence})
        else:
            blocked.append(match)
    unused = [
        record
        for number, record in enumerate(records)
        if record["lock"] == relative and number not in used
    ]
    if (repo_root / EXCEPTION_FILE).read_bytes() != raw:
        raise ValueError("Advisory exception table changed during its verification.")
    return {
        "passed": not blocked and not unused,
        "accepted": accepted,
        "blocked": blocked,
        "unused": unused,
        "exception_file_sha256": hashlib.sha256(raw).hexdigest(),
    }
