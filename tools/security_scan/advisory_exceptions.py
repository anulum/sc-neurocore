# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact advisory exceptions with installed fixed-code evidence

"""Accept named Grype matches only while exact installed vendor fixes remain present."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any
import urllib.request

from packaging.version import Version
from tools.security_scan.locked_dependency_inventory import inventory_locks

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

EXCEPTION_FILE = Path("tools/security_scan/advisory_exceptions.toml")
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


def load_exceptions(raw: bytes) -> tuple[dict[str, str], ...]:
    """Read exact, dated Python fixed-code exception records.

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
        if row["package"] != "python" or row["reason"] != "fixed_code_present":
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
        if (
            re.fullmatch(
                r"https://raw\.githubusercontent\.com/python/cpython/[0-9a-f]{40}/Lib/poplib\.py",
                row["source"],
            )
            is None
            or row["advisory"] != "CVE-2025-15367"
        ):
            raise ValueError("Advisory exception vendor source is unsupported.")
        if datetime.fromisoformat(row["review_by"]).date() < datetime.now(timezone.utc).date():
            raise ValueError("Advisory exception requires renewed review.")
        if Version(row["version"]) < Version(row["fixed_version"]):
            raise ValueError("Advisory exception version predates the vendor fix.")
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
) -> dict[str, Any]:
    """Keep every finding visible and verify named fixed-code evidence at use.

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
        Installed environment whose physical fixed source is read.

    Returns
    -------
    dict
        Accepted and blocked findings, unused records and the blocking result.

    Raises
    ------
    OSError, ValueError
        The table, installed proof or immutable vendor source is unavailable.
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
                installed = stream.read(65537)
            digest = hashlib.sha256(installed).hexdigest()
            holds = len(installed) <= 65536 and digest == record["proof_sha256"]
            if holds:
                # load_exceptions pins HTTPS and an immutable CPython source path.
                with urllib.request.urlopen(record["source"], timeout=30) as response:  # nosec B310
                    upstream = response.read(65537)
                holds = len(upstream) <= 65536 and upstream == installed
        evidence = {"holds": holds, "source": record["source"], "proof_sha256": digest}
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
