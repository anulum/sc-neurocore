# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact npm advisory exceptions for development-only packages without a released fix

"""Accept a named npm advisory only while its package is development-only and has no released fix.

One reason exists. ``dev_only_fix_unreleased`` accepts, for at most two weeks per
review, an npm audit advisory on one package of one lock when:

- every installation of that package in the lock is the recorded version and is
  marked as a development dependency, which npm installs only for development;
- the advisory, as npm reports it, names that package and exactly the recorded
  vulnerable range, of the form ``<=X.Y.Z`` or ``<X.Y.Z``;
- the npm registry lists no final release outside that range.

Every other advisory still blocks the audit, every finding must rest on an
advisory, and a record that matches no finding blocks it too, so a stale record
cannot stay. The OSV lane of the security packet applies the same records to the
same lock, through the package, version, identifier and affected range OSV reports.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tomllib
from typing import Any, cast
from urllib.parse import quote

from tools.security_scan.advisory_exceptions import ReadUrl, read_url
from tools.security_scan.locked_native_reports import npm_lock_packages, npm_report_passed

EXCEPTION_FILE = Path("tools/security_scan/npm_advisory_exceptions.toml")
NPM_REGISTRY = "https://registry.npmjs.org/"
REVIEW_WINDOW = timedelta(days=14)
_DOCUMENT_LIMIT = 32 * 1024 * 1024
_REASON = "dev_only_fix_unreleased"
# The package name and exact version forms that npm_lock_packages accepts from a lock.
_NUMBER = r"(?:0|[1-9][0-9]*)"
_PRERELEASE = rf"(?:{_NUMBER}|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
_VERSION = re.compile(
    rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}"
    rf"(?:-{_PRERELEASE}(?:\.{_PRERELEASE})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_NAME = re.compile(r"(?:@[a-z0-9._~-]+/)?[a-z0-9._~-]+")
_ADVISORY = re.compile(r"GHSA(?:-[23456789cfghjmpqrvwx]{4}){3}")
_RANGE = re.compile(r"(<=?)((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))")
_FIELDS = frozenset(
    {
        "lock",
        "package",
        "version",
        "advisory",
        "vulnerable_range",
        "reason",
        "review_by",
        "rationale",
    }
)


def _final(version: str) -> tuple[int, int, int] | None:
    """Return the numeric triple of a final release, or None for a prerelease.

    Raises
    ------
    ValueError
        The text is not an exact npm version.
    """
    if _VERSION.fullmatch(version) is None:
        raise ValueError("npm version is not exact.")
    core = version.split("+", 1)[0]
    if "-" in core:
        return None
    major, minor, patch = (int(part) for part in core.split("."))
    return major, minor, patch


def in_range(version: str, vulnerable_range: str) -> bool:
    """Tell whether a final npm release falls inside a supported vulnerable range.

    Parameters
    ----------
    version : str
        Exact npm version of a final release.
    vulnerable_range : str
        ``<=X.Y.Z`` or ``<X.Y.Z``, the only forms a record may name.

    Returns
    -------
    bool
        True when the release is at or below (``<=``) or below (``<``) the bound.

    Raises
    ------
    ValueError
        The range form is unsupported or the version is not a final release.
    """
    match = _RANGE.fullmatch(vulnerable_range)
    if match is None:
        raise ValueError("npm advisory range form is unsupported.")
    release, bound = _final(version), _final(match.group(2))
    if release is None or bound is None:
        raise ValueError("npm advisory range needs final releases.")
    return release <= bound if match.group(1) == "<=" else release < bound


def released_fixes(document: bytes, vulnerable_range: str) -> list[str]:
    """Return the final releases in an npm registry document that lie outside the range.

    Parameters
    ----------
    document : bytes
        JSON package document from the npm registry; its ``versions`` member maps every
        published version.
    vulnerable_range : str
        Vulnerable range named by the exception record.

    Returns
    -------
    list of str
        Final releases outside the range; prereleases are not installable fixes.

    Raises
    ------
    ValueError
        The document lists no versions, or a version is not exact.
    """
    payload = json.loads(document)
    versions = payload.get("versions") if isinstance(payload, dict) else None
    if not isinstance(versions, dict) or not versions:
        raise ValueError("npm registry document has no version list.")
    fixes = [
        version
        for version in versions
        if _final(version) is not None and not in_range(version, vulnerable_range)
    ]
    return sorted(fixes, key=lambda version: _final(version) or (0, 0, 0))


def load_npm_exceptions(raw: bytes, today: date) -> tuple[dict[str, str], ...]:
    """Read exact, dated npm advisory exception records.

    Parameters
    ----------
    raw : bytes
        Complete UTF-8 TOML table, including an empty exception list.
    today : date
        The UTC date against which each review date is checked.

    Returns
    -------
    tuple of dict
        Validated records keyed by lock, package, version and advisory.

    Raises
    ------
    ValueError
        A field, path, name, version, advisory, range, review date or the record
        cardinality is invalid.
    """
    payload = tomllib.loads(raw.decode("utf-8"))
    rows = payload.get("exception")
    if set(payload) != {"exception"} or not isinstance(rows, list):
        raise ValueError("npm advisory exception table is invalid.")
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
            raise ValueError("npm advisory exception fields are invalid.")
        lock = PurePosixPath(row["lock"])
        if (
            lock.is_absolute()
            or ".." in lock.parts
            or lock.as_posix() != row["lock"]
            or lock.name != "package-lock.json"
        ):
            raise ValueError("npm advisory exception lock must be a relative package-lock.json.")
        if _NAME.fullmatch(row["package"]) is None or _final(row["version"]) is None:
            raise ValueError("npm advisory exception package or version is invalid.")
        if row["reason"] != _REASON or _ADVISORY.fullmatch(row["advisory"]) is None:
            raise ValueError("npm advisory exception reason or advisory is unsupported.")
        if not in_range(row["version"], row["vulnerable_range"]):
            raise ValueError("npm advisory exception version lies outside its vulnerable range.")
        review_by = datetime.fromisoformat(row["review_by"]).date()
        if review_by < today:
            raise ValueError("npm advisory exception requires renewed review.")
        if review_by > today + REVIEW_WINDOW:
            raise ValueError("npm advisory exception review date exceeds two weeks.")
        scope = tuple(row[field] for field in ("lock", "package", "version", "advisory"))
        if scope in seen:
            raise ValueError("npm advisory exception scope is repeated.")
        seen.add(scope)
        records.append(row)
    return tuple(records)


def _advisories(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect every advisory object of an npm audit report whose findings all rest on one.

    Raises
    ------
    ValueError
        The report holds no finding, its total counter disagrees with its findings, a
        finding's ``via`` list holds something other than advisory objects and names
        of other findings, or a finding reaches no advisory through those names.
    """
    findings = payload.get("vulnerabilities")
    metadata = payload.get("metadata")
    counts = metadata.get("vulnerabilities") if isinstance(metadata, dict) else None
    if not isinstance(findings, dict) or not findings:
        raise ValueError("npm report holds no finding to decide.")
    if not isinstance(counts, dict) or counts.get("total") != len(findings):
        raise ValueError("npm finding counters disagree with the findings.")
    advisories: list[dict[str, Any]] = []
    grounded: set[str] = set()
    for name, finding in findings.items():
        via = finding.get("via") if isinstance(finding, dict) else None
        if not isinstance(via, list) or not via:
            raise ValueError("npm finding names no cause.")
        for cause in via:
            if isinstance(cause, dict):
                advisories.append(cause)
                grounded.add(name)
            elif not isinstance(cause, str) or cause not in findings:
                raise ValueError("npm finding names an unknown cause.")
    # A finding that names only other findings rests on an advisory through one of them.
    growing = True
    while growing:
        reached = {
            name
            for name, finding in findings.items()
            if name not in grounded
            and any(cause in grounded for cause in finding["via"] if isinstance(cause, str))
        }
        grounded |= reached
        growing = bool(reached)
    if grounded != set(findings):
        raise ValueError("npm finding reaches no advisory.")
    return advisories


def _record_evidence(
    record: dict[str, str], packages: dict[str, Any], read: ReadUrl
) -> tuple[bool, dict[str, Any]]:
    """Check one record against the lock's installations and the registry's releases.

    Returns
    -------
    tuple of bool and dict
        Whether every installation is the recorded development-only version while the
        registry lists no fix, and the installations and released fixes found.
    """
    installations = [
        {"path": path, "version": entry.get("version"), "dev": entry.get("dev") is True}
        for path, entry in sorted(packages.items())
        if path and entry.get("name", path.rsplit("node_modules/", 1)[-1]) == record["package"]
    ]
    fixes = released_fixes(
        read(NPM_REGISTRY + quote(record["package"], safe="@"), _DOCUMENT_LIMIT + 1),
        record["vulnerable_range"],
    )
    holds = (
        bool(installations)
        and all(item["version"] == record["version"] and item["dev"] for item in installations)
        and not fixes
    )
    return holds, {"installations": installations, "released_fixes": fixes}


def npm_lock_decision(
    repo_root: Path,
    relative: str,
    lock_raw: bytes,
    payload: dict[str, Any],
    *,
    read: ReadUrl = read_url,
    today: date | None = None,
) -> dict[str, Any]:
    """Keep every npm finding visible and verify the named exceptions at use.

    Parameters
    ----------
    repo_root : Path
        Checkout containing the reviewed exception table.
    relative : str
        Audited lock path within the checkout.
    lock_raw : bytes
        The lock bytes the audit validated and submitted.
    payload : dict
        npm audit report already validated by ``npm_report_passed``.
    read : callable, optional
        Reader of the npm registry document, bounded by a byte limit.
    today : date, optional
        UTC date for the review window; the current one by default.

    Returns
    -------
    dict
        Accepted and blocked advisories, unused records and the blocking result.

    Raises
    ------
    OSError, ValueError
        The table, the lock, the report or a registry document is unavailable
        or invalid.
    """
    day = today or datetime.now(timezone.utc).date()
    raw = (repo_root / EXCEPTION_FILE).read_bytes()
    records = [record for record in load_npm_exceptions(raw, day) if record["lock"] == relative]
    if "error" in payload:
        raise ValueError("npm audit reported a provider error.")
    packages = json.loads(lock_raw)["packages"]
    accepted: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    used: set[int] = set()
    for advisory in _advisories(payload):
        url = advisory.get("url")
        identifier = url.rsplit("/", 1)[-1] if isinstance(url, str) else ""
        chosen = [
            (number, record)
            for number, record in enumerate(records)
            if url == f"https://github.com/advisories/{record['advisory']}"
            and advisory.get("name") == record["package"]
            and advisory.get("range") == record["vulnerable_range"]
        ]
        if not chosen:
            blocked.append({"advisory": identifier, "package": advisory.get("name")})
            continue
        number, record = chosen[0]
        holds, evidence = _record_evidence(record, packages, read)
        if holds:
            used.add(number)
            accepted.append(
                {
                    "advisory": record["advisory"],
                    "package": record["package"],
                    "reason": record["reason"],
                    "review_by": record["review_by"],
                    "evidence": evidence,
                }
            )
        else:
            blocked.append(
                {"advisory": identifier, "package": record["package"], "evidence": evidence}
            )
    unused = [record for number, record in enumerate(records) if number not in used]
    if (repo_root / EXCEPTION_FILE).read_bytes() != raw:
        raise ValueError("npm advisory exception table changed during its verification.")
    return {
        # A lock without a record never passes on findings, whatever the report holds.
        "passed": bool(records) and bool(accepted) and not blocked and not unused,
        "accepted": accepted,
        "blocked": blocked,
        "unused": unused,
        "exception_file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def npm_audit_passed(
    repo_root: Path,
    relative: str,
    lock_raw: bytes,
    rc: int,
    payload: object,
    directory: Path,
    *,
    read: ReadUrl = read_url,
    today: date | None = None,
) -> bool:
    """Pass a complete npm audit with no findings, or with only named, verified exceptions.

    Parameters
    ----------
    repo_root : Path
        Checkout containing the reviewed exception table.
    relative : str
        Audited lock path within the checkout.
    lock_raw : bytes
        The lock bytes the audit validated and submitted.
    rc : int
        Exit status of npm audit; 1 is its status for reported findings.
    payload : object
        JSON report returned by npm audit.
    directory : Path
        Packet directory that receives the exception decision when findings exist.
    read : callable, optional
        Reader of the npm registry document, bounded by a byte limit.
    today : date, optional
        UTC date for the review window; the current one by default.

    Returns
    -------
    bool
        True for a clean report, or for findings that every one stem from a named,
        verified exception while every record of this lock is used.

    Raises
    ------
    OSError, ValueError
        The report is incomplete, or the exception evidence is unavailable.
    """
    if npm_report_passed(payload, len(npm_lock_packages(lock_raw))):
        return rc == 0
    # npm_report_passed raises for every root that is not a complete report object.
    report = cast(dict[str, Any], payload)
    decision = npm_lock_decision(repo_root, relative, lock_raw, report, read=read, today=today)
    (directory / "npm-advisory-exceptions.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    return decision["passed"] is True and rc == 1


def _osv_range(vulnerability: dict[str, Any], package: str) -> str | None:
    """Return the one affected npm range OSV states for ``package``, in record form.

    Only a single SEMVER range from 0 to a last affected release (``<=X.Y.Z``) or to a
    fixed release (``<X.Y.Z``) is expressed; anything else gives None.
    """
    entries = vulnerability.get("affected")
    for affected in entries if isinstance(entries, list) else []:
        if not isinstance(affected, dict):
            continue
        name = affected.get("package")
        if not isinstance(name, dict) or (name.get("ecosystem"), name.get("name")) != (
            "npm",
            package,
        ):
            continue
        ranges = affected.get("ranges")
        if not isinstance(ranges, list) or len(ranges) != 1 or not isinstance(ranges[0], dict):
            return None
        events = ranges[0].get("events")
        if ranges[0].get("type") != "SEMVER" or not isinstance(events, list) or len(events) != 2:
            return None
        start, end = events
        if start != {"introduced": "0"} or not isinstance(end, dict) or len(end) != 1:
            return None
        if isinstance(end.get("last_affected"), str):
            return f"<={end['last_affected']}"
        if isinstance(end.get("fixed"), str):
            return f"<{end['fixed']}"
        return None
    return None


def osv_npm_decision(
    repo_root: Path,
    report: dict[str, Any],
    *,
    read: ReadUrl = read_url,
    today: date | None = None,
) -> dict[str, Any]:
    """Decide OSV findings with the npm records: accept only those a record names and proves.

    Parameters
    ----------
    repo_root : Path
        Checkout containing the reviewed exception table and the scanned locks.
    report : dict
        OSV-Scanner JSON report whose ``results`` list was validated by the runner.
    read : callable, optional
        Reader of the npm registry document, bounded by a byte limit.
    today : date, optional
        UTC date for the review window; the current one by default.

    Returns
    -------
    dict
        Accepted and blocked vulnerabilities and the blocking result. Every finding
        outside an npm lock record, or whose record does not hold, is blocked, and so
        is every result, package or vulnerability entry the decision cannot read.

    Raises
    ------
    OSError, ValueError
        The table, a lock or a registry document is unavailable or invalid, or the
        table changed during the decision.
    """
    day = today or datetime.now(timezone.utc).date()
    raw = (repo_root / EXCEPTION_FILE).read_bytes()
    records = load_npm_exceptions(raw, day)
    root = repo_root.resolve()
    locks: dict[str, dict[str, Any]] = {}
    accepted: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    # An entry this decision cannot read is blocked, never skipped: an exit-1 report must
    # not pass with a finding nobody examined.
    for result in report["results"]:
        if not isinstance(result, dict) or not isinstance(result.get("packages", []), list):
            blocked.append({"id": None, "package": None, "lock": None, "unreadable": "result"})
            continue
        source = result.get("source")
        path = source.get("path") if isinstance(source, dict) else None
        relative = None
        if isinstance(path, str) and Path(path).resolve().is_relative_to(root):
            relative = Path(path).resolve().relative_to(root).as_posix()
        for package in result.get("packages", []):
            if not isinstance(package, dict) or not isinstance(
                package.get("vulnerabilities", []), list
            ):
                blocked.append(
                    {"id": None, "package": None, "lock": relative, "unreadable": "package"}
                )
                continue
            named = package.get("package")
            info: dict[str, Any] = named if isinstance(named, dict) else {}
            for vulnerability in package.get("vulnerabilities", []):
                identifier = vulnerability.get("id") if isinstance(vulnerability, dict) else None
                finding = {"id": identifier, "package": info.get("name"), "lock": relative}
                if (
                    not isinstance(vulnerability, dict)
                    or not isinstance(identifier, str)
                    or not identifier
                ):
                    blocked.append({**finding, "unreadable": "vulnerability"})
                    continue
                chosen = [
                    record
                    for record in records
                    if relative == record["lock"]
                    and (info.get("ecosystem"), info.get("name"), info.get("version"))
                    == ("npm", record["package"], record["version"])
                    and identifier == record["advisory"]
                    and _osv_range(vulnerability, record["package"]) == record["vulnerable_range"]
                ]
                if not chosen:
                    blocked.append(finding)
                    continue
                record = chosen[0]
                if record["lock"] not in locks:
                    locks[record["lock"]] = json.loads((repo_root / record["lock"]).read_bytes())[
                        "packages"
                    ]
                holds, evidence = _record_evidence(record, locks[record["lock"]], read)
                (accepted if holds else blocked).append({**finding, "evidence": evidence})
    if (repo_root / EXCEPTION_FILE).read_bytes() != raw:
        raise ValueError("npm advisory exception table changed during its verification.")
    return {
        "passed": bool(accepted) and not blocked,
        "accepted": accepted,
        "blocked": blocked,
        "exception_file_sha256": hashlib.sha256(raw).hexdigest(),
    }
