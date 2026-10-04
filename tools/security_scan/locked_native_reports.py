# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete npm identities and native audit responses

"""Validate exact npm lock identities and complete npm and Rust audit reports."""

from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

_NUMBER = r"(?:0|[1-9][0-9]*)"
_PRERELEASE = rf"(?:{_NUMBER}|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
_VERSION = re.compile(
    rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}"
    rf"(?:-{_PRERELEASE}(?:\.{_PRERELEASE})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_NAME = re.compile(r"(?:@[a-z0-9._~-]+/)?[a-z0-9._~-]+")


def npm_lock_packages(raw: bytes) -> tuple[dict[str, str], ...]:
    """Bind every installed npm identity before a scanner can silently omit it.

    Parameters
    ----------
    raw : bytes
        Complete version-2 or version-3 package-lock.json input.

    Returns
    -------
    tuple of dict
        Installation paths and exact registry package names and versions.
        Repeated installations remain separate for report count validation.

    Raises
    ------
    ValueError
        An entry is unversioned, linked, empty or lacks a public npm registry
        identity. Such inputs need a separately reviewed advisory adapter.
    """
    payload = json.loads(raw)
    if not isinstance(payload, dict) or type(payload.get("lockfileVersion")) is not int:
        raise ValueError("npm lock format is unavailable.")
    packages = payload.get("packages")
    if payload["lockfileVersion"] not in (2, 3) or not isinstance(packages, dict):
        raise ValueError("npm audit requires a version-2 or version-3 package inventory.")
    if not isinstance(packages.get(""), dict):
        raise ValueError("npm lock root identity is unavailable.")
    identities: list[dict[str, str]] = []
    for path, entry in sorted(packages.items()):
        if path == "":
            continue
        if not path.startswith("node_modules/") or not isinstance(entry, dict) or entry.get("link"):
            raise ValueError("npm linked or nonregistry packages need reviewed identities.")
        name = entry.get("name", path.rsplit("node_modules/", 1)[-1])
        version, resolved = entry.get("version"), entry.get("resolved")
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            raise ValueError("npm package name is unavailable.")
        if not isinstance(version, str) or _VERSION.fullmatch(version) is None:
            raise ValueError("npm package requires an exact version identity.")
        if not isinstance(resolved, str):
            raise ValueError("npm registry origin is unavailable.")
        origin = urlsplit(resolved)
        if origin.scheme != "https" or origin.netloc != "registry.npmjs.org":
            raise ValueError("npm nonregistry packages need a reviewed advisory adapter.")
        identities.append({"path": path, "name": name, "version": version})
    if not identities:
        raise ValueError("npm dependency inventory is empty.")
    return tuple(identities)


def npm_report_passed(payload: object, package_count: int) -> bool:
    """Require complete npm package coverage and zero findings of every severity.

    Parameters
    ----------
    payload : object
        JSON response returned by the real npm audit command.
    package_count : int
        Number of validated installation identities in the committed lock.

    Returns
    -------
    bool
        True only for a complete version-2 report with no findings or errors.

    Raises
    ------
    ValueError
        Report format, severity counters or complete package coverage is invalid.
        Boolean values cannot masquerade as integer zero counters.
    """
    if not isinstance(payload, dict) or type(payload.get("auditReportVersion")) is not int:
        raise ValueError("npm audit report version is unavailable.")
    metadata = payload.get("metadata")
    if payload["auditReportVersion"] != 2 or not isinstance(metadata, dict):
        raise ValueError("npm audit report format is invalid.")
    counts, dependencies = metadata.get("vulnerabilities"), metadata.get("dependencies")
    if not isinstance(counts, dict) or not isinstance(payload.get("vulnerabilities"), dict):
        raise ValueError("npm audit findings are unavailable.")
    severities = ("info", "low", "moderate", "high", "critical")
    if any(type(counts.get(key)) is not int or counts[key] < 0 for key in (*severities, "total")):
        raise ValueError("npm severity counters are invalid.")
    if counts["total"] != sum(counts[key] for key in severities):
        raise ValueError("npm severity counters are inconsistent.")
    if (
        type(package_count) is not int
        or package_count <= 0
        or not isinstance(dependencies, dict)
        or type(dependencies.get("total")) is not int
        or dependencies["total"] != package_count
    ):
        raise ValueError("npm audit does not cover the complete lock package inventory.")
    return counts["total"] == 0 and not payload["vulnerabilities"] and "error" not in payload


def rust_report_passed(payload: object) -> bool:
    """Refuse suppressed Rust audits and malformed or inconsistent findings.

    Parameters
    ----------
    payload : object
        JSON report returned by cargo audit with warnings denied.

    Returns
    -------
    bool
        True only for an unfiltered report with zero advisories and warnings.

    Raises
    ------
    ValueError
        Settings, warning coverage or finding counters are incomplete or invalid.
    """
    if not isinstance(payload, dict):
        raise ValueError("Rust audit JSON root is invalid.")
    findings, settings, warnings = (
        payload.get("vulnerabilities"),
        payload.get("settings"),
        payload.get("warnings"),
    )
    if not isinstance(settings, dict) or any(
        settings.get(key) != [] for key in ("ignore", "target_arch", "target_os")
    ):
        raise ValueError("Rust advisory coverage must not be suppressed or filtered.")
    if "severity" not in settings or settings["severity"] is not None:
        raise ValueError("Rust advisory severity must not be filtered.")
    informational = settings.get("informational_warnings")
    if not isinstance(informational, list) or not all(
        warning in informational for warning in ("unmaintained", "unsound", "notice")
    ):
        raise ValueError("Rust audit warning coverage is incomplete.")
    if not isinstance(warnings, dict) or any(
        not isinstance(rows, list) for rows in warnings.values()
    ):
        raise ValueError("Rust audit warning lists are unavailable.")
    if not isinstance(findings, dict) or not isinstance(findings.get("list"), list):
        raise ValueError("Rust audit findings are unavailable.")
    count, found = findings.get("count"), findings.get("found")
    if type(count) is not int or count < 0 or type(found) is not bool:
        raise ValueError("Rust audit finding counters are invalid.")
    if count != len(findings["list"]) or found != (count > 0):
        raise ValueError("Rust audit finding counters are inconsistent.")
    return count == 0 and not any(warnings.values())
