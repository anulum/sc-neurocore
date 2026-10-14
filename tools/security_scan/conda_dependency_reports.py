# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Conda scanner input and report validation

"""Bind native Syft identities to the lock and require unfiltered Grype results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml

SYFT_VERSION = "1.54.1"
GRYPE_VERSION = "0.120.1"
GRYPE_DATABASE_URL = "https://grype.anchore.io/databases"


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Native Conda report object is unavailable.")
    return value


def conda_lock_artifacts(lock_raw: bytes) -> dict[str, tuple[str, str, str, str]]:
    """Extract every exact artifact identity from the supported Pixi lock.

    Parameters
    ----------
    lock_raw : bytes
        Committed lock bytes with one default linux-64 environment.

    Returns
    -------
    dict of str to tuple
        Artifact URL to SHA-256, package name, version and build.

    Raises
    ------
    ValueError, yaml.YAMLError
        Format, selection, archive identity or digest is incomplete.
    """
    lock = yaml.safe_load(lock_raw)
    if not isinstance(lock, dict) or not isinstance(lock.get("packages"), list):
        raise ValueError("Pixi lock package inventory is unavailable.")
    environments = lock.get("environments")
    if not isinstance(environments, dict) or set(environments) != {"default"}:
        raise ValueError("Conda audit requires one default environment.")
    platforms = _object(environments["default"]).get("packages")
    if not isinstance(platforms, dict) or set(platforms) != {"linux-64"}:
        raise ValueError("Conda audit requires the locked linux-64 platform.")
    expected: dict[str, tuple[str, str, str, str]] = {}
    for row in lock["packages"]:
        if not isinstance(row, dict) or not isinstance(row.get("conda"), str):
            raise ValueError("Conda audit requires exact Conda artifact identities.")
        url, digest = row["conda"], row.get("sha256")
        if (
            not isinstance(digest, str)
            or re.fullmatch("[0-9a-f]{64}", digest) is None
            or (url in expected or urlsplit(url).scheme != "https")
        ):
            raise ValueError("Conda lock artifact digest or identity is invalid.")
        filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        if filename.endswith(".conda"):
            stem = filename.removesuffix(".conda")
        elif filename.endswith(".tar.bz2"):
            stem = filename.removesuffix(".tar.bz2")
        else:
            raise ValueError("Conda lock archive type is unsupported.")
        parts = stem.rsplit("-", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError("Conda lock archive identity is invalid.")
        name, version, build = parts
        expected[url] = (digest, name, version, build)
    selected = platforms["linux-64"]
    if (
        not isinstance(selected, list)
        or not expected
        or any(
            not isinstance(row, dict)
            or set(row) != {"conda"}
            or (not isinstance(row["conda"], str))
            for row in selected
        )
    ):
        raise ValueError("Conda environment package selection is invalid.")
    urls = [row["conda"] for row in selected]
    if len(set(urls)) != len(urls) or set(urls) != set(expected):
        raise ValueError("Conda environment does not cover the complete lock.")
    return expected


def syft_lock_inventory(
    payload: object, lock_raw: bytes, prefix: Path
) -> dict[str, dict[str, Any]]:
    """Require every locked Conda artifact in a native installed-environment SBOM.

    Parameters
    ----------
    payload : object
        Complete Syft JSON response from the Conda metadata cataloger.
    lock_raw : bytes
        Exact committed Pixi lock bytes.
    prefix : Path
        Installed default environment reported by native Pixi.

    Returns
    -------
    dict of str to dict
        Native artifact records, keyed by their unique package identifiers.

    Raises
    ------
    ValueError, yaml.YAMLError
        Scanner identity, source, package identity or lock coverage is invalid.
        Additional environments, platforms or PyPI records require an adapter.
    """
    expected = conda_lock_artifacts(lock_raw)
    if not isinstance(payload, dict):
        raise ValueError("Syft report is invalid.")
    descriptor, source = payload.get("descriptor"), payload.get("source")
    if not isinstance(descriptor, dict) or (descriptor.get("name"), descriptor.get("version")) != (
        "syft",
        SYFT_VERSION,
    ):
        raise ValueError("Syft report does not identify the pinned scanner.")
    if (
        not isinstance(source, dict)
        or source.get("type") != "directory"
        or (_object(source.get("metadata")).get("path") != str(prefix))
    ):
        raise ValueError("Syft source does not identify the installed environment.")
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Syft inventory does not cover the complete lock.")
    artifacts: dict[str, dict[str, Any]] = {}
    observed: dict[str, str] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("type") != "conda"
            or (row.get("foundBy") != "conda-meta-cataloger")
        ):
            raise ValueError("Syft package was not read from Conda metadata.")
        identity, metadata = row.get("id"), row.get("metadata")
        if (
            not isinstance(identity, str)
            or not identity
            or identity in artifacts
            or (not isinstance(metadata, dict))
        ):
            raise ValueError("Syft package identifier or metadata is invalid.")
        url, digest = metadata.get("url"), metadata.get("sha256")
        if (
            not isinstance(url, str)
            or url not in expected
            or url in observed
            or (digest != expected[url][0])
        ):
            raise ValueError("Syft package differs from its locked artifact.")
        if any(
            not isinstance(row.get(field), str)
            or not row[field]
            or (row[field] != metadata.get(field))
            for field in ("name", "version")
        ):
            raise ValueError("Syft package name or version differs from its metadata.")
        if (row["name"], row["version"], metadata.get("build")) != expected[url][1:]:
            raise ValueError("Syft package differs from its locked archive identity.")
        cpes = row.get("cpes")
        if (
            not isinstance(cpes, list)
            or not cpes
            or any(
                not isinstance(cpe, dict)
                or not isinstance(cpe.get("cpe"), str)
                or not cpe["cpe"].startswith("cpe:2.3:")
                for cpe in cpes
            )
        ):
            raise ValueError("Syft package has no usable CPE identity.")
        observed[url] = digest
        artifacts[identity] = row
    return artifacts


def grype_matches(
    payload: object, artifacts: dict[str, dict[str, Any]], prefix: Path
) -> list[dict[str, Any]]:
    """Validate native Grype policy and return every unsuppressed match.

    Parameters
    ----------
    payload : object
        Complete native Grype JSON response.
    artifacts : dict of str to dict
        Validated native Syft artifact identifiers and package records.
    prefix : Path
        Installed environment used to generate the input SBOM.

    Returns
    -------
    list of dict
        Every native match; no severity, fix-state or reachability filter.

    Raises
    ------
    ValueError
        Report source, settings, database validity or match identities disagree.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        raise ValueError("Grype match report is unavailable.")
    descriptor = payload.get("descriptor")
    if not isinstance(descriptor, dict) or (descriptor.get("name"), descriptor.get("version")) != (
        "grype",
        GRYPE_VERSION,
    ):
        raise ValueError("Grype report does not identify the pinned scanner.")
    if payload.get("source") != {"type": "directory", "target": str(prefix)}:
        raise ValueError("Grype source differs from the installed environment.")
    cfg = descriptor.get("configuration")
    if not isinstance(cfg, dict) or any(
        cfg.get(key) != value
        for key, value in {
            "ignore": [],
            "exclude": [],
            "only-fixed": False,
            "only-notfixed": False,
            "ignore-wontfix": "",
            "fail-on-severity": "",
            "vex-documents": [],
            "vex-add": [],
            "show-suppressed": True,
            "include-matcher-suppressions": True,
            "match-upstream-kernel-headers": True,
        }.items()
    ):
        raise ValueError("Grype report contains filtered or suppressed settings.")
    if _object(_object(cfg.get("match")).get("stock")).get("using-cpes") is not True:
        raise ValueError("Grype Conda CPE matching is unavailable.")
    db = cfg.get("db")
    if not isinstance(db, dict) or any(
        db.get(key) != value
        for key, value in {
            "auto-update": True,
            "validate-by-hash-on-start": True,
            "validate-age": True,
            "require-update-check": True,
            "max-allowed-built-age": 432000000000000,
            "max-update-check-frequency": 0,
            "update-url": GRYPE_DATABASE_URL,
        }.items()
    ):
        raise ValueError("Grype database freshness or origin settings are invalid.")
    database = descriptor.get("db")
    status = database.get("status") if isinstance(database, dict) else None
    if (
        not isinstance(status, dict)
        or status.get("valid") is not True
        or (
            not isinstance(status.get("from"), str)
            or not status["from"].startswith(GRYPE_DATABASE_URL + "/")
        )
    ):
        raise ValueError("Grype database is not a valid official snapshot.")
    built = status.get("built")
    if not isinstance(built, str):
        raise ValueError("Grype database timestamp is unavailable.")
    when = datetime.fromisoformat(built.replace("Z", "+00:00"))
    if (
        when.tzinfo is None
        or not 0 <= (datetime.now(timezone.utc) - when).total_seconds() <= 432000
    ):
        raise ValueError("Grype database timestamp is outside the audit freshness bound.")
    if payload.get("ignoredMatches", []) != []:
        raise ValueError("Grype report contains ignored matches.")
    rows = payload.get("matches")
    if not isinstance(rows, list):
        raise ValueError("Grype match report is unavailable.")
    for row in rows:
        artifact = row.get("artifact") if isinstance(row, dict) else None
        vulnerability = row.get("vulnerability") if isinstance(row, dict) else None
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("id"), str)
            or (artifact["id"] not in artifacts)
            or (
                not isinstance(vulnerability, dict)
                or not isinstance(vulnerability.get("id"), str)
                or not vulnerability["id"]
            )
        ):
            raise ValueError("Grype finding has an unknown artifact or advisory.")
        actual = artifacts[artifact["id"]]
        if any(artifact.get(key) != actual[key] for key in ("name", "version", "type")):
            raise ValueError("Grype finding package differs from its Syft identity.")
    return rows
