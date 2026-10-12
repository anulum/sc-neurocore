# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-finding advisory exceptions verified on every audit run

"""Accept one advisory finding only while a machine check of its reason still holds.

An exception names one package version and one advisory. It carries one of two
reasons, and the audit re-establishes that reason from live sources each time:

``no_fixed_version_in_channel``
    The locked version is the newest one its channel publishes. The check
    fails as soon as the channel offers any newer version, so a published fix
    turns the audit red without anyone editing this record.

``component_not_in_package``
    The advisory concerns a program that the locked package does not ship. The
    check downloads the exact locked artefact, verifies its digest against the
    lock and requires that no file of the package matches the declared paths.

There is no severity, confidence or reachability filter, no wildcard and no
expiry by date: an exception lives exactly as long as its evidence does. An
exception that no longer matches a finding is itself a failure, so the record
cannot outlive the finding it was written for.
"""

from __future__ import annotations

import bz2
from collections.abc import Callable
from dataclasses import asdict, dataclass
import fnmatch
import hashlib
import http.client
import io
import json
from pathlib import Path
import ssl
import subprocess
import tarfile
import tomllib
from typing import Any
import urllib.parse
import zipfile

REASONS = ("no_fixed_version_in_channel", "component_not_in_package")
EXCEPTION_FILE = Path("tools/security_scan/advisory_exceptions.toml")
CHANNEL_INDEX = "https://api.anaconda.org/package/{channel}/{package}"
INDEX_LIMIT = 32 * 1024 * 1024
ARTEFACT_LIMIT = 64 * 1024 * 1024
REDIRECT_LIMIT = 5
_COMMON = {"ecosystem", "lock", "package", "version", "advisory", "reason", "recorded", "source"}
_BY_REASON = {
    "no_fixed_version_in_channel": {"channel", "subdir"},
    "component_not_in_package": {"component", "absent"},
}

Fetch = Callable[[str, int], bytes]


@dataclass(frozen=True)
class AdvisoryException:
    """One reviewed exception for one advisory against one locked package version."""

    ecosystem: str
    lock: str
    package: str
    version: str
    advisory: str
    reason: str
    recorded: str
    source: str
    channel: str = ""
    subdir: str = ""
    component: str = ""
    absent: tuple[str, ...] = ()


def load_exceptions(raw: bytes) -> tuple[AdvisoryException, ...]:
    """Parse the exception record and refuse anything it does not fully specify.

    Parameters
    ----------
    raw : bytes
        Complete committed exception file.

    Returns
    -------
    tuple of AdvisoryException
        Exceptions in file order.

    Raises
    ------
    ValueError
        A table has an unknown or missing key, an empty value, an unknown
        reason, a wildcard package, version or advisory, an empty or blanket
        path list, or repeats an earlier lock, package, version and advisory.
    """
    document = tomllib.loads(raw.decode("utf-8"))
    if set(document) - {"exception"} or not isinstance(document.get("exception", []), list):
        raise ValueError("Advisory exception file has an unknown top-level table.")
    exceptions: list[AdvisoryException] = []
    seen: set[tuple[str, str, str, str]] = set()
    for table in document.get("exception", []):
        reason = table.get("reason") if isinstance(table, dict) else None
        if reason not in REASONS:
            raise ValueError("Advisory exception has no supported reason.")
        if set(table) != _COMMON | _BY_REASON[reason]:
            raise ValueError("Advisory exception keys do not match its reason.")
        absent = table.get("absent", [])
        texts = [value for key, value in table.items() if key != "absent"]
        if not all(isinstance(value, str) and value.strip() == value and value for value in texts):
            raise ValueError("Advisory exception values must be non-empty text.")
        if any(
            "*" in table[key] or "?" in table[key] for key in ("package", "version", "advisory")
        ):
            raise ValueError("Advisory exception identity cannot contain a wildcard.")
        if reason == "component_not_in_package" and (
            not isinstance(absent, list)
            or not absent
            or not all(isinstance(item, str) and item.strip("*/? ") for item in absent)
        ):
            raise ValueError("Advisory exception needs concrete absent paths.")
        identity = (table["lock"], table["package"], table["version"], table["advisory"])
        if identity in seen:
            raise ValueError("Advisory exception is recorded twice.")
        seen.add(identity)
        exceptions.append(AdvisoryException(**{**table, "absent": tuple(absent)}))
    return tuple(exceptions)


def numeric_version(text: str) -> tuple[int, ...]:
    """Read a dotted all-numeric version for ordering.

    Parameters
    ----------
    text : str
        Version as the channel and the lock spell it.

    Returns
    -------
    tuple of int
        Components in order, comparable with another such tuple.

    Raises
    ------
    ValueError
        The version has a non-numeric component. Such versions are not
        ordered by guesswork; the exception then cannot be verified.
    """
    parts = text.split(".")
    if not parts or not all(part.isdigit() for part in parts):
        raise ValueError("Version is not dotted numeric and cannot be ordered.")
    return tuple(int(part) for part in parts)


def channel_versions(raw: bytes, subdir: str) -> tuple[str, ...]:
    """List the versions a channel publishes for one platform.

    Parameters
    ----------
    raw : bytes
        Package record of the channel index, exactly as received.
    subdir : str
        Platform directory, for example ``linux-64``.

    Returns
    -------
    tuple of str
        Distinct versions with at least one file for that platform, sorted.

    Raises
    ------
    ValueError
        The record is not the expected object or lists no file for the platform.
    """
    payload = json.loads(raw)
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        raise ValueError("Channel index record has no file list.")
    versions = {
        row["version"]
        for row in files
        if isinstance(row, dict)
        and isinstance(row.get("version"), str)
        and isinstance(row.get("attrs"), dict)
        and row["attrs"].get("subdir") == subdir
    }
    if not versions:
        raise ValueError("Channel index lists no file for the platform.")
    return tuple(sorted(versions))


def newest_version_is(versions: tuple[str, ...], locked: str) -> bool:
    """Tell whether the locked version is published and nothing newer is.

    Parameters
    ----------
    versions : tuple of str
        Versions the channel publishes for the platform.
    locked : str
        Version recorded in the lock.

    Returns
    -------
    bool
        True only if the locked version is listed and is the greatest.

    Raises
    ------
    ValueError
        A listed version cannot be ordered numerically.
    """
    if locked not in versions:
        return False
    return max(numeric_version(version) for version in versions) == numeric_version(locked)


def conda_package_paths(name: str, artefact: bytes) -> tuple[str, ...]:
    """Read the file list a conda package declares for itself.

    Parameters
    ----------
    name : str
        File name of the artefact; its suffix selects the container format.
    artefact : bytes
        Complete artefact, already verified against the lock digest.

    Returns
    -------
    tuple of str
        Paths of ``info/paths.json`` in package order.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        The container is of an unknown format, has no single metadata
        archive, cannot be decompressed, or declares no usable path list.
    """
    try:
        if name.endswith(".conda"):
            with zipfile.ZipFile(io.BytesIO(artefact)) as outer:
                members = [item for item in outer.namelist() if item.startswith("info-")]
                if len(members) != 1 or not members[0].endswith(".tar.zst"):
                    raise ValueError("Conda package has no single metadata archive.")
                compressed = outer.read(members[0])
            inner = subprocess.run(
                ["zstd", "--decompress", "--stdout"],
                input=compressed,
                capture_output=True,
                check=True,
                timeout=120,
            ).stdout
        elif name.endswith(".tar.bz2"):
            inner = bz2.decompress(artefact)
        else:
            raise ValueError("Conda package container format is unknown.")
        with tarfile.open(fileobj=io.BytesIO(inner), mode="r:") as archive:
            member = archive.extractfile("info/paths.json")
            if member is None:
                raise ValueError("Conda package metadata has no path list.")
            listing = json.loads(member.read())
    except (KeyError, zipfile.BadZipFile, tarfile.TarError) as exc:
        raise ValueError("Conda package container is unreadable.") from exc
    rows = listing.get("paths") if isinstance(listing, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Conda package declares no paths.")
    paths = tuple(
        str(row["_path"])
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("_path"), str) and row["_path"]
    )
    if len(paths) != len(rows):
        raise ValueError("Conda package path list is malformed.")
    return paths


def locked_conda_artefact(lock_packages: list[Any], package: str, version: str) -> tuple[str, str]:
    """Find the one locked artefact of a package version.

    Parameters
    ----------
    lock_packages : list
        Package records of the parsed Pixi lock.
    package : str
        Package name as the scanner reports it.
    version : str
        Locked version as the scanner reports it.

    Returns
    -------
    tuple of str
        Download address and SHA-256 recorded in the lock.

    Raises
    ------
    ValueError
        The lock holds no such artefact, more than one, or no digest for it.
    """
    prefix = f"{package}-{version}-"
    found = [
        (row["conda"], row.get("sha256"))
        for row in lock_packages
        if isinstance(row, dict)
        and isinstance(row.get("conda"), str)
        and row["conda"].rsplit("/", 1)[-1].startswith(prefix)
        and row["conda"].rsplit("/", 1)[-1][len(prefix) :].count("-") == 0
    ]
    if len(found) != 1 or not isinstance(found[0][1], str) or len(found[0][1]) != 64:
        raise ValueError("Lock does not bind exactly one artefact of the package version.")
    return found[0][0], found[0][1]


def fetch(address: str, limit: int, context: ssl.SSLContext | None = None) -> bytes:
    """Download one body over verified TLS with a size bound.

    Parameters
    ----------
    address : str
        HTTPS address of a channel index record or a locked artefact.
    limit : int
        Largest body, in bytes, that is accepted as evidence.
    context : ssl.SSLContext, optional
        Trust settings; the platform's verified defaults when omitted.

    Returns
    -------
    bytes
        The complete body.

    Raises
    ------
    ValueError, OSError
        An address on the way is not HTTPS, the transfer fails, the source
        redirects too often or answers without the body, or the body exceeds
        the limit.
    """
    trust = context or ssl.create_default_context()
    for _ in range(REDIRECT_LIMIT + 1):
        parts = urllib.parse.urlsplit(address)
        if parts.scheme != "https" or not parts.hostname:
            raise ValueError("Exception evidence is fetched over HTTPS only.")
        connection = http.client.HTTPSConnection(
            parts.hostname, parts.port, timeout=60, context=trust
        )
        try:
            target = parts.path + ("?" + parts.query if parts.query else "")
            connection.request("GET", target, headers={"User-Agent": "sc-neurocore-lock-audit"})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                address = urllib.parse.urljoin(address, response.getheader("Location") or "")
                continue
            if response.status != 200:
                raise ValueError("Exception evidence source did not return the body.")
            body = response.read(limit + 1)
        finally:
            connection.close()
        if len(body) > limit:
            raise ValueError("Exception evidence exceeds the audit evidence limit.")
        return body
    raise ValueError("Exception evidence source redirects too often.")


def verify_exception(
    exception: AdvisoryException, lock_packages: list[Any], fetcher: Fetch
) -> dict[str, Any]:
    """Re-establish the reason of one exception from live sources.

    Parameters
    ----------
    exception : AdvisoryException
        The recorded exception.
    lock_packages : list
        Package records of the parsed Pixi lock the finding belongs to.
    fetcher : callable
        Transport taking an address and a size limit and returning the body.

    Returns
    -------
    dict
        ``holds`` and the evidence it rests on: for a channel check the
        address, body digest and versions seen; for a content check the
        artefact address, its digest, the number of declared paths and the
        paths that matched. Nothing here is a judgement of severity.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        Evidence is unavailable, oversized, unverifiable or malformed. The
        caller treats every such error as an exception that does not hold.
    """
    if exception.reason == "no_fixed_version_in_channel":
        address = CHANNEL_INDEX.format(channel=exception.channel, package=exception.package)
        raw = fetcher(address, INDEX_LIMIT)
        versions = channel_versions(raw, exception.subdir)
        return {
            "holds": newest_version_is(versions, exception.version),
            "index": address,
            "index_sha256": hashlib.sha256(raw).hexdigest(),
            "newest_versions": list(versions[-5:]),
        }
    address, digest = locked_conda_artefact(lock_packages, exception.package, exception.version)
    artefact = fetcher(address, ARTEFACT_LIMIT)
    if hashlib.sha256(artefact).hexdigest() != digest:
        raise ValueError("Downloaded artefact does not match the lock digest.")
    paths = conda_package_paths(address.rsplit("/", 1)[-1], artefact)
    present = sorted(
        path for path in paths if any(fnmatch.fnmatchcase(path, item) for item in exception.absent)
    )
    return {
        "holds": not present,
        "artefact": address,
        "artefact_sha256": digest,
        "declared_paths": len(paths),
        "present": present,
    }


def decide_matches(
    matches: list[Any],
    exceptions: tuple[AdvisoryException, ...],
    lock_packages: list[Any],
    fetcher: Fetch,
) -> dict[str, Any]:
    """Decide every scanner match of one lock against its recorded exceptions.

    Parameters
    ----------
    matches : list
        Scanner match rows of one lock, each with package, version and id.
    exceptions : tuple of AdvisoryException
        Exceptions recorded for that lock and ecosystem.
    lock_packages : list
        Package records of the parsed lock.
    fetcher : callable
        Transport used by the live checks.

    Returns
    -------
    dict
        ``accepted`` (match and evidence), ``blocked`` (match and why) and
        ``unused`` (exceptions that matched no finding). The lock is
        acceptable only when ``blocked`` and ``unused`` are both empty.
        A match without an exception, a malformed match and an exception
        whose check fails or errors are all blocked.
    """
    by_identity = {(item.package, item.version, item.advisory): item for item in exceptions}
    used: set[tuple[str, str, str]] = set()
    accepted: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for match in matches:
        fields = (
            [match.get("package"), match.get("version"), match.get("id")]
            if isinstance(match, dict)
            else []
        )
        if len(fields) != 3 or not all(isinstance(field, str) for field in fields):
            blocked.append({"match": fields, "why": "match is malformed"})
            continue
        identity = (str(fields[0]), str(fields[1]), str(fields[2]))
        exception = by_identity.get(identity)
        if exception is None:
            blocked.append({"match": list(identity), "why": "no exception recorded"})
            continue
        used.add(identity)
        try:
            evidence = verify_exception(exception, lock_packages, fetcher)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            blocked.append({"match": list(identity), "why": f"check failed: {type(exc).__name__}"})
            continue
        row = {"exception": asdict(exception), "evidence": evidence}
        if evidence["holds"] is True:
            accepted.append(row)
        else:
            blocked.append({**row, "match": list(identity), "why": "reason no longer holds"})
    unused = [asdict(item) for key, item in by_identity.items() if key not in used]
    return {"accepted": accepted, "blocked": blocked, "unused": unused}


def pixi_lock_decision(
    payload: dict[str, Any],
    returncode: int,
    relative: str,
    lock_packages: list[Any],
    exceptions: tuple[AdvisoryException, ...],
    fetcher: Fetch,
) -> dict[str, Any]:
    """Decide one Pixi lock from its validated scanner report and the exceptions.

    Parameters
    ----------
    payload : dict
        Scanner report whose lists and counters were already validated.
    returncode : int
        Scanner exit status: zero without matches, one with matches.
    relative : str
        Repository path of the lock, as the exception record names it.
    lock_packages : list
        Package records of the parsed lock.
    exceptions : tuple of AdvisoryException
        Every recorded exception; only those of this lock are used.
    fetcher : callable
        Transport used by the live checks; each address is fetched once.

    Returns
    -------
    dict
        The decision of ``decide_matches`` plus ``passed``. A lock passes only
        if nothing is ignored, unaudited or blocked, no exception is unused
        and the exit status agrees with the presence of matches. Packages the
        scanner could not audit are never covered by an exception.
    """
    cache: dict[str, bytes] = {}

    def once(address: str, limit: int) -> bytes:
        if address not in cache:
            cache[address] = fetcher(address, limit)
        return cache[address]

    mine = tuple(item for item in exceptions if (item.ecosystem, item.lock) == ("pixi", relative))
    matches = payload["matches"]
    decision = decide_matches(matches, mine, lock_packages, once)
    uncovered = any(payload[key] for key in ("ignored", "not_audited", "unused_ignores"))
    decision["passed"] = (
        not uncovered
        and not decision["blocked"]
        and not decision["unused"]
        and returncode == (1 if matches else 0)
    )
    return decision
