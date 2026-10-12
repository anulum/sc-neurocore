# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-finding advisory exceptions verified on every audit run

"""Accept one advisory finding only while a machine check of its reason still holds.

An exception names one package version. For a scanner finding it also names the
advisory and carries one of two reasons; for a package the scanner cannot audit
it carries the third. The audit re-establishes the reason from live sources
each time:

``no_fixed_version_in_channel``
    The locked artefact is the newest version and the newest build its own
    channel publishes for its platform; channel and platform are read from
    the lock, never from this record. The check fails as soon as the channel
    offers a newer version or a newer build of the same version, so a
    published fix turns the audit red without anyone editing this record.

``component_not_in_package``
    The advisory concerns a program that the locked package does not ship. The
    check downloads the exact locked artefact, verifies its digest against the
    lock and requires that no file of the package matches the declared paths.

``vendor_records_reviewed``
    The scanner has no advisory data for the vendor channel the package comes
    from, so the audit reads the vendor's records itself. The check requires
    that the vendor channel still lists the locked file with the lock digest,
    that the vulnerability records bound to the vendor's product identifiers
    are exactly the reviewed ones, each with exactly the products it named
    when it was reviewed, and that none of them names a package of the lock
    as its product. Which vendor identifiers belong to a channel is fixed in
    this module (:data:`VENDOR_IDENTIFIERS`), keyed by the host and channel
    read from the lock; a table cannot choose them, and a package from a
    channel without an entry cannot be excepted. A new vendor record, or a
    reviewed record whose product list changed, turns the audit red until it
    is read again. Records that carry no product identifier yet are not seen.

There is no severity, confidence or reachability filter, no wildcard and no
expiry by date: an exception lives exactly as long as its evidence does. An
exception that no longer matches a finding or an unaudited package is itself
a failure, so the record cannot outlive what it was written for.
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
import re
import ssl
import subprocess
import sys
import tarfile
import time
from typing import Any
import urllib.parse
import zipfile

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REASONS = ("no_fixed_version_in_channel", "component_not_in_package", "vendor_records_reviewed")
COVERAGE_REASON = "vendor_records_reviewed"
EXCEPTION_FILE = Path("tools/security_scan/advisory_exceptions.toml")
CHANNEL_INDEX = "https://api.anaconda.org/package/{channel}/{package}"
INDEX_LIMIT = 32 * 1024 * 1024
ARTEFACT_LIMIT = 64 * 1024 * 1024
REDIRECT_LIMIT = 5
VENDOR_RECORDS = (
    "https://services.nvd.nist.gov/rest/json/cves/2.0?virtualMatchString=cpe:2.3:a:{vendor}:*"
)
_COMMON = {"ecosystem", "lock", "package", "version", "reason", "recorded", "source"}
ECOSYSTEM = "pixi"
CHANNEL_HOST = "conda.anaconda.org"
_PLAIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
VENDOR_IDENTIFIERS: dict[tuple[str, str], tuple[str, ...]] = {
    ("conda.modular.com", "max"): ("modular",),
}
"""Vendor parts of product identifiers, per channel host and channel name.

The vulnerability database files a vendor's records under these strings. The
mapping is the one reviewed place that says whose records speak for a channel;
a second spelling of a vendor is added here, with its test.
"""
RETRY_PAUSES = (2.0, 6.0)
"""Seconds to wait before the second and the third attempt at a busy source."""
_BUSY = frozenset({403, 408, 429, 500, 502, 503, 504})
_RECORD = re.compile(r"CVE-[0-9]{4}-[0-9]{4,}")
_PRODUCT = re.compile(r"\S+")
_BY_REASON: dict[str, set[str]] = {
    "no_fixed_version_in_channel": {"advisory"},
    "component_not_in_package": {"advisory", "component", "absent"},
    "vendor_records_reviewed": {"reviewed"},
}


class EvidenceUnavailable(OSError):
    """A source of exception evidence did not answer; the reason is neither held nor refuted."""


Fetch = Callable[[str, int], bytes]


@dataclass(frozen=True)
class AdvisoryException:
    """One reviewed exception for one locked package version.

    A finding exception names its ``advisory``. A coverage exception lists the
    vendor records already ``reviewed``, each with the products it named at
    review; it has no advisory.
    """

    ecosystem: str
    lock: str
    package: str
    version: str
    reason: str
    recorded: str
    source: str
    advisory: str = ""
    component: str = ""
    absent: tuple[str, ...] = ()
    reviewed: tuple[tuple[str, tuple[str, ...]], ...] = ()


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
        reason or ecosystem, a wildcard package, version or advisory, an
        empty or blanket path list, a reviewed list that is not a list of
        distinct records each with its distinct products, or repeats an
        earlier lock, package, version and advisory.
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
        reviewed = table.get("reviewed", [])
        texts = [value for key, value in table.items() if key not in ("absent", "reviewed")]
        if not all(isinstance(value, str) and value.strip() == value and value for value in texts):
            raise ValueError("Advisory exception values must be non-empty text.")
        if table["ecosystem"] != ECOSYSTEM:
            raise ValueError("Advisory exceptions exist for the Pixi ecosystem only.")
        advisory = table.get("advisory", "")
        if any(
            "*" in text or "?" in text for text in (table["package"], table["version"], advisory)
        ):
            raise ValueError("Advisory exception identity cannot contain a wildcard.")
        if reason == "component_not_in_package" and (
            not isinstance(absent, list)
            or not absent
            or not all(isinstance(item, str) and item.strip("*/? ") for item in absent)
        ):
            raise ValueError("Advisory exception needs concrete absent paths.")
        if reason == COVERAGE_REASON:
            reviewed = reviewed_records(reviewed)
        identity = (table["lock"], table["package"], table["version"], advisory)
        if identity in seen:
            raise ValueError("Advisory exception is recorded twice.")
        seen.add(identity)
        exceptions.append(
            AdvisoryException(**{**table, "absent": tuple(absent), "reviewed": tuple(reviewed)})
        )
    return tuple(exceptions)


def reviewed_records(rows: object) -> list[tuple[str, tuple[str, ...]]]:
    """Read the reviewed vendor records of one coverage exception.

    Parameters
    ----------
    rows : object
        Value of the ``reviewed`` key: a list of tables, each with ``record``
        and ``products``.

    Returns
    -------
    list of tuple
        Record name and its sorted products, in table order.

    Raises
    ------
    ValueError
        The value is not a list of such tables, a record name is not a
        database record name or repeats, or a product list is empty, repeats
        a product or holds anything but plain words.
    """
    if not isinstance(rows, list):
        raise ValueError("Reviewed vendor records must be a list of tables.")
    records: list[tuple[str, tuple[str, ...]]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"record", "products"}:
            raise ValueError("A reviewed vendor record needs a record name and its products.")
        name, products = row["record"], row["products"]
        if (
            not isinstance(name, str)
            or not _RECORD.fullmatch(name)
            or name in {seen for seen, _ in records}
            or not isinstance(products, list)
            or not products
            or not all(isinstance(item, str) and _PRODUCT.fullmatch(item) for item in products)
            or len(set(products)) != len(products)
        ):
            raise ValueError("A reviewed vendor record is not distinct and fully specified.")
        records.append((name, tuple(sorted(products))))
    return records


def channel_vendors(address: str) -> tuple[str, ...]:
    """Read the vendor identifiers that speak for the channel of a locked address.

    Parameters
    ----------
    address : str
        Download address of a locked conda artefact.

    Returns
    -------
    tuple of str
        Vendor parts of the product identifiers recorded for that channel.

    Raises
    ------
    ValueError
        The channel host and name have no entry in :data:`VENDOR_IDENTIFIERS`;
        a package from such a channel cannot be excepted by vendor records.
    """
    parts = urllib.parse.urlsplit(address)
    channel = parts.path.strip("/").split("/")[0]
    vendors = VENDOR_IDENTIFIERS.get((parts.hostname or "", channel))
    if not vendors:
        raise ValueError("No vendor identifiers are recorded for the channel of the artefact.")
    return vendors


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


def channel_files(raw: bytes, subdir: str) -> tuple[dict[str, Any], ...]:
    """List the files a channel publishes for one platform.

    Parameters
    ----------
    raw : bytes
        Package record of the channel index, exactly as received.
    subdir : str
        Platform directory, for example ``linux-64``.

    Returns
    -------
    tuple of dict
        One row per file of that platform, with ``version``, ``basename``,
        ``build_number`` and ``sha256`` as the index states them.

    Raises
    ------
    ValueError
        The record is not the expected object, lists no file for the
        platform, or a file of the platform lacks a version, a name or an
        integer build number.
    """
    payload = json.loads(raw)
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        raise ValueError("Channel index record has no file list.")
    rows = [
        row
        for row in files
        if isinstance(row, dict)
        and isinstance(row.get("attrs"), dict)
        and row["attrs"].get("subdir") == subdir
    ]
    if not rows:
        raise ValueError("Channel index lists no file for the platform.")
    listed: list[dict[str, Any]] = []
    for row in rows:
        number = row["attrs"].get("build_number")
        if (
            not isinstance(row.get("version"), str)
            or not isinstance(row.get("basename"), str)
            or isinstance(number, bool)
            or not isinstance(number, int)
        ):
            raise ValueError("Channel index file record is incomplete.")
        listed.append(
            {
                "version": row["version"],
                "basename": row["basename"],
                "build_number": number,
                "sha256": row.get("sha256"),
            }
        )
    return tuple(listed)


def newest_file_is(
    files: tuple[dict[str, Any], ...], version: str, basename: str, digest: str
) -> bool:
    """Tell whether the locked file is the newest version and the newest build.

    Parameters
    ----------
    files : tuple of dict
        Files the channel publishes for the platform.
    version : str
        Version recorded in the lock.
    basename : str
        Platform directory and file name of the locked artefact.
    digest : str
        SHA-256 the lock records for that artefact.

    Returns
    -------
    bool
        True only if the index lists the locked file with the lock's digest,
        no listed version is greater than the locked one, and no file of the
        locked version has a greater build number than the locked file.

    Raises
    ------
    ValueError
        A listed version cannot be ordered numerically.
    """
    locked = [row for row in files if row["basename"] == basename]
    if len(locked) != 1 or locked[0]["version"] != version or locked[0]["sha256"] != digest:
        return False
    if max(numeric_version(row["version"]) for row in files) != numeric_version(version):
        return False
    builds = [row["build_number"] for row in files if row["version"] == version]
    return bool(max(builds) == locked[0]["build_number"])


def locked_channel(address: str) -> tuple[str, str, str]:
    """Read channel, platform and file name from the address a lock installs from.

    Parameters
    ----------
    address : str
        Download address of a locked conda artefact.

    Returns
    -------
    tuple of str
        Channel, platform directory and file name.

    Raises
    ------
    ValueError
        The address is not ``https://conda.anaconda.org/<channel>/<platform>/<file>``
        with plain names; an exception cannot be verified for any other source.
    """
    parts = urllib.parse.urlsplit(address)
    segments = parts.path.strip("/").split("/")
    if (
        parts.scheme != "https"
        or parts.hostname != CHANNEL_HOST
        or parts.query
        or parts.fragment
        or len(segments) != 3
        or not all(_PLAIN.fullmatch(segment) for segment in segments)
    ):
        raise ValueError("Locked artefact does not come from a recognised channel address.")
    return segments[0], segments[1], segments[2]


def locked_index(address: str) -> tuple[str, str]:
    """Read the platform index address and the file name from a locked address.

    Parameters
    ----------
    address : str
        Download address of a locked conda artefact on any channel host.

    Returns
    -------
    tuple of str
        Address of ``repodata.json`` beside the artefact, and the file name.

    Raises
    ------
    ValueError
        The address is not ``https://<host>/<channel...>/<platform>/<file>``
        with plain names and without query or fragment.
    """
    parts = urllib.parse.urlsplit(address)
    segments = parts.path.strip("/").split("/")
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.query
        or parts.fragment
        or len(segments) < 3
        or not all(_PLAIN.fullmatch(segment) for segment in segments)
    ):
        raise ValueError("Locked artefact does not come from a plain channel address.")
    return address.rsplit("/", 1)[0] + "/repodata.json", segments[-1]


def listed_digest(raw: bytes, name: str) -> str | None:
    """Read the digest a platform index states for one file.

    Parameters
    ----------
    raw : bytes
        ``repodata.json`` of the platform, exactly as received.
    name : str
        File name of the locked artefact.

    Returns
    -------
    str or None
        SHA-256 the index states for the file, or None if the index does not
        list the file, for example after the vendor withdrew it.

    Raises
    ------
    ValueError
        The index is not an object with package tables.
    """
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Channel platform index has no package tables.")
    for key in ("packages", "packages.conda"):
        table = payload.get(key)
        if not isinstance(table, dict):
            raise ValueError("Channel platform index has no package tables.")
        row = table.get(name)
        if isinstance(row, dict) and isinstance(row.get("sha256"), str):
            return str(row["sha256"])
    return None


def vendor_records(raw: bytes, vendor: str) -> dict[str, tuple[str, ...]]:
    """Read every vulnerability record bound to a vendor and the products it names.

    Parameters
    ----------
    raw : bytes
        Answer of the vulnerability database to a query for all records whose
        product identifiers carry the vendor.
    vendor : str
        Vendor part of the product identifiers.

    Returns
    -------
    dict
        Record name to the sorted products of that vendor the record names.

    Raises
    ------
    ValueError
        The answer is not the complete result (it is paginated or malformed),
        a record has no name, or a record names no product of the vendor.
    """
    payload = json.loads(raw)
    rows = payload.get("vulnerabilities") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or payload.get("totalResults") != len(rows):
        raise ValueError("Vendor record answer is not one complete result.")
    records: dict[str, tuple[str, ...]] = {}
    for row in rows:
        record = row.get("cve") if isinstance(row, dict) else None
        if not isinstance(record, dict):
            raise ValueError("Vendor record is malformed.")
        name, configurations = record.get("id"), record.get("configurations")
        if not isinstance(name, str) or not isinstance(configurations, list):
            raise ValueError("Vendor record is malformed.")
        products: set[str] = set()
        for configuration in configurations:
            for node in configuration.get("nodes", []):
                for match in node.get("cpeMatch", []):
                    fields = str(match.get("criteria", "")).split(":")
                    if len(fields) > 4 and fields[3] == vendor:
                        products.add(fields[4])
        if not products:
            raise ValueError("Vendor record names no product of the vendor.")
        records[name] = tuple(sorted(products))
    return records


def locked_names(lock_packages: list[Any]) -> frozenset[str]:
    """List the package names of a lock in the spelling of product identifiers.

    Parameters
    ----------
    lock_packages : list
        Package records of the parsed Pixi lock.

    Returns
    -------
    frozenset of str
        Lower-case names with hyphens and dots as underscores. A conda file
        name is ``<name>-<version>-<build>``, so the name is everything before
        the last two hyphens.
    """
    names = set()
    for row in lock_packages:
        if isinstance(row, dict) and isinstance(row.get("conda"), str):
            stem = row["conda"].rsplit("/", 1)[-1].rsplit("-", 2)[0]
            names.add(re.sub(r"[-.]", "_", stem.lower()))
    return frozenset(names)


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
    """Download one body over verified TLS with a size bound and a bounded retry.

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
    EvidenceUnavailable
        The source refused the connection, timed out or answered that it is
        busy on every attempt. This says nothing about the reason itself.
    ValueError, OSError
        An address on the way is not HTTPS, the certificate is not trusted,
        the source redirects too often or answers without the body, or the
        body exceeds the limit. None of these is retried.
    """
    trust = context or ssl.create_default_context()
    pauses = list(RETRY_PAUSES)
    while True:
        try:
            return _fetch_once(address, limit, trust)
        except (ConnectionError, TimeoutError) as exc:
            if not pauses:
                raise EvidenceUnavailable("Exception evidence source did not answer.") from exc
            time.sleep(pauses.pop(0))


def _fetch_once(address: str, limit: int, trust: ssl.SSLContext) -> bytes:
    """Make one attempt; a busy source raises ConnectionError so the caller may retry."""
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
            if response.status in _BUSY:
                raise ConnectionError("Exception evidence source is busy.")
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
        channel and platform read from the lock, the index address, its body
        digest, the versions seen and the build numbers of the locked
        version; for a content check the artefact address, its digest, the
        number of declared paths and the paths that matched; for a vendor
        check both addresses and body digests, the digest the vendor lists,
        the vendor identifiers of the channel, the vendor's records with
        their products, and the records that are unreviewed, withdrawn,
        changed since review or name a locked package. Nothing here is a
        judgement of severity.

    Raises
    ------
    ValueError, OSError, subprocess.SubprocessError
        Evidence is unavailable, oversized, unverifiable or malformed. The
        caller treats every such error as an exception that does not hold.
    """
    address, digest = locked_conda_artefact(lock_packages, exception.package, exception.version)
    if exception.reason == COVERAGE_REASON:
        index, name = locked_index(address)
        listing = fetcher(index, INDEX_LIMIT)
        listed = listed_digest(listing, name)
        vendors = channel_vendors(address)
        records: dict[str, tuple[str, ...]] = {}
        sources: dict[str, str] = {}
        for vendor in vendors:
            source = VENDOR_RECORDS.format(vendor=vendor)
            answer = fetcher(source, INDEX_LIMIT)
            sources[source] = hashlib.sha256(answer).hexdigest()
            for name, products in vendor_records(answer, vendor).items():
                records[name] = tuple(sorted({*records.get(name, ()), *products}))
        expected = dict(exception.reviewed)
        names = locked_names(lock_packages)
        naming = sorted(name for name, products in records.items() if names & set(products))
        unreviewed = sorted(set(records) - set(expected))
        withdrawn = sorted(set(expected) - set(records))
        changed = sorted(
            name for name in records if name in expected and records[name] != expected[name]
        )
        return {
            "holds": listed == digest and not (naming or unreviewed or withdrawn or changed),
            "index": index,
            "index_sha256": hashlib.sha256(listing).hexdigest(),
            "listed_sha256": listed,
            "vendors": list(vendors),
            "records_sha256": sources,
            "vendor_records": {name: list(products) for name, products in records.items()},
            "naming_a_locked_package": naming,
            "unreviewed": unreviewed,
            "withdrawn": withdrawn,
            "changed_since_review": changed,
        }
    if exception.reason == "no_fixed_version_in_channel":
        channel, subdir, name = locked_channel(address)
        index = CHANNEL_INDEX.format(channel=channel, package=exception.package)
        raw = fetcher(index, INDEX_LIMIT)
        files = channel_files(raw, subdir)
        versions = sorted({row["version"] for row in files}, key=numeric_version)
        return {
            "holds": newest_file_is(files, exception.version, f"{subdir}/{name}", digest),
            "channel": channel,
            "platform": subdir,
            "index": index,
            "index_sha256": hashlib.sha256(raw).hexdigest(),
            "newest_versions": versions[-5:],
            "builds_of_locked_version": sorted(
                row["build_number"] for row in files if row["version"] == exception.version
            ),
        }
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
    key: str = "id",
) -> dict[str, Any]:
    """Decide every scanner row of one kind against its recorded exceptions.

    Parameters
    ----------
    matches : list
        Scanner rows of one lock: matches, each with package, version and id,
        or unaudited packages, each with package and version.
    exceptions : tuple of AdvisoryException
        Exceptions of that kind recorded for that lock and ecosystem.
    lock_packages : list
        Package records of the parsed lock.
    fetcher : callable
        Transport used by the live checks.
    key : str
        Row field that holds the advisory name; empty for unaudited packages,
        which have none.

    Returns
    -------
    dict
        ``accepted`` (match and evidence), ``blocked`` (match and why) and
        ``unused`` (exceptions that matched no finding). The lock is
        acceptable only when ``blocked`` and ``unused`` are both empty.
        A match without an exception, a malformed match and an exception
        whose check fails or errors are all blocked. A source that did not
        answer blocks with the reason ``source unavailable``, apart from a
        reason that no longer holds.
    """
    by_identity = {(item.package, item.version, item.advisory): item for item in exceptions}
    used: set[tuple[str, str, str]] = set()
    accepted: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for match in matches:
        fields = (
            [match.get("package"), match.get("version"), match.get(key) if key else ""]
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
        except EvidenceUnavailable:
            # An outage is not a changed reason: say so, and still block.
            blocked.append({"match": list(identity), "why": "source unavailable"})
            continue
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
        if nothing is ignored or blocked, no exception is unused and the exit
        status agrees with the presence of matches. A package the scanner
        could not audit is blocked unless a vendor-record exception for
        exactly that package version holds.
    """
    cache: dict[str, bytes] = {}

    def once(address: str, limit: int) -> bytes:
        if address not in cache:
            cache[address] = fetcher(address, limit)
        return cache[address]

    mine = tuple(item for item in exceptions if (item.ecosystem, item.lock) == ("pixi", relative))
    findings = tuple(item for item in mine if item.reason != COVERAGE_REASON)
    coverage = tuple(item for item in mine if item.reason == COVERAGE_REASON)
    matches = payload["matches"]
    decision = decide_matches(matches, findings, lock_packages, once)
    unaudited = decide_matches(payload["not_audited"], coverage, lock_packages, once, key="")
    for part in ("accepted", "blocked", "unused"):
        decision[part] += unaudited[part]
    uncovered = any(payload[key] for key in ("ignored", "unused_ignores"))
    decision["passed"] = (
        not uncovered
        and not decision["blocked"]
        and not decision["unused"]
        and returncode == (1 if matches else 0)
    )
    return decision


def require_known_locks(exceptions: tuple[AdvisoryException, ...], audited: set[str]) -> None:
    """Refuse an exception record that names a lock the audit does not read.

    Parameters
    ----------
    exceptions : tuple of AdvisoryException
        Every recorded exception.
    audited : set of str
        Repository paths of the committed locks of the ecosystem.

    Raises
    ------
    ValueError
        A table names a lock that is not among them. Such a table would
        never be compared with a finding and so could never become unused.
    """
    if any(item.lock not in audited for item in exceptions):
        raise ValueError("Advisory exception names a lock that is not audited.")
