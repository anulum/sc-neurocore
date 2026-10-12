# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Advisory exceptions hold only while their evidence does

"""Decide real exception records against real package containers and index bodies."""

from __future__ import annotations

import bz2
import hashlib
import http.server
import io
import json
from pathlib import Path
import shutil
import ssl
import subprocess
import tarfile
import threading
from typing import Any
import zipfile

import pytest
import yaml

from tools.security_scan.advisory_exceptions import (
    CHANNEL_INDEX,
    EXCEPTION_FILE,
    VENDOR_RECORDS,
    channel_files,
    conda_package_paths,
    decide_matches,
    fetch,
    listed_digest,
    load_exceptions,
    locked_conda_artefact,
    locked_channel,
    locked_index,
    locked_names,
    newest_file_is,
    numeric_version,
    pixi_lock_decision,
    require_known_locks,
    vendor_records,
)

ROOT = Path(__file__).resolve().parents[2]
LOCK = "src/sc_neurocore/accel/mojo/pixi.lock"
HEADER = (
    'ecosystem = "pixi"\nlock = "'
    + LOCK
    + '"\nrecorded = "2026-10-12"\nsource = "https://example.org/a"\n'
)


def _metadata_tar(paths: list[str]) -> bytes:
    """Build the metadata archive of a conda package that declares the given paths."""
    body = json.dumps({"paths": [{"_path": path} for path in paths], "paths_version": 1}).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo("info/paths.json")
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))
    return buffer.getvalue()


def _conda(paths: list[str]) -> bytes:
    """Build a real ``.conda`` container: a zip holding a zstd-compressed metadata archive."""
    assert shutil.which("zstd") is not None, "The conda container format needs the zstd program."
    compressed = subprocess.run(
        ["zstd", "--stdout"], input=_metadata_tar(paths), capture_output=True, check=True
    ).stdout
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as outer:
        outer.writestr("metadata.json", '{"conda_pkg_format_version": 2}')
        outer.writestr("info-tool-1.0-h0_0.tar.zst", compressed)
    return buffer.getvalue()


ARTEFACT = "https://conda.anaconda.org/channel/linux-64/tool-1.0-h0_0.conda"
NO_FIX = (
    'package = "tool"\nversion = "1.0"\nadvisory = "{}"\nreason = "no_fixed_version_in_channel"\n'
)
ABSENT = (
    'package = "tool"\nversion = "1.0"\nadvisory = "{}"\nreason = "component_not_in_package"\n'
    'component = "reader"\nabsent = ["*reader*"]\n'
)
VENDOR_ARTEFACT = "https://packages.example/max/linux-64/kit-2.0-release.conda"
VENDOR_INDEX = "https://packages.example/max/linux-64/repodata.json"
REVIEWED = (
    'package = "kit"\nversion = "2.0"\nreason = "vendor_records_reviewed"\n'
    'vendor = "maker"\nreviewed = ["CVE-2025-1000"]\n'
)


def _index(files: list[tuple[str, int, str]], subdir: str = "linux-64") -> bytes:
    """Build a channel index record from version, build number and digest per file."""
    rows = [
        {
            "version": version,
            "basename": f"{subdir}/tool-{version}-h0_{number}.conda",
            "sha256": digest,
            "attrs": {"subdir": subdir, "build_number": number},
        }
        for version, number, digest in files
    ]
    return json.dumps({"files": rows}).encode()


def _table(extra: str) -> bytes:
    """Return one exception table with the common keys and the given reason keys."""
    return ("[[exception]]\n" + HEADER + extra).encode()


def test_the_committed_record_names_real_locked_artefacts() -> None:
    """Every committed exception binds one artefact of a lock the audit reads."""
    exceptions = load_exceptions((ROOT / EXCEPTION_FILE).read_bytes())
    assert exceptions
    require_known_locks(exceptions, {LOCK})
    for item in exceptions:
        packages = yaml.safe_load((ROOT / item.lock).read_bytes())["packages"]
        address, digest = locked_conda_artefact(packages, item.package, item.version)
        assert len(digest) == 64
        if item.reason == "vendor_records_reviewed":
            assert locked_index(address)[0].startswith("https://conda.modular.com/max/")
            assert item.vendor == "modular" and item.advisory == ""
        else:
            assert locked_channel(address)[:2] == ("conda-forge", "linux-64")
            assert item.advisory and not item.vendor and not item.reviewed
        assert item.recorded and item.source.startswith("https://")


@pytest.mark.parametrize(
    "extra",
    [
        'package = "tool"\nversion = "1.0"\nadvisory = "CVE-1"\nreason = "trust_me"\n',
        NO_FIX.format("CVE-1") + 'channel = "conda-forge"\n',
        NO_FIX.format("CVE-1") + 'severity = "low"\n',
        NO_FIX.format("CVE-1").replace('"tool"', '"*"'),
        NO_FIX.format("CVE-*"),
        NO_FIX.format("CVE-1").replace('"1.0"', '""'),
        ABSENT.format("CVE-1").replace('["*reader*"]', "[]"),
        ABSENT.format("CVE-1").replace('["*reader*"]', '["*"]'),
        ABSENT.format("CVE-1").replace('component = "reader"\n', ""),
        NO_FIX.format("CVE-1").replace('advisory = "CVE-1"\n', ""),
        REVIEWED + 'advisory = "CVE-2025-1000"\n',
        REVIEWED.replace('"maker"', '"Maker Inc"'),
        REVIEWED.replace('["CVE-2025-1000"]', '["CVE-2025-1000", "CVE-2025-1000"]'),
        REVIEWED.replace('["CVE-2025-1000"]', '["any record"]'),
        REVIEWED.replace('["CVE-2025-1000"]', '"CVE-2025-1000"'),
        REVIEWED.replace('"kit"', '"k?t"'),
    ],
)
def test_an_underspecified_or_blanket_exception_is_refused(extra: str) -> None:
    """Unknown reasons, missing or extra keys, wildcards and empty values do not load.

    Parameters
    ----------
    extra : str
        Reason-specific part of one exception table.
    """
    with pytest.raises(ValueError):
        load_exceptions(_table(extra))


def test_repeated_foreign_and_unbound_exceptions_are_refused() -> None:
    """One finding has one record, of the audited ecosystem, for a lock the audit reads."""
    once = _table(NO_FIX.format("CVE-1"))
    assert len(load_exceptions(once)) == 1
    with pytest.raises(ValueError, match="twice"):
        load_exceptions(once + b"\n" + once)
    with pytest.raises(ValueError, match="top-level"):
        load_exceptions(b'[settings]\nallow = "all"\n')
    for other in ("PIXI", "uv", "pixi "):
        with pytest.raises(ValueError):
            load_exceptions(once.replace(b'ecosystem = "pixi"', f'ecosystem = "{other}"'.encode()))
    require_known_locks(load_exceptions(once), {LOCK, "other/pixi.lock"})
    with pytest.raises(ValueError, match="not audited"):
        require_known_locks(load_exceptions(once), {LOCK + ".old"})
    require_known_locks((), set())


def test_the_channel_is_read_from_the_lock_and_only_a_plain_address_counts() -> None:
    """Channel and platform come from the locked address; anything else cannot be verified."""
    assert locked_channel(ARTEFACT) == ("channel", "linux-64", "tool-1.0-h0_0.conda")
    for address in (
        ARTEFACT.replace("https://", "http://"),
        ARTEFACT.replace("conda.anaconda.org", "conda.example"),
        ARTEFACT + "?x=1",
        ARTEFACT + "#x",
        ARTEFACT.replace("/channel/", "/channel/extra/"),
        ARTEFACT.replace("/channel/", "/../"),
        ARTEFACT.replace("/channel/", "/chan nel/"),
        "https://conda.anaconda.org/tool-1.0-h0_0.conda",
    ):
        with pytest.raises(ValueError, match="recognised channel"):
            locked_channel(address)


def test_newest_means_newest_version_and_newest_build_of_the_locked_file() -> None:
    """A newer version, a newer build, a missing file and a foreign digest all end the exception."""
    name = "linux-64/tool-1.0-h0_0.conda"
    files = channel_files(_index([("0.9", 0, "x"), ("1.0", 0, "d")]), "linux-64")
    assert newest_file_is(files, "1.0", name, "d") is True
    assert numeric_version("1.22.10") > numeric_version("1.22.2")
    newer_version = channel_files(_index([("1.0", 0, "d"), ("1.1", 0, "x")]), "linux-64")
    assert newest_file_is(newer_version, "1.0", name, "d") is False
    newer_build = channel_files(_index([("1.0", 0, "d"), ("1.0", 1, "x")]), "linux-64")
    assert newest_file_is(newer_build, "1.0", name, "d") is False
    assert newest_file_is(newer_build, "1.0", "linux-64/tool-1.0-h0_1.conda", "x") is True
    assert newest_file_is(files, "1.0", name, "other-digest") is False
    assert newest_file_is(files, "0.9", name, "d") is False
    assert newest_file_is(files, "1.0", "linux-64/tool-1.0-h9_0.conda", "d") is False
    with pytest.raises(ValueError):
        newest_file_is(
            channel_files(_index([("1.0", 0, "d"), ("1.1rc1", 0, "x")]), "linux-64"),
            "1.0",
            name,
            "d",
        )
    with pytest.raises(ValueError):
        channel_files(_index([("1.0", 0, "d")], "osx-64"), "linux-64")
    with pytest.raises(ValueError):
        channel_files(b"[]", "linux-64")
    for broken in (
        {"version": 1, "basename": name, "attrs": {"subdir": "linux-64", "build_number": 0}},
        {"version": "1.0", "attrs": {"subdir": "linux-64", "build_number": 0}},
        {"version": "1.0", "basename": name, "attrs": {"subdir": "linux-64", "build_number": True}},
        {"version": "1.0", "basename": name, "attrs": {"subdir": "linux-64", "build_number": "0"}},
    ):
        with pytest.raises(ValueError, match="incomplete"):
            channel_files(json.dumps({"files": [broken, "text"]}).encode(), "linux-64")


def test_both_container_formats_yield_their_declared_paths() -> None:
    """The path list is read from real zip-and-zstd and bzip2 containers alike."""
    paths = ["bin/tool", "lib/libtool.so.1"]
    assert conda_package_paths("tool-1.0-h0_0.conda", _conda(paths)) == tuple(paths)
    legacy = bz2.compress(_metadata_tar(paths))
    assert conda_package_paths("tool-1.0-h0_0.tar.bz2", legacy) == tuple(paths)
    with pytest.raises(ValueError):
        conda_package_paths("tool-1.0-h0_0.whl", legacy)
    with pytest.raises(ValueError):
        conda_package_paths("tool-1.0-h0_0.conda", b"not a zip")
    with pytest.raises(ValueError):
        conda_package_paths("tool-1.0-h0_0.conda", _conda([]))


def _case(
    paths: list[str], files: list[tuple[str, int, str]]
) -> tuple[list[Any], dict[str, bytes], bytes]:
    """Return a one-package lock, the bodies its checks fetch, and its exception record.

    The digest ``LOCKED`` in a file row is replaced by the digest of the built artefact.
    """
    artefact = _conda(paths)
    digest = hashlib.sha256(artefact).hexdigest()
    lock = [{"conda": ARTEFACT, "sha256": digest}]
    listed = [
        (version, number, digest if mark == "LOCKED" else mark) for version, number, mark in files
    ]
    bodies = {
        ARTEFACT: artefact,
        CHANNEL_INDEX.format(channel="channel", package="tool"): _index(listed),
    }
    record = _table(NO_FIX.format("CVE-1")) + b"\n" + _table(ABSENT.format("CVE-2"))
    return lock, bodies, record


MATCHES = [
    {"package": "tool", "version": "1.0", "id": "CVE-1"},
    {"package": "tool", "version": "1.0", "id": "CVE-2"},
]


def test_a_finding_is_accepted_only_while_its_evidence_holds() -> None:
    """Each reason is re-established from the fetched bodies, and fails when they change."""
    lock, bodies, record = _case(["bin/tool"], [("0.9", 0, "x"), ("1.0", 0, "LOCKED")])
    exceptions = load_exceptions(record)
    held = decide_matches(MATCHES, exceptions, lock, lambda address, limit: bodies[address])
    assert [len(held[key]) for key in ("accepted", "blocked", "unused")] == [2, 0, 0]
    assert held["accepted"][0]["evidence"]["channel"] == "channel"
    assert held["accepted"][0]["evidence"]["builds_of_locked_version"] == [0]
    assert held["accepted"][1]["evidence"]["declared_paths"] == 1

    for files in (
        [("1.0", 0, "LOCKED"), ("1.1", 0, "x")],
        [("1.0", 0, "LOCKED"), ("1.0", 1, "x")],
        [("1.0", 0, "another-digest")],
    ):
        lock, bodies, record = _case(["bin/tool", "bin/reader"], files)
        changed = decide_matches(
            MATCHES, load_exceptions(record), lock, lambda address, limit: bodies[address]
        )
        assert changed["accepted"] == []
        assert [row["why"] for row in changed["blocked"]] == ["reason no longer holds"] * 2
        assert changed["blocked"][1]["evidence"]["present"] == ["bin/reader"]


def test_missing_altered_and_unrecorded_evidence_all_block() -> None:
    """No record, a failed transfer, a foreign artefact, an absent package and a malformed match."""
    lock, bodies, record = _case(["bin/tool"], [("1.0", 0, "LOCKED")])
    exceptions = load_exceptions(record)
    unrecorded = decide_matches(
        [{"package": "tool", "version": "1.0", "id": "CVE-9"}, "CVE-1", {"package": "tool"}],
        exceptions,
        lock,
        lambda address, limit: bodies[address],
    )
    assert [row["why"] for row in unrecorded["blocked"]] == [
        "no exception recorded",
        "match is malformed",
        "match is malformed",
    ]
    assert len(unrecorded["unused"]) == 2

    def unreachable(address: str, limit: int) -> bytes:
        raise OSError("transfer failed")

    assert [
        row["why"] for row in decide_matches(MATCHES, exceptions, lock, unreachable)["blocked"]
    ] == ["check failed: OSError"] * 2
    foreign = dict(bodies)
    foreign[ARTEFACT] = _conda(["bin/other"])
    swapped = decide_matches(
        MATCHES[1:], exceptions[1:], lock, lambda address, limit: foreign[address]
    )
    assert swapped["blocked"][0]["why"] == "check failed: ValueError"
    for packages in (
        [],
        [
            {
                "conda": ARTEFACT.replace("conda.anaconda.org", "mirror.example"),
                "sha256": lock[0]["sha256"],
            }
        ],
    ):
        absent = decide_matches(
            MATCHES[:1], exceptions[:1], packages, lambda address, limit: bodies[address]
        )
        assert absent["blocked"][0]["why"] == "check failed: ValueError"


def test_a_lock_passes_only_with_every_match_excepted_and_nothing_else_open() -> None:
    """Unaudited packages, scanner ignores, stale records and a wrong exit status all fail."""
    lock, bodies, record = _case(["bin/tool"], [("1.0", 0, "LOCKED")])
    exceptions = load_exceptions(record)
    calls: list[str] = []

    def fetcher(address: str, limit: int) -> bytes:
        calls.append(address)
        return bodies[address]

    report = {"matches": MATCHES, "ignored": [], "not_audited": [], "unused_ignores": []}
    assert pixi_lock_decision(report, 1, LOCK, lock, exceptions, fetcher)["passed"] is True
    assert sorted(calls) == sorted(bodies), "each address is fetched once"
    assert pixi_lock_decision(report, 0, LOCK, lock, exceptions, fetcher)["passed"] is False
    for key in ("ignored", "not_audited", "unused_ignores"):
        opened = {**report, key: [{"package": "vendor-tool"}]}
        assert pixi_lock_decision(opened, 1, LOCK, lock, exceptions, fetcher)["passed"] is False
    clean: dict[str, Any] = {"matches": [], "ignored": [], "not_audited": [], "unused_ignores": []}
    assert pixi_lock_decision(clean, 0, LOCK, lock, (), fetcher)["passed"] is True
    stale = pixi_lock_decision(clean, 0, LOCK, lock, exceptions, fetcher)
    assert stale["passed"] is False and len(stale["unused"]) == 2
    elsewhere = pixi_lock_decision(report, 1, "other/pixi.lock", lock, exceptions, fetcher)
    assert elsewhere["passed"] is False and len(elsewhere["blocked"]) == 2


def test_malformed_containers_and_ambiguous_locks_are_refused() -> None:
    """A container without one metadata archive, a broken path list and a loose lock all fail."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as outer:
        outer.writestr("metadata.json", "{}")
    with pytest.raises(ValueError, match="single metadata archive"):
        conda_package_paths("tool-1.0-h0_0.conda", buffer.getvalue())
    body = json.dumps({"paths": [{"_path": "bin/tool"}, {"path": "bin/other"}]}).encode()
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        member = tarfile.TarInfo("info/paths.json")
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))
        archive.addfile(tarfile.TarInfo("info/empty"))
    with pytest.raises(ValueError, match="malformed"):
        conda_package_paths("tool-1.0-h0_0.tar.bz2", bz2.compress(archive_bytes.getvalue()))
    directory = io.BytesIO()
    with tarfile.open(fileobj=directory, mode="w") as archive:
        folder = tarfile.TarInfo("info/paths.json")
        folder.type = tarfile.DIRTYPE
        archive.addfile(folder)
    with pytest.raises(ValueError, match="no path list"):
        conda_package_paths("tool-1.0-h0_0.tar.bz2", bz2.compress(directory.getvalue()))
    with pytest.raises(ValueError, match="unreadable"):
        conda_package_paths(
            "tool-1.0-h0_0.tar.bz2", bz2.compress(_metadata_tar([])[:0] + b"\0" * 1024)
        )

    row = {"conda": "https://c.example/linux-64/tool-1.0-h0_0.conda", "sha256": "a" * 64}
    longer = {"conda": "https://c.example/linux-64/tool-1.0-extra-h0_0.conda", "sha256": "b" * 64}
    assert locked_conda_artefact([row, longer, "text"], "tool", "1.0") == (row["conda"], "a" * 64)
    for packages in ([], [row, row], [{**row, "sha256": None}]):
        with pytest.raises(ValueError, match="exactly one artefact"):
            locked_conda_artefact(packages, "tool", "1.0")


def test_evidence_is_fetched_over_verified_tls_with_a_size_bound(tmp_path: Path) -> None:
    """A real TLS server is trusted only through its certificate, and only up to the limit.

    Parameters
    ----------
    tmp_path : Path
        Directory for the server's certificate and key.
    """
    openssl = shutil.which("openssl")
    assert openssl is not None, "The TLS transport test needs the openssl program."
    certificate, key = tmp_path / "server.pem", tmp_path / "server.key"
    subprocess.run(
        [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1"]
        + ["-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost"]
        + ["-keyout", str(key), "-out", str(certificate)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    evidence = b"evidence" * 16

    class Handler(http.server.BaseHTTPRequestHandler):
        """Serve one body, three redirects and one refusal without an access log."""

        def do_GET(self) -> None:  # noqa: N802 - name fixed by the standard library
            """Answer by path: the body, a redirect to it, a loop, or a plain-HTTP target."""
            targets = {"/moved": "/body.bin?x=1", "/loop": "/loop", "/plain": "http://localhost/a"}
            if self.path.startswith("/body.bin"):
                self.send_response(200)
                self.send_header("Content-Length", str(len(evidence)))
                self.end_headers()
                self.wfile.write(evidence)
            elif self.path in targets:
                self.send_response(302)
                self.send_header("Location", targets[self.path])
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def log_message(self, format: str, *arguments: Any) -> None:
            """Keep the test output free of request lines."""

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    served = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    served.minimum_version = ssl.TLSVersion.TLSv1_2
    served.load_cert_chain(str(certificate), str(key))
    server.socket = served.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"https://localhost:{server.server_address[1]}"
        trusting = ssl.create_default_context(cafile=str(certificate))
        assert fetch(base + "/body.bin", 1024, trusting) == evidence
        assert fetch(base + "/moved", 1024, trusting) == evidence
        with pytest.raises(ValueError, match="limit"):
            fetch(base + "/body.bin", 16, trusting)
        with pytest.raises(ValueError, match="too often"):
            fetch(base + "/loop", 1024, trusting)
        with pytest.raises(ValueError, match="HTTPS only"):
            fetch(base + "/plain", 1024, trusting)
        with pytest.raises(ValueError, match="did not return"):
            fetch(base + "/absent", 1024, trusting)
        with pytest.raises(OSError):
            fetch(base + "/body.bin", 1024)
        with pytest.raises(ValueError, match="HTTPS only"):
            fetch(base.replace("https://", "http://") + "/body.bin", 1024, trusting)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_two_findings_on_one_package_share_one_download() -> None:
    """Evidence fetched for one finding is reused for the next within a lock decision."""
    lock, bodies, record = _case(["bin/tool"], [("1.0", 0, "LOCKED")])
    record += b"\n" + _table(NO_FIX.format("CVE-3"))
    calls: list[str] = []

    def fetcher(address: str, limit: int) -> bytes:
        calls.append(address)
        return bodies[address]

    report = {
        "matches": [*MATCHES, {"package": "tool", "version": "1.0", "id": "CVE-3"}],
        "ignored": [],
        "not_audited": [],
        "unused_ignores": [],
    }
    decision = pixi_lock_decision(report, 1, LOCK, lock, load_exceptions(record), fetcher)
    assert decision["passed"] is True and len(decision["accepted"]) == 3
    assert len(calls) == 2


def _records(rows: dict[str, list[str]], total: int | None = None) -> bytes:
    """Build a database answer from record names and the product identifiers they carry."""
    listed = [
        {
            "cve": {
                "id": name,
                "configurations": [
                    {"nodes": [{"cpeMatch": [{"criteria": item} for item in criteria]}]}
                ],
            }
        }
        for name, criteria in rows.items()
    ]
    return json.dumps(
        {"totalResults": len(listed) if total is None else total, "vulnerabilities": listed}
    ).encode()


def _vendor_case(
    rows: dict[str, list[str]], listed: str | None = "LOCKED"
) -> tuple[list[Any], dict[str, bytes]]:
    """Return a lock with one vendor package and the two bodies its check fetches."""
    digest = hashlib.sha256(b"kit").hexdigest()
    lock = [{"conda": VENDOR_ARTEFACT, "sha256": digest}, {"conda": ARTEFACT, "sha256": "a" * 64}]
    table = {} if listed is None else {"sha256": digest if listed == "LOCKED" else listed}
    index = {"packages": {}, "packages.conda": {"kit-2.0-release.conda": table}}
    bodies = {
        VENDOR_INDEX: json.dumps(index).encode(),
        VENDOR_RECORDS.format(vendor="maker"): _records(rows),
    }
    return lock, bodies


OTHER_PRODUCT = {"CVE-2025-1000": ["cpe:2.3:a:maker:server:*:*:*:*:*:*:*:*"]}
UNAUDITED = [{"package": "kit", "version": "2.0", "ecosystem": "max", "environments": ["default"]}]


def test_the_vendor_index_and_the_vendor_records_are_read_as_they_are() -> None:
    """Index address, listed digest, record products and lock names come from the data."""
    assert locked_index(VENDOR_ARTEFACT) == (VENDOR_INDEX, "kit-2.0-release.conda")
    for address in (
        VENDOR_ARTEFACT.replace("https://", "http://"),
        VENDOR_ARTEFACT + "?token=1",
        VENDOR_ARTEFACT + "#part",
        "https://packages.example/linux-64/kit-2.0-release.conda",
        "https:///max/linux-64/kit-2.0-release.conda",
        VENDOR_ARTEFACT.replace("/max/", "/m x/"),
    ):
        with pytest.raises(ValueError, match="plain channel"):
            locked_index(address)

    name = "kit-2.0-release.conda"
    modern = json.dumps({"packages": {}, "packages.conda": {name: {"sha256": "d"}}}).encode()
    legacy = json.dumps({"packages": {name: {"sha256": "e"}}, "packages.conda": {}}).encode()
    assert listed_digest(modern, name) == "d" and listed_digest(legacy, name) == "e"
    assert listed_digest(modern, "kit-2.1-release.conda") is None
    unsigned = json.dumps({"packages": {name: {}}, "packages.conda": {name: "text"}}).encode()
    assert listed_digest(unsigned, name) is None
    for broken in (b"[]", b'{"packages": {}}', b'{"packages": [], "packages.conda": {}}'):
        with pytest.raises(ValueError, match="package tables"):
            listed_digest(broken, name)

    answer = _records(
        {
            "CVE-2025-1000": [
                "cpe:2.3:a:maker:server:*:*:*:*:*:*:*:*",
                "cpe:2.3:a:maker:agent:1.0:*:*:*:*:*:*:*",
                "cpe:2.3:o:other:system:*:*:*:*:*:*:*:*",
                "short",
            ],
            "CVE-2026-2000": ["cpe:2.3:a:maker:kit:*:*:*:*:*:*:*:*"],
        }
    )
    assert vendor_records(answer, "maker") == {
        "CVE-2025-1000": ("agent", "server"),
        "CVE-2026-2000": ("kit",),
    }
    assert vendor_records(_records({}), "maker") == {}
    for broken in (
        b"[]",
        b'{"vulnerabilities": "none"}',
        _records(OTHER_PRODUCT, total=2001),
        json.dumps({"totalResults": 1, "vulnerabilities": ["text"]}).encode(),
        json.dumps({"totalResults": 1, "vulnerabilities": [{"cve": {"id": 7}}]}).encode(),
        json.dumps({"totalResults": 1, "vulnerabilities": [{"cve": {"id": "CVE-1"}}]}).encode(),
        _records({"CVE-2025-1000": ["cpe:2.3:a:other:server:*:*:*:*:*:*:*:*"]}),
    ):
        with pytest.raises(ValueError):
            vendor_records(broken, "maker")

    rows = [
        {"conda": "https://c.example/linux-64/ld_impl_linux-64-2.46.1-h0_0.conda"},
        {"conda": "https://c.example/noarch/Ruamel.Yaml-0.18-py_0.conda"},
        {"pypi": "https://files.example/wheel.whl"},
        "text",
    ]
    assert locked_names(rows) == {"ld_impl_linux_64", "ruamel_yaml"}


def test_an_unaudited_package_is_accepted_only_through_its_vendor_records() -> None:
    """The file must stay listed and the vendor's records must be the reviewed ones."""
    exceptions = load_exceptions(_table(REVIEWED))
    assert exceptions[0].reviewed == ("CVE-2025-1000",) and exceptions[0].advisory == ""
    report: dict[str, Any] = {
        "matches": [],
        "ignored": [],
        "not_audited": UNAUDITED,
        "unused_ignores": [],
    }
    lock, bodies = _vendor_case(OTHER_PRODUCT)
    calls: list[str] = []

    def fetcher(address: str, limit: int) -> bytes:
        calls.append(address)
        return bodies[address]

    held = pixi_lock_decision(report, 0, LOCK, lock, exceptions, fetcher)
    assert held["passed"] is True and sorted(calls) == sorted(bodies)
    evidence = held["accepted"][0]["evidence"]
    assert evidence["vendor_records"] == {"CVE-2025-1000": ["server"]}
    assert evidence["listed_sha256"] == lock[0]["sha256"] and evidence["index"] == VENDOR_INDEX

    changes: list[tuple[dict[str, list[str]], str | None, str]] = [
        (
            {**OTHER_PRODUCT, "CVE-2026-2000": OTHER_PRODUCT["CVE-2025-1000"]},
            "LOCKED",
            "unreviewed",
        ),
        ({}, "LOCKED", "withdrawn"),
        ({"CVE-2025-1000": ["cpe:2.3:a:maker:kit:*:*:*:*:*:*:*:*"]}, "LOCKED", "naming"),
        (OTHER_PRODUCT, "f" * 64, "listed"),
        (OTHER_PRODUCT, None, "listed"),
    ]
    for rows, listed, cause in changes:
        lock, changed = _vendor_case(rows, listed)
        decision = pixi_lock_decision(
            report, 0, LOCK, lock, exceptions, lambda address, limit: changed[address]
        )
        assert decision["passed"] is False and decision["accepted"] == []
        blocked = decision["blocked"][0]
        assert blocked["why"] == "reason no longer holds" and blocked["match"] == ["kit", "2.0", ""]
        found = blocked["evidence"]
        assert {
            "unreviewed": found["unreviewed"] == ["CVE-2026-2000"],
            "withdrawn": found["withdrawn"] == ["CVE-2025-1000"],
            "naming": found["naming_a_locked_package"] == ["CVE-2025-1000"],
            "listed": found["listed_sha256"] != lock[0]["sha256"],
        }[cause]

    lock, bodies = _vendor_case(OTHER_PRODUCT)

    def served(address: str, limit: int) -> bytes:
        return bodies[address]

    unrecorded = pixi_lock_decision(report, 0, LOCK, lock, (), served)
    assert unrecorded["passed"] is False
    assert unrecorded["blocked"][0]["why"] == "no exception recorded"
    audited = pixi_lock_decision({**report, "not_audited": []}, 0, LOCK, lock, exceptions, served)
    assert audited["passed"] is False and len(audited["unused"]) == 1
    other_version = [{**UNAUDITED[0], "version": "2.1"}]
    moved = pixi_lock_decision(
        {**report, "not_audited": other_version}, 0, LOCK, lock, exceptions, served
    )
    assert moved["passed"] is False and len(moved["blocked"]) == len(moved["unused"]) == 1
    bodies[VENDOR_RECORDS.format(vendor="maker")] = _records(OTHER_PRODUCT, total=2001)
    partial = pixi_lock_decision(report, 0, LOCK, lock, exceptions, served)
    assert partial["blocked"][0]["why"] == "check failed: ValueError"


def test_a_finding_exception_and_a_coverage_exception_do_not_stand_in_for_each_other() -> None:
    """An unaudited package needs its own reason, and so does a finding on that package."""
    lock, bodies = _vendor_case(OTHER_PRODUCT)
    finding = _table(NO_FIX.format("CVE-1").replace('"tool"', '"kit"').replace('"1.0"', '"2.0"'))

    def served(address: str, limit: int) -> bytes:
        return bodies[address]

    report: dict[str, Any] = {
        "matches": [],
        "ignored": [],
        "not_audited": UNAUDITED,
        "unused_ignores": [],
    }
    crossed = pixi_lock_decision(report, 0, LOCK, lock, load_exceptions(finding), served)
    assert crossed["passed"] is False
    assert [row["why"] for row in crossed["blocked"]] == ["no exception recorded"]
    assert len(crossed["unused"]) == 1
    matched = {
        **report,
        "matches": [{"package": "kit", "version": "2.0", "id": "CVE-2025-1000"}],
        "not_audited": [],
    }
    reverse = pixi_lock_decision(matched, 1, LOCK, lock, load_exceptions(_table(REVIEWED)), served)
    assert reverse["passed"] is False
    assert [row["why"] for row in reverse["blocked"]] == ["no exception recorded"]
    assert len(reverse["unused"]) == 1
