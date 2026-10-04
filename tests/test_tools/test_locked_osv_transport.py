# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real OSV HTTP and connection failure custody

"""Preserve real provider refusals and socket failures through public audit surfaces."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest

from tools.security_scan.locked_dependency_reports import (
    OSV_PAGE_LIMIT,
    OSV_RESPONSE_LIMIT,
    audit_osv_queries,
    osv_page_outcome,
)

ROOT = Path(__file__).resolve().parents[2]
PENDING = [
    (0, {"package": {"ecosystem": "Go", "name": "stdlib"}, "version": "1.27.1"}),
    (3, {"package": {"ecosystem": "Go", "name": "example.invalid/library"}, "version": "1.4.0"}),
]


def test_empty_osv_inventory_refuses_before_creating_a_packet(tmp_path: Path) -> None:
    """An absent identity inventory cannot create a clean-looking audit packet.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory for an attempted audit evidence packet.
    """
    target = tmp_path / "packet"
    with pytest.raises(ValueError, match="exact package identities"):
        audit_osv_queries((), target)
    assert not target.exists()


def test_real_osv_http_refusal_retains_status_and_response(tmp_path: Path) -> None:
    """Keep the actual OSV HTTP 400 body before rejecting an invalid version query.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory retaining the actual provider request and refusal bytes.

    Notes
    -----
    OSV forbids supplying both a version and a versioned package URL:
    https://google.github.io/osv.dev/post-v1-querybatch/#version-rules
    This test contacts the official provider; no response is substituted.
    """
    target = tmp_path / "packet"
    with pytest.raises(ValueError, match="HTTP 200"):
        audit_osv_queries(
            ({"package": {"purl": "pkg:pypi/mlflow@0.4.0"}, "version": "0.4.0"},), target
        )
    request = (target / "request-1.json").read_bytes()
    response = (target / "response-1.json").read_bytes()
    receipt = json.loads((target / "response-1-receipt.json").read_text())
    assert receipt["endpoint"] == "https://api.osv.dev/v1/querybatch"
    assert receipt["http_status"] == 400
    assert receipt["request_sha256"] == hashlib.sha256(request).hexdigest()
    assert receipt["response_sha256"] == hashlib.sha256(response).hexdigest()
    assert receipt["response_bytes"] == len(response) > 0
    assert json.loads(response)["code"] == 3


def test_real_connection_refusal_blocks_cli_and_retains_transport(tmp_path: Path) -> None:
    """A real refused proxy connection fails the audit with typed transport evidence.

    Parameters
    ----------
    tmp_path : Path
        Fresh independent Git checkout and retained CLI audit packet.

    Notes
    -----
    The bound socket deliberately does not listen. The operating system refuses
    the real connection; no HTTP server or advisory response is substituted.
    Requirement profiles and the Julia manifest are copied unchanged.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copytree(ROOT / "requirements", checkout / "requirements")
    source = ROOT / "src/sc_neurocore/accel/julia/sc_compte_wm_network/Manifest.toml"
    lock = checkout / "Manifest.toml"
    shutil.copy(source, lock)
    original = lock.read_bytes()
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(checkout)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        ["git", "add", "requirements", "Manifest.toml"],
        cwd=checkout,
        check=True,
        capture_output=True,
        timeout=30,
    )
    target = tmp_path / "packet"
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        proxy = f"http://127.0.0.1:{reserved.getsockname()[1]}"
        environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "HTTPS_PROXY": proxy,
            "https_proxy": proxy,
            "HTTP_PROXY": proxy,
            "http_proxy": proxy,
            "NO_PROXY": "",
            "no_proxy": "",
        }
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "tools.security_scan.locked_dependency_audit",
                "--repo-root",
                str(checkout),
                "--ecosystem",
                "julia",
                "--output-dir",
                str(target),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
        )
    assert process.returncode == 1
    report = json.loads((target / "summary.json").read_text())
    assert report["passed"] is False and report["errors"] == []
    assert report["audits"] == [
        {"path": "Manifest.toml", "passed": False, "error_type": "URLError"}
    ]
    request = (target / "lock-1/request-1.json").read_bytes()
    receipt = json.loads((target / "lock-1/response-1-receipt.json").read_text())
    assert receipt["http_status"] is None
    assert receipt["error_type"] == "URLError"
    assert receipt["reason_type"] == "ConnectionRefusedError"
    assert receipt["request_sha256"] == hashlib.sha256(request).hexdigest()
    assert "response_sha256" not in receipt
    assert not (target / "lock-1/response-1.json").exists()
    assert lock.read_bytes() == original


def test_osv_page_binds_findings_and_continuations_to_query_positions() -> None:
    """Keep every advisory with its query position and follow each continuation once."""
    raw = json.dumps(
        {
            "results": [
                {"vulns": [{"id": "GO-2026-0001", "modified": "2026-09-01T00:00:00Z"}]},
                {"vulns": [{"id": "GO-2026-0002"}], "next_page_token": "page-two"},
            ]
        }
    ).encode()
    findings, following = osv_page_outcome(1, raw, PENDING, set())
    assert findings == [
        (0, {"id": "GO-2026-0001", "modified": "2026-09-01T00:00:00Z"}),
        (3, {"id": "GO-2026-0002"}),
    ]
    assert following == [(3, {**PENDING[1][1], "page_token": "page-two"})]
    assert "page_token" not in PENDING[1][1]
    assert osv_page_outcome(OSV_PAGE_LIMIT - 1, raw, PENDING, {(0, "page-two")}) == (
        findings,
        following,
    )
    assert osv_page_outcome(OSV_PAGE_LIMIT, b'{"results": [{}, {}]}', PENDING, set()) == ([], [])


@pytest.mark.parametrize(
    ("payload", "refusal"),
    [
        ([], "every requested identity"),
        ({"results": {}}, "every requested identity"),
        ({"results": [{}]}, "every requested identity"),
        ({"results": [{}, {}, {}]}, "every requested identity"),
        ({"results": [{}, "unanswered"]}, "unanswered package query"),
        ({"results": [{}, {"error": {"code": 3}}]}, "unanswered package query"),
        ({"results": [{"vulns": {"id": "GO-2026-0001"}}, {}]}, "records are invalid"),
        ({"results": [{"vulns": ["GO-2026-0001"]}, {}]}, "identity is invalid"),
        ({"results": [{"vulns": [{"id": 1}]}, {}]}, "identity is invalid"),
        ({"results": [{"vulns": [{"id": ""}]}, {}]}, "identity is empty"),
        ({"results": [{}, {"next_page_token": 2}]}, "invalid or repeated"),
        ({"results": [{}, {"next_page_token": ""}]}, "invalid or repeated"),
        ({"results": [{}, {"next_page_token": "followed"}]}, "invalid or repeated"),
    ],
)
def test_osv_page_refuses_incomplete_or_invalid_answers(payload: object, refusal: str) -> None:
    """A page that omits, refuses or malforms an answer cannot count as coverage.

    Parameters
    ----------
    payload : object
        Decoded response body that does not answer both submitted queries.
    refusal : str
        Refusal that names the incomplete part of the response.
    """
    with pytest.raises(ValueError, match=refusal):
        osv_page_outcome(1, json.dumps(payload).encode(), PENDING, {(3, "followed")})


def test_osv_page_refuses_oversized_bodies_and_unbounded_continuation() -> None:
    """Refuse a body beyond the evidence limit and a continuation past the page budget."""
    with pytest.raises(ValueError, match="evidence limit"):
        osv_page_outcome(1, b" " * (OSV_RESPONSE_LIMIT + 1), PENDING, set())
    continued = json.dumps({"results": [{}, {"next_page_token": "further"}]}).encode()
    with pytest.raises(ValueError, match="bounded page budget"):
        osv_page_outcome(OSV_PAGE_LIMIT, continued, PENDING, set())
