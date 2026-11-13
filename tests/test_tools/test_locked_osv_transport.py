# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real OSV HTTP and connection failure custody

"""Preserve real provider refusals and socket failures through public audit surfaces."""

from __future__ import annotations

from email.message import Message
import hashlib
import io
import json
import os
from pathlib import Path
import select
import shutil
import socket
import ssl
import subprocess
import sys
import threading
from typing import Any
import urllib.error
import urllib.request
import urllib.response

import pytest

from tools.security_scan.locked_dependency_reports import (
    OSV_ATTEMPTS,
    OSV_ENDPOINT,
    OSV_PAGE_LIMIT,
    OSV_RESPONSE_LIMIT,
    OSV_RETRY_DELAYS,
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
    refused = {
        "http_status": None,
        "error_type": "URLError",
        "reason_type": "ConnectionRefusedError",
    }
    assert receipt["retried_attempts"] == [refused] * (OSV_ATTEMPTS - 1)
    assert receipt["request_sha256"] == hashlib.sha256(request).hexdigest()
    assert "response_sha256" not in receipt
    assert not (target / "lock-1/response-1.json").exists()
    assert lock.read_bytes() == original


class _FlakyConnectProxy:
    """A real CONNECT proxy that refuses its first tunnels with 503 and relays the later ones."""

    def __init__(self, refusals: int) -> None:
        """Listen on a free local port and serve in background threads.

        Parameters
        ----------
        refusals : int
            How many CONNECT requests, counted from the first, get a 503.
        """
        self.server = socket.create_server(("127.0.0.1", 0))
        self.refusals = refusals
        self.targets: list[str] = []
        threading.Thread(target=self._serve, daemon=True).start()

    @property
    def url(self) -> str:
        """Return the proxy's URL for a `ProxyHandler`."""
        return f"http://127.0.0.1:{self.server.getsockname()[1]}"

    def _serve(self) -> None:
        """Accept connections until the listening socket closes."""
        while True:
            try:
                client, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(client,), daemon=True).start()

    def _handle(self, client: socket.socket) -> None:
        """Answer one CONNECT: refuse it, or tunnel it to the OSV host only.

        Parameters
        ----------
        client : socket.socket
            The accepted client connection.
        """
        with client:
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = client.recv(4096)
                if not chunk:
                    return
                head += chunk
            target = head.split(b" ", 2)[1].decode("ascii")
            self.targets.append(target)
            if len(self.targets) <= self.refusals or target != "api.osv.dev:443":
                client.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                return
            with socket.create_connection(("api.osv.dev", 443), timeout=30) as upstream:
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                ends = [client, upstream]
                while True:
                    readable, _, _ = select.select(ends, [], [], 30)
                    if not readable:
                        return
                    for end in readable:
                        data = end.recv(65536)
                        if not data:
                            return
                        (upstream if end is client else client).sendall(data)

    def close(self) -> None:
        """Stop accepting connections."""
        self.server.close()


def test_a_refused_tunnel_is_retried_and_the_real_provider_answers(tmp_path: Path) -> None:
    """A real 503 from the proxy is retried; the second attempt reaches OSV itself.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory retaining the actual request, response and receipt.

    Notes
    -----
    The proxy is a real local process thread and the provider is the official
    OSV endpoint; no response is substituted. Only the wait is recorded
    instead of slept.
    """
    proxy = _FlakyConnectProxy(refusals=1)
    waits: list[float] = []
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy.url}))
    try:
        report = audit_osv_queries(
            ({"package": {"ecosystem": "PyPI", "name": "jinja2"}, "version": "3.1.6"},),
            tmp_path / "packet",
            opener=opener.open,
            sleep=waits.append,
        )
    finally:
        proxy.close()
    assert proxy.targets == ["api.osv.dev:443", "api.osv.dev:443"]
    assert waits == [OSV_RETRY_DELAYS[0]]
    assert report["coverage_complete"] is True and len(report["pages"]) == 1
    receipt = json.loads((tmp_path / "packet/response-1-receipt.json").read_text())
    assert receipt["http_status"] == 200
    assert receipt["retried_attempts"] == [
        {"http_status": None, "error_type": "URLError", "reason_type": "OSError"}
    ]
    response = (tmp_path / "packet/response-1.json").read_bytes()
    assert receipt["response_sha256"] == hashlib.sha256(response).hexdigest()


class _PreparedOutcomes:
    """Answer each request with the next prepared outcome and keep every body sent.

    The provider cannot be made to return 429 or 5xx on demand, so these tests
    drive the retry policy with prepared outcomes in place of the transport.
    """

    def __init__(self, outcomes: list[tuple[int, bytes] | OSError]) -> None:
        """Keep the outcomes in the order the attempts will meet them.

        Parameters
        ----------
        outcomes : list
            `(status, body)` pairs, or exceptions to raise.
        """
        self.outcomes = outcomes
        self.sent: list[bytes] = []

    def __call__(self, request: urllib.request.Request, /, *, timeout: float) -> Any:
        """Return or raise the next outcome, as `urlopen` would.

        Parameters
        ----------
        request : urllib.request.Request
            The prepared OSV request.
        timeout : float
            The socket timeout the caller asked for.

        Returns
        -------
        Any
            A response object for a 200 outcome.
        """
        assert request.full_url == OSV_ENDPOINT and timeout > 0
        assert isinstance(request.data, bytes)
        self.sent.append(request.data)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, OSError):
            raise outcome
        status, body = outcome
        if status != 200:
            raise urllib.error.HTTPError(
                OSV_ENDPOINT, status, "refused", Message(), io.BytesIO(body)
            )
        return urllib.response.addinfourl(io.BytesIO(body), Message(), OSV_ENDPOINT, code=200)


QUERY = ({"package": {"ecosystem": "PyPI", "name": "jinja2"}, "version": "3.1.6"},)
CLEAN = b'{"results": [{}]}'


def test_transport_failures_and_busy_answers_are_retried_and_kept(tmp_path: Path) -> None:
    """A timeout, then a 503, then an answer: the audit passes with both failures kept.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory for the audit packet.
    """
    outcomes = _PreparedOutcomes(
        [urllib.error.URLError(TimeoutError()), (503, b"busy"), (200, CLEAN)]
    )
    waits: list[float] = []
    report = audit_osv_queries(QUERY, tmp_path / "packet", opener=outcomes, sleep=waits.append)
    assert report["passed"] is True and report["coverage_complete"] is True
    assert waits == list(OSV_RETRY_DELAYS)
    assert len(set(outcomes.sent)) == 1 and len(outcomes.sent) == 3
    receipt = json.loads((tmp_path / "packet/response-1-receipt.json").read_text())
    assert receipt["http_status"] == 200
    assert receipt["retried_attempts"] == [
        {"http_status": None, "error_type": "URLError", "reason_type": "TimeoutError"},
        {
            "http_status": 503,
            "response_bytes": 4,
            "response_sha256": hashlib.sha256(b"busy").hexdigest(),
        },
    ]
    assert (tmp_path / "packet/response-1-attempt-2.json").read_bytes() == b"busy"
    assert (tmp_path / "packet/response-1.json").read_bytes() == CLEAN


def test_a_refusal_that_is_not_transient_is_final_at_once(tmp_path: Path) -> None:
    """A 400 is not retried: one attempt, no wait, the refusal body kept.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory for the audit packet.
    """
    outcomes = _PreparedOutcomes([(400, b'{"code": 3}')])
    waits: list[float] = []
    with pytest.raises(ValueError, match="HTTP 200"):
        audit_osv_queries(QUERY, tmp_path / "packet", opener=outcomes, sleep=waits.append)
    assert waits == [] and len(outcomes.sent) == 1
    receipt = json.loads((tmp_path / "packet/response-1-receipt.json").read_text())
    assert receipt["http_status"] == 400 and receipt["retried_attempts"] == []
    assert (tmp_path / "packet/response-1.json").read_bytes() == b'{"code": 3}'


def test_busy_answers_stop_at_the_attempt_bound(tmp_path: Path) -> None:
    """Three busy answers: two waits, then the last status refuses the audit.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory for the audit packet.
    """
    outcomes = _PreparedOutcomes([(503, b"a"), (429, b"b"), (504, b"c")])
    waits: list[float] = []
    with pytest.raises(ValueError, match="HTTP 200"):
        audit_osv_queries(QUERY, tmp_path / "packet", opener=outcomes, sleep=waits.append)
    assert waits == list(OSV_RETRY_DELAYS) and len(outcomes.sent) == OSV_ATTEMPTS
    receipt = json.loads((tmp_path / "packet/response-1-receipt.json").read_text())
    assert receipt["http_status"] == 504
    assert [row["http_status"] for row in receipt["retried_attempts"]] == [503, 429]
    assert (tmp_path / "packet/response-1.json").read_bytes() == b"c"


def test_transport_failures_stop_at_the_attempt_bound(tmp_path: Path) -> None:
    """Three transport failures: the last one is raised and only types are kept.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory for the audit packet.
    """
    outcomes = _PreparedOutcomes(
        [TimeoutError("first"), urllib.error.URLError(ConnectionResetError()), TimeoutError("last")]
    )
    waits: list[float] = []
    with pytest.raises(TimeoutError, match="last"):
        audit_osv_queries(QUERY, tmp_path / "packet", opener=outcomes, sleep=waits.append)
    assert waits == list(OSV_RETRY_DELAYS)
    receipt = json.loads((tmp_path / "packet/response-1-receipt.json").read_text())
    assert receipt["http_status"] is None
    assert (receipt["error_type"], receipt["reason_type"]) == ("TimeoutError", "TimeoutError")
    assert [(row["error_type"], row["reason_type"]) for row in receipt["retried_attempts"]] == [
        ("TimeoutError", "TimeoutError"),
        ("URLError", "ConnectionResetError"),
    ]
    assert "first" not in json.dumps(receipt) and "last" not in json.dumps(receipt)
    assert not (tmp_path / "packet/response-1.json").exists()


def _certificate_failure() -> ssl.SSLCertVerificationError:
    """Return a certificate failure carrying the text code the ssl module sets as `reason`."""
    failure = ssl.SSLCertVerificationError(1, "bad certificate")
    failure.reason = "CERTIFICATE_VERIFY_FAILED"
    failure.library = "SSL"
    return failure


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (urllib.error.URLError(_certificate_failure()), "URLError"),
        (_certificate_failure(), "SSLCertVerificationError"),
    ],
)
def test_a_failed_certificate_check_is_final_at_once(
    tmp_path: Path, failure: OSError, error_type: str
) -> None:
    """A certificate that fails verification is not transient: one attempt, no wait.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory for the audit packet.
    failure : OSError
        The certificate failure as urlopen wraps it, and as raised directly.
    error_type : str
        The exception type the receipt must name.
    """
    outcomes = _PreparedOutcomes([failure, (200, CLEAN)])
    waits: list[float] = []
    with pytest.raises(OSError):
        audit_osv_queries(QUERY, tmp_path / "packet", opener=outcomes, sleep=waits.append)
    assert waits == [] and len(outcomes.sent) == 1
    receipt = json.loads((tmp_path / "packet/response-1-receipt.json").read_text())
    assert receipt["retried_attempts"] == []
    assert (receipt["error_type"], receipt["reason_type"]) == (
        error_type,
        "SSLCertVerificationError",
    )
    assert "bad certificate" not in json.dumps(receipt)


def test_another_ssl_failure_is_retried(tmp_path: Path) -> None:
    """An ssl failure other than a failed certificate check, such as an early EOF, is retried.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory for the audit packet.
    """
    outcomes = _PreparedOutcomes([urllib.error.URLError(ssl.SSLEOFError(8, "EOF")), (200, CLEAN)])
    waits: list[float] = []
    report = audit_osv_queries(QUERY, tmp_path / "packet", opener=outcomes, sleep=waits.append)
    assert report["passed"] is True and len(outcomes.sent) == 2
    assert waits == [OSV_RETRY_DELAYS[0]]
    receipt = json.loads((tmp_path / "packet/response-1-receipt.json").read_text())
    assert receipt["retried_attempts"] == [
        {"http_status": None, "error_type": "URLError", "reason_type": "SSLEOFError"}
    ]


def test_an_answer_after_a_retry_keeps_its_findings_and_blocks(tmp_path: Path) -> None:
    """An advisory in the answer after a busy attempt is kept and fails the audit.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory for the audit packet.
    """
    record = {"id": "PYSEC-2026-0001", "modified": "2026-10-01T00:00:00Z"}
    answer = json.dumps({"results": [{"vulns": [record]}]}).encode()
    outcomes = _PreparedOutcomes([(502, b"gateway"), (200, answer)])
    waits: list[float] = []
    report = audit_osv_queries(QUERY, tmp_path / "packet", opener=outcomes, sleep=waits.append)
    assert waits == [OSV_RETRY_DELAYS[0]]
    assert report["passed"] is False and report["coverage_complete"] is True
    assert report["results"] == [[record]]
    assert report["pages"] == [{"number": 1, "sha256": hashlib.sha256(answer).hexdigest()}]


def test_every_retry_has_a_bounded_wait() -> None:
    """One wait between each pair of attempts, each positive and short."""
    assert len(OSV_RETRY_DELAYS) == OSV_ATTEMPTS - 1
    assert all(0 < delay <= 10 for delay in OSV_RETRY_DELAYS)


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
