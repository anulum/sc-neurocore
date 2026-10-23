# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop HTTP admission through real transport faults

"""Exercise real HTTP refusal, elapsed admission deadlines and late cancellation."""

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.runtime import DesktopBackend, DesktopInstance
from tests.test_studio_desktop_runtime import _environment, _request


@pytest.fixture
def admitted_profile(tmp_path: Path) -> tuple[dict[str, str], DesktopInstance, bytes]:
    """Retain an origin admitted by the actual backend before transport faults.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Private home and XDG directories for real startup and recovery.

    Returns
    -------
    tuple[dict[str, str], DesktopInstance, bytes]
        Process environment, actual admitted identity and retained configuration.
    """
    env = _environment(tmp_path)
    backend = DesktopBackend(env)
    try:
        instance = backend.start()
        configuration = (backend.paths.config / "desktop.json").read_bytes()
    finally:
        report = backend.close()
        assert report is not None and report.reaped and report.returncode == 0
    return env, instance, configuration


@pytest.mark.parametrize(
    "fault,message",
    [
        ("none", None),
        ("health-json", "HTTP admission failed"),
        ("health-status", "did not report healthy startup"),
        ("ui-digest", "interface does not match this installation"),
    ],
)
def test_transport_corruption_refuses_the_real_application_and_retains_its_origin(
    admitted_profile: tuple[dict[str, str], DesktopInstance, bytes],
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    message: str | None,
) -> None:
    """Forward actual application responses through a deliberately corrupting hop.

    Parameters
    ----------
    admitted_profile : tuple[dict[str, str], DesktopInstance, bytes]
        Successfully admitted real profile, without a running backend.
    monkeypatch : pytest.MonkeyPatch
        Redirect only this owner's real socket connections through the local relay.
    fault : str
        Forward unchanged, or corrupt health JSON/status or served UI bytes.
    message : str or None
        Expected public refusal, or successful admission for the control case.
    """
    env, first, original = admitted_profile
    target = ("127.0.0.1", int(first.origin.rsplit(":", 1)[1]))
    responses: list[tuple[str, int, bytes, bytes]] = []
    completed: list[str] = []

    class Relay(BaseHTTPRequestHandler):
        """Forward to the real backend and record original versus transmitted bytes."""

        def do_GET(self) -> None:
            """Fetch an actual response before applying the selected wire corruption."""
            connection = http.client.HTTPConnection(*target, timeout=2)
            try:
                connection.request("GET", self.path)
                response = connection.getresponse()
                actual = response.read(1_048_577)
                outgoing = actual
                if self.path == "/api/health" and fault == "health-json":
                    outgoing = b"{"
                elif self.path == "/api/health" and fault == "health-status":
                    outgoing = b'{"status":"warming"}'
                elif self.path == "/studios/sc-neurocore/" and fault == "ui-digest":
                    outgoing += b"\n"
                responses.append((self.path, response.status, actual, outgoing))
                self.send_response(response.status)
                self.send_header("Content-Length", str(len(outgoing)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(outgoing)
                self.wfile.flush()
                completed.append(self.path)
            finally:
                connection.close()

    relay = ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    relay_address = ("127.0.0.1", int(relay.server_address[1]))
    relay_thread = threading.Thread(
        target=lambda: relay.serve_forever(poll_interval=0.05), daemon=True
    )
    relay_thread.start()
    normal_connection = socket.create_connection
    owner_thread = threading.get_ident()

    def connect(
        address: tuple[str, int],
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        """Retain actual socket creation, redirecting only the owner's target hop."""
        destination = (
            relay_address
            if address == target and threading.get_ident() == owner_thread
            else address
        )
        return normal_connection(destination, timeout, source_address)

    backend = DesktopBackend(env)
    try:
        with monkeypatch.context() as transport:
            transport.setattr(socket, "create_connection", connect)
            if message is None:
                instance = backend.start()
                assert instance.origin == first.origin
                assert instance.ui_sha256 == first.ui_sha256
            else:
                with pytest.raises(DesktopStartRefused, match=message):
                    backend.start()
            assert (backend.paths.config / "desktop.json").read_bytes() == original
    finally:
        try:
            report = backend.close()
            assert report is not None and report.reaped and report.returncode == 0
        finally:
            relay.shutdown()
            relay.server_close()
            relay_thread.join(timeout=5)
            assert not relay_thread.is_alive()
    assert len(responses) == (2 if fault in {"none", "ui-digest"} else 1)
    assert completed == [row[0] for row in responses]
    assert all(row[1] == 200 for row in responses)
    assert responses[0][0] == "/api/health" and json.loads(responses[0][2]) == {"status": "ok"}
    if len(responses) == 2:
        assert responses[1][0] == "/studios/sc-neurocore/"
        assert hashlib.sha256(responses[1][2]).hexdigest() == first.ui_sha256
    resumed = DesktopBackend(env)
    try:
        recovered = resumed.start()
        assert recovered.origin == first.origin
        assert _request(recovered.origin, "/api/health") == {"status": "ok"}
        assert (resumed.paths.config / "desktop.json").read_bytes() == original
    finally:
        collected = resumed.close()
        assert collected is not None and collected.reaped and collected.returncode == 0


@pytest.mark.parametrize("after_health", [False, True], ids=["before-http", "after-health"])
def test_elapsed_http_admission_deadline_collects_the_real_backend(
    admitted_profile: tuple[dict[str, str], DesktopInstance, bytes],
    monkeypatch: pytest.MonkeyPatch,
    after_health: bool,
) -> None:
    """Use actual elapsed time before HTTP admission or after a real health read.

    Parameters
    ----------
    admitted_profile : tuple[dict[str, str], DesktopInstance, bytes]
        Actual successfully retained origin and configuration.
    monkeypatch : pytest.MonkeyPatch
        Preserve the normal client operation while delaying its owner.
    after_health : bool
        Delay after real health bytes, or before the first actual HTTP request.
    """
    env, _, original = admitted_profile
    backend = DesktopBackend(env)
    observed: list[bytes] = []
    normal_opener = urllib.request.build_opener
    normal_read = http.client.HTTPResponse.read
    started = time.monotonic()

    def delayed_opener(
        *handlers: urllib.request.BaseHandler | type[urllib.request.BaseHandler],
    ) -> urllib.request.OpenerDirector:
        """Build the genuine opener after a real scheduling pause past the deadline."""
        monkeypatch.setattr(urllib.request, "build_opener", normal_opener)
        observed.append(b"before-http")
        time.sleep(max(0.0, started + 35 - time.monotonic()))
        return normal_opener(*handlers)

    def delayed_read(response: http.client.HTTPResponse, amount: int | None = None) -> bytes:
        """Return the actual health bytes after the owner resumes past its deadline."""
        actual = normal_read(response, amount)
        monkeypatch.setattr(http.client.HTTPResponse, "read", normal_read)
        assert json.loads(actual) == {"status": "ok"}
        observed.append(actual)
        time.sleep(max(0.0, started + 35 - time.monotonic()))
        return actual

    if after_health:
        monkeypatch.setattr(http.client.HTTPResponse, "read", delayed_read)
    else:
        monkeypatch.setattr(urllib.request, "build_opener", delayed_opener)
    try:
        with pytest.raises(DesktopStartRefused, match="readiness deadline expired"):
            backend.start(timeout_seconds=30)
        assert len(observed) == 1
        assert (backend.paths.config / "desktop.json").read_bytes() == original
    finally:
        report = backend.close()
        assert report is not None and report.reaped and report.returncode == 0
    assert backend.returncode == 0


def test_cancellation_after_actual_ui_read_keeps_the_previous_origin(
    admitted_profile: tuple[dict[str, str], DesktopInstance, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel through the public owner after both actual HTTP admissions succeed.

    Parameters
    ----------
    admitted_profile : tuple[dict[str, str], DesktopInstance, bytes]
        Real profile whose previously admitted configuration must survive cancellation.
    monkeypatch : pytest.MonkeyPatch
        Preserve actual HTTP reads and request cancellation after the served UI bytes.
    """
    env, first, original = admitted_profile
    backend = DesktopBackend(env)
    normal_read = http.client.HTTPResponse.read
    observed: list[str] = []

    def read_and_cancel(response: http.client.HTTPResponse, amount: int | None = None) -> bytes:
        """Keep real response bytes and cancel only after the complete actual UI read."""
        actual = normal_read(response, amount)
        observed.append(response.geturl())
        if response.geturl().endswith("/studios/sc-neurocore/"):
            assert hashlib.sha256(actual).hexdigest() == first.ui_sha256
            monkeypatch.setattr(http.client.HTTPResponse, "read", normal_read)
            backend.cancel_startup()
        return actual

    monkeypatch.setattr(http.client.HTTPResponse, "read", read_and_cancel)
    try:
        with pytest.raises(DesktopStartRefused, match="startup was cancelled"):
            backend.start()
        assert observed == [first.origin + "/api/health", first.origin + "/studios/sc-neurocore/"]
        assert (backend.paths.config / "desktop.json").read_bytes() == original
    finally:
        report = backend.close()
        assert report is not None and report.reaped and report.returncode == 0
