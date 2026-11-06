# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — crates.io release status contracts

"""Read the engine crate identity and the registry answer exactly as the publish jobs do.

Registry answers come from a real loopback HTTP server that records each request, so the
identifying User-Agent and the requested path are observed rather than assumed.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import http.server
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from tools.crates_io_release_status import (
    USER_AGENT,
    crate_identity,
    main,
    release_status,
)

ROOT = Path(__file__).resolve().parents[2]


class _Registry(http.server.ThreadingHTTPServer):
    """Loopback registry answering every request with one status and body."""

    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(("127.0.0.1", 0), _RegistryHandler)
        self.status = status
        self.body = body
        self.requests: list[tuple[str, str | None]] = []


class _RegistryHandler(http.server.BaseHTTPRequestHandler):
    """Record the request path and User-Agent, then send the configured answer."""

    server: _Registry

    def do_GET(self) -> None:
        """Answer one crate-version query."""
        self.server.requests.append((self.path, self.headers.get("User-Agent")))
        self.send_response(self.server.status)
        self.send_header("Content-Length", str(len(self.server.body)))
        self.end_headers()
        self.wfile.write(self.server.body)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the server's request log out of the test output."""


@contextmanager
def registry(status: int, body: bytes = b"{}") -> Iterator[tuple[str, _Registry]]:
    """Serve one registry answer on an ephemeral loopback port.

    Parameters
    ----------
    status : int
        HTTP status of every answer.
    body : bytes, optional
        Response body of every answer.

    Yields
    ------
    tuple
        The API base address and the running server.
    """
    server = _Registry(status, body)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/api/v1/crates", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_the_engine_manifest_names_its_crate_and_version() -> None:
    """The committed engine manifest gives the identity the publish job uploads."""
    name, version = crate_identity(ROOT / "engine" / "Cargo.toml")
    assert name == "sc_neurocore_engine"
    assert version.count(".") == 2


@pytest.mark.parametrize(
    "manifest",
    [
        "[workspace]\nmembers = []\n",
        '[package]\nname = "x"\nversion.workspace = true\n',
        '[package]\nname = ""\nversion = "1.0.0"\n',
        '[package]\nversion = "1.0.0"\n',
    ],
)
def test_a_manifest_without_a_literal_identity_is_refused(tmp_path: Path, manifest: str) -> None:
    """A missing package table, an inherited version or an empty name cannot be queried.

    Parameters
    ----------
    tmp_path : Path
        Directory of the candidate manifest.
    manifest : str
        Manifest text without a usable literal identity.
    """
    path = tmp_path / "Cargo.toml"
    path.write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError, match="no literal package name and version"):
        crate_identity(path)


@pytest.mark.parametrize(("status", "expected"), [(200, "published"), (404, "unpublished")])
def test_the_registry_answer_becomes_a_status(status: int, expected: str) -> None:
    """HTTP 200 and 404 are the two answers that decide whether to upload.

    Parameters
    ----------
    status : int
        Registry answer.
    expected : str
        Release status reported for it.
    """
    with registry(status) as (api, server):
        assert release_status("sc_neurocore_engine", "4.0.0", api) == expected
    assert server.requests == [("/api/v1/crates/sc_neurocore_engine/4.0.0", USER_AGENT)]


@pytest.mark.parametrize("status", [204, 403, 500])
def test_any_other_answer_stops_the_publication(status: int) -> None:
    """A refused, failed or unexpected answer is an error that keeps its status and body.

    Parameters
    ----------
    status : int
        Registry answer other than 200 or 404.
    """
    body = b"" if status == 204 else b"We require that all requests include a User-Agent."
    with registry(status, body) as (api, _), pytest.raises(RuntimeError) as caught:
        release_status("sc_neurocore_engine", "4.0.0", api)
    assert f"HTTP {status}" in str(caught.value)
    assert body.decode() in str(caught.value)


def test_the_user_agent_identifies_the_project_and_its_contact() -> None:
    """crates.io refuses anonymous clients, so every query names this project."""
    assert USER_AGENT.startswith("sc-neurocore-release ")
    assert "https://github.com/anulum/sc-neurocore" in USER_AGENT


def test_the_command_prints_only_the_status_on_standard_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The publish job reads standard output, so it carries the status word alone."""
    with registry(404) as (api, _):
        assert main([str(ROOT / "engine" / "Cargo.toml"), "--api", api]) == 0
    captured = capsys.readouterr()
    assert captured.out == "unpublished\n"
    assert "is unpublished on crates.io" in captured.err


@pytest.mark.parametrize("failure", ["manifest", "registry", "unreachable"])
def test_the_command_fails_without_a_status_when_it_cannot_decide(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    """A missing manifest, a refusing registry or no registry gives exit 1 and no status.

    Parameters
    ----------
    tmp_path : Path
        Directory without a manifest.
    capsys : pytest.CaptureFixture
        Captured standard output and error.
    failure : str
        The input that cannot produce a status.
    """
    manifest = str(ROOT / "engine" / "Cargo.toml")
    if failure == "manifest":
        assert main([str(tmp_path / "Cargo.toml")]) == 1
    elif failure == "registry":
        with registry(403, b"forbidden") as (api, _):
            assert main([manifest, "--api", api]) == 1
    else:
        with registry(200) as (api, _):
            closed = api
        assert main([manifest, "--api", closed]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "crates.io release status unavailable" in captured.err


def test_the_script_runs_as_the_workflow_calls_it() -> None:
    """``python3 tools/crates_io_release_status.py engine/Cargo.toml`` runs from the root."""
    with registry(200) as (api, _):
        completed = subprocess.run(
            [
                sys.executable,
                "tools/crates_io_release_status.py",
                "engine/Cargo.toml",
                "--api",
                api,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "published\n"
