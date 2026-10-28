# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio CLI startup integration

"""Check HTTP readiness at the real Studio CLI's browser-opener boundary."""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPENER = '''# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Browser-opener HTTP observation

"""Observe the served URL without opening or automating a GUI."""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

record = {"url": sys.argv[2]}
try:
    with urllib.request.urlopen(sys.argv[2], timeout=2) as response:
        record.update(status=response.status, body_bytes=len(response.read()))
except (urllib.error.URLError, TimeoutError) as error:
    record.update(error_type=type(error).__name__, error=str(error))
with Path(sys.argv[1]).open("a") as output:
    output.write(json.dumps(record) + "\\n")
'''


@contextmanager
def _launch_studio(
    root: Path, port: int, overrides: dict[str, str] | None = None
) -> Iterator[tuple[subprocess.Popen[str], Path]]:
    opener = root / "opener.py"
    marker = root / "opened.jsonl"
    opener.write_text(OPENER)
    temporary = root / "tmp"
    temporary.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SC_NEUROCORE_STUDIO_")
    }
    env.update(
        PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(temporary),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        SC_NEUROCORE_STUDIO_JOB_ROOT=str(root / "jobs"),
        SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH=str(root / "audit.jsonl"),
        SC_NEUROCORE_STUDIO_BENCHMARK_DATABANK=str(root / "benchmark.jsonl"),
        BROWSER=shlex.join((sys.executable, str(opener), str(marker), "%s")),
    )
    env.update(overrides or {})
    with (root / "cli.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "sc_neurocore.cli", "studio", "--port", str(port)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            yield process, marker
        finally:
            if process.poll() is None:
                # Exercise console shutdown; Uvicorn re-raises the received
                # signal after draining. SIGINT unwinds Python and preserves
                # subprocess coverage, whereas re-raised SIGTERM skips atexit.
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise AssertionError("Studio did not stop after termination") from None


def _wait_for_opening(process: subprocess.Popen[str], marker: Path) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if marker.exists() and marker.read_text().endswith("\n"):
            return
        assert process.poll() is None, (marker.parent / "cli.log").read_text()
        time.sleep(0.05)
    raise AssertionError("The serving Studio did not invoke its browser opener")


def test_studio_opens_a_serving_entry_once(tmp_path: Path) -> None:
    """Open an HTTP-ready entry once across two server notification periods.

    Parameters
    ----------
    tmp_path:
        Disposable launcher, opener-observation and runtime storage directory.
    """
    with socket.socket() as selected:
        selected.bind(("127.0.0.1", 0))
        port = selected.getsockname()[1]
    with _launch_studio(tmp_path, port) as (process, marker):
        _wait_for_opening(process, marker)
        opened = json.loads(marker.read_text())
        assert opened["status"] == 200, opened
        assert opened["body_bytes"] > 0
        assert opened["url"].startswith(f"http://127.0.0.1:{port}/")
        with urllib.request.urlopen(opened["url"], timeout=2) as response:
            assert response.status == 200
        time.sleep(32.2)
        assert process.poll() is None
        assert len(marker.read_text().splitlines()) == 1
    assert process.returncode is not None


def test_studio_bind_failure_does_not_open_a_browser(tmp_path: Path) -> None:
    """Refuse an occupied port before invoking any browser opener.

    Parameters
    ----------
    tmp_path:
        Disposable launcher, opener-observation and runtime storage directory.
    """
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        port = occupied.getsockname()[1]
        with _launch_studio(tmp_path, port) as (process, marker):
            assert process.wait(timeout=20) != 0
            assert not marker.exists()
    assert "address already in use" in (tmp_path / "cli.log").read_text().lower()


@pytest.mark.parametrize(
    ("port", "overrides", "diagnostic"),
    [
        (0, {}, "range 1..65535"),
        (65536, {}, "range 1..65535"),
        (
            8001,
            {"SC_NEUROCORE_STUDIO_CORS_ORIGINS": "http://localhost:8001"},
            "CORS origins must include the selected loopback URL",
        ),
        (
            8001,
            {"SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS": "http://localhost:8001"},
            "WebSocket origins must include the selected loopback URL",
        ),
    ],
)
def test_studio_invalid_launch_settings_do_not_open_a_browser(
    tmp_path: Path, port: int, overrides: dict[str, str], diagnostic: str
) -> None:
    """Refuse invalid ports and mismatched origins before opening any URL.

    Parameters
    ----------
    tmp_path:
        Disposable CLI process and observation directory.
    port:
        Selected CLI port, including both values outside the admitted range.
    overrides:
        Explicit operator configuration for the refused launch.
    diagnostic:
        Required caller-visible reason for refusal.
    """
    with _launch_studio(tmp_path, port, overrides) as (process, marker):
        assert process.wait(timeout=20) == 1
        assert not marker.exists()
    assert diagnostic in (tmp_path / "cli.log").read_text()


def test_studio_explicit_origins_with_policies_open_a_serving_entry(tmp_path: Path) -> None:
    """Serve the configured entry with route enforcement and explicit origins.

    Parameters
    ----------
    tmp_path:
        Disposable process, runtime storage and HTTP observation directory.
    """
    with socket.socket() as selected:
        selected.bind(("127.0.0.1", 0))
        port = selected.getsockname()[1]
    overrides = {
        "SC_NEUROCORE_STUDIO_CORS_ORIGINS": f"http://127.0.0.1:{port}",
        "SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "true",
        "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "false",
    }
    with _launch_studio(tmp_path, port, overrides) as (process, marker):
        _wait_for_opening(process, marker)
        opened = json.loads(marker.read_text())
        assert opened["status"] == 200, opened
        assert opened["body_bytes"] > 0
        assert opened["url"].startswith(f"http://127.0.0.1:{port}/")
    assert "UNPROTECTED local development" not in (tmp_path / "cli.log").read_text()
