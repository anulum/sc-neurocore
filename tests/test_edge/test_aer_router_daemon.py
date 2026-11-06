# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AER router daemon lifecycle tests

"""Lifecycle tests for the AER router supervisor with real router processes.

The router directory is a test directory holding either a stand-in ``aer_router``
shell script or a small Go program built with the real ``go`` toolchain, so the
supervisor's build, start and stop run real processes and log what they do.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sc_neurocore.edge import aer_router

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="a POSIX shell script stands in for the router binary"
)
needs_go = pytest.mark.skipif(
    shutil.which("go") is None, reason="the Go toolchain is not installed"
)


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the supervisor's INFO messages captured so far."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == aer_router.__name__ and record.levelno == logging.INFO
    ]


def _read_when_written(path: Path, timeout: float = 10.0) -> str:
    """Return a file's text once the router process has written it.

    Parameters
    ----------
    path : pathlib.Path
        File the router writes on start.
    timeout : float, optional
        Seconds to wait before failing.

    Returns
    -------
    str
        The file's text.
    """
    deadline = time.monotonic() + timeout
    while not path.is_file() or not path.read_text(encoding="utf-8"):
        assert time.monotonic() < deadline, f"{path.name} was not written"
        time.sleep(0.05)
    return path.read_text(encoding="utf-8")


def _stand_in_router(directory: Path) -> Path:
    """Write an ``aer_router`` script that records its directory and then waits.

    Parameters
    ----------
    directory : pathlib.Path
        Router directory that receives the script.

    Returns
    -------
    pathlib.Path
        File the script writes its working directory to once it runs.
    """
    router = directory / "aer_router"
    router.write_text("#!/bin/sh\npwd > started.txt\nexec sleep 30\n", encoding="utf-8")
    router.chmod(0o755)
    return directory / "started.txt"


def test_default_router_dir_is_the_packaged_go_service() -> None:
    """Without ``router_dir`` the supervisor uses the router shipped with the package."""
    daemon = aer_router.AERRoutingDaemon(port=9101)
    expected = (
        Path(aer_router.__file__).resolve().parent.parent
        / "accel"
        / "go"
        / "services"
        / "aer_router"
    )
    assert daemon._router_dir == expected
    assert (expected / "main.go").is_file()
    assert daemon._port == 9101


@posix_only
def test_start_without_build_runs_the_router_and_stop_ends_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``start(build=False)`` runs the existing binary in its directory; ``stop`` ends it."""
    started = _stand_in_router(tmp_path)
    daemon = aer_router.AERRoutingDaemon(port=9101, router_dir=tmp_path)
    with caplog.at_level(logging.INFO, logger=aer_router.__name__):
        daemon.start(build=False)
        process = daemon._process
        try:
            assert process is not None and process.poll() is None
            assert _read_when_written(started).strip() == str(tmp_path)
        finally:
            # A failed check must not leave the router running.
            daemon.stop()
    assert process.poll() is not None
    assert daemon._process is None
    assert _messages(caplog) == [
        "Starting the AER router on UDP port 9101",
        "Stopped the AER router",
    ]


@posix_only
@needs_go
def test_start_builds_the_router_with_go_before_running_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``start()`` builds ``main.go`` into ``aer_router`` with Go, then runs that binary."""
    (tmp_path / "main.go").write_text(
        'package main\n\nimport "os"\n\nfunc main() {\n'
        '\t_ = os.WriteFile("started.txt", []byte("built"), 0o644)\n\tselect {}\n}\n',
        encoding="utf-8",
    )
    daemon = aer_router.AERRoutingDaemon(port=9102, router_dir=tmp_path)
    with caplog.at_level(logging.INFO, logger=aer_router.__name__):
        daemon.start()
        try:
            assert (tmp_path / "aer_router").is_file()
            assert _read_when_written(tmp_path / "started.txt") == "built"
        finally:
            daemon.stop()
    assert _messages(caplog) == [
        f"Building the Go AER router in {tmp_path}",
        "Starting the AER router on UDP port 9102",
        "Stopped the AER router",
    ]


@needs_go
def test_a_failed_go_build_raises_and_starts_nothing(tmp_path: Path) -> None:
    """A Go build failure is raised and no router process is left behind."""
    (tmp_path / "main.go").write_text(
        "package main\n\nfunc main() { undefined() }\n", encoding="utf-8"
    )
    daemon = aer_router.AERRoutingDaemon(router_dir=tmp_path)
    try:
        with pytest.raises(subprocess.CalledProcessError):
            daemon.start()
        assert daemon._process is None
    finally:
        # Should a router start after all, it must not outlive the test.
        daemon.stop()


def test_stop_without_a_running_router_does_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """``stop`` before ``start`` changes nothing and logs nothing."""
    daemon = aer_router.AERRoutingDaemon()
    with caplog.at_level(logging.INFO, logger=aer_router.__name__):
        daemon.stop()
    assert daemon._process is None
    assert _messages(caplog) == []
