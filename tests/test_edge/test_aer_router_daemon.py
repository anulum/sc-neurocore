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
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from sc_neurocore.bridges.aer_router import SpikePacket
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
        'package main\n\nimport ("os"; "time")\n\nfunc main() {\n'
        '\t_ = os.WriteFile("started.txt", []byte("built"), 0o644)\n\ttime.Sleep(time.Hour)\n}\n',
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


@posix_only
def test_supervisor_passes_port_to_go_process(tmp_path: Path) -> None:
    """The launched program receives the requested port instead of its default."""
    router = tmp_path / "aer_router"
    router.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > args.txt\nexec sleep 30\n',
        encoding="utf-8",
    )
    router.chmod(0o755)
    daemon = aer_router.AERRoutingDaemon(port=9103, router_dir=tmp_path)
    daemon.start(build=False)
    try:
        assert _read_when_written(tmp_path / "args.txt").splitlines() == ["--port", "9103"]
    finally:
        daemon.stop()


@posix_only
def test_stop_kills_and_reaps_process_that_ignores_termination(tmp_path: Path) -> None:
    """The supervisor reaps a child that keeps running after SIGTERM."""
    router = tmp_path / "aer_router"
    router.write_text(
        "#!/bin/sh\ntrap '' TERM\necho ready > ready.txt\nexec sleep 30\n",
        encoding="utf-8",
    )
    router.chmod(0o755)
    daemon = aer_router.AERRoutingDaemon(router_dir=tmp_path)
    daemon.start(build=False)
    process = daemon._process
    assert process is not None
    try:
        assert _read_when_written(tmp_path / "ready.txt").strip() == "ready"
        daemon.stop()
        assert process.poll() is not None
        assert daemon._process is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)


@needs_go
def test_configured_go_router_ignores_stale_ack_after_restart(tmp_path: Path) -> None:
    """Keep the destination live across two Go processes on the same UDP port."""
    shutil.copyfile(aer_router._PACKAGE_ROUTER_DIR / "main.go", tmp_path / "main.go")
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as destination,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as source,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation,
    ):
        destination.bind(("127.0.0.1", 0))
        destination.settimeout(3.0)
        source.bind(("127.0.0.1", 0))
        source.settimeout(3.0)
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        reservation.close()

        target = f"127.0.0.1:{destination.getsockname()[1]}"
        router_address = ("127.0.0.1", port)
        daemon = aer_router.AERRoutingDaemon(port=port, router_dir=tmp_path, routes={12: target})
        packet = SpikePacket(source_id=8, target_id=12, timestamp=17, spike_len=9, sequence=-42)
        data = packet.encode()
        daemon.start()
        try:
            daemon.start(build=False)
            source.sendto(data, router_address)
            forwarded, address = destination.recvfrom(64)
            assert address[1] == port
            delivery = SpikePacket.decode(forwarded)
            assert delivery.sequence > 0
            assert delivery == replace(packet, sequence=delivery.sequence)
            ack, _ = source.recvfrom(64)
            assert ack == data[-8:]
            stale_ack = forwarded[-8:]
        finally:
            daemon.stop()

        daemon.start(build=False)
        try:
            source.sendto(data, router_address)
            forwarded, address = destination.recvfrom(64)
            assert address[1] == port
            current_ack = forwarded[-8:]
            assert current_ack != stale_ack
            ack, _ = source.recvfrom(64)
            assert ack == data[-8:]

            # The destination sends all three frames through the same loopback socket.
            barrier = replace(packet, sequence=-43, timestamp=18)
            destination.sendto(stale_ack, address)
            destination.sendto(data, address)
            destination.sendto(barrier.encode(), address)
            forwarded, _ = destination.recvfrom(64)
            assert SpikePacket.decode(forwarded).timestamp == barrier.timestamp
            ack, _ = destination.recvfrom(64)
            assert ack == barrier.encode()[-8:]

            destination.sendto(current_ack, address)
            destination.sendto(data, address)
            forwarded, _ = destination.recvfrom(64)
            new_delivery = SpikePacket.decode(forwarded)
            assert new_delivery.sequence > 0
            assert new_delivery == replace(packet, sequence=new_delivery.sequence)
            assert forwarded[-8:] != current_ack
            ack, _ = destination.recvfrom(64)
            assert ack == data[-8:]
        finally:
            daemon.stop()


@posix_only
def test_start_refuses_a_child_that_exits_during_startup(tmp_path: Path) -> None:
    """An early child failure is reported and leaves no tracked process."""
    router = tmp_path / "aer_router"
    router.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    router.chmod(0o755)
    daemon = aer_router.AERRoutingDaemon(port=9103, router_dir=tmp_path)
    with pytest.raises(RuntimeError, match=r"UDP port 9103 \(exit 7\)"):
        daemon.start(build=False)
    assert daemon._process is None


@posix_only
def test_stop_clears_an_already_exited_child(tmp_path: Path) -> None:
    """Stopping an already-exited process clears its supervisor state."""
    _stand_in_router(tmp_path)
    daemon = aer_router.AERRoutingDaemon(port=9104, router_dir=tmp_path)
    daemon.start(build=False)
    process = daemon._process
    assert process is not None
    try:
        process.terminate()
        process.wait(timeout=2.0)
    finally:
        daemon.stop()
    assert daemon._process is None
