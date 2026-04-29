# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AER router daemon lifecycle tests

"""Lifecycle contract tests for the high-level AER router daemon."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sc_neurocore.edge import aer_router


class _Process:
    def __init__(self) -> None:
        self.terminated = False
        self.wait_timeout: float | None = None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float) -> None:
        self.wait_timeout = timeout


def test_aer_router_start_builds_and_spawns_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    run_calls: list[dict[str, object]] = []
    popen_calls: list[dict[str, object]] = []
    sleeps: list[float] = []
    process = _Process()

    def run(args: list[str], cwd: str, check: bool) -> None:
        run_calls.append({"args": args, "cwd": cwd, "check": check})

    def popen(args: list[str], cwd: str, stdout: object, stderr: object) -> _Process:
        popen_calls.append({"args": args, "cwd": cwd, "stdout": stdout, "stderr": stderr})
        return process

    monkeypatch.setattr(aer_router.subprocess, "run", run)
    monkeypatch.setattr(aer_router.subprocess, "Popen", popen)
    monkeypatch.setattr(aer_router.time, "sleep", sleeps.append)

    daemon = aer_router.AERRoutingDaemon(port=9101)
    daemon.start(build=True)

    expected_dir = (
        Path(aer_router.__file__).resolve().parent.parent
        / "accel"
        / "go"
        / "services"
        / "aer_router"
    )
    assert daemon._router_dir == expected_dir
    assert daemon._port == 9101
    assert run_calls == [
        {
            "args": ["go", "build", "-o", "aer_router", "main.go"],
            "cwd": str(expected_dir),
            "check": True,
        }
    ]
    assert popen_calls == [
        {
            "args": ["./aer_router"],
            "cwd": str(expected_dir),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
    ]
    assert sleeps == [0.5]
    assert daemon._process is process


def test_aer_router_start_can_skip_build(monkeypatch: pytest.MonkeyPatch) -> None:
    built = {"called": False}
    spawned: list[list[str]] = []

    def run(*args: object, **kwargs: object) -> None:
        built["called"] = True

    def popen(args: list[str], **kwargs: object) -> _Process:
        spawned.append(args)
        return _Process()

    monkeypatch.setattr(aer_router.subprocess, "run", run)
    monkeypatch.setattr(aer_router.subprocess, "Popen", popen)
    monkeypatch.setattr(aer_router.time, "sleep", lambda _: None)

    daemon = aer_router.AERRoutingDaemon()
    daemon.start(build=False)

    assert built["called"] is False
    assert spawned == [["./aer_router"]]


def test_aer_router_stop_terminates_process() -> None:
    daemon = aer_router.AERRoutingDaemon()
    process = _Process()
    daemon._process = process

    daemon.stop()

    assert process.terminated is True
    assert process.wait_timeout == 2.0
    assert daemon._process is None


def test_aer_router_stop_without_process_is_noop() -> None:
    daemon = aer_router.AERRoutingDaemon()

    daemon.stop()

    assert daemon._process is None
