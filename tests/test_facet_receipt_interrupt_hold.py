# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — An interrupt during the validator's start leaves no orphan

"""An interrupt that lands while the validator starts is held until the process is owned.

``subprocess.Popen`` starts the child before it returns. An interrupt raised in
between left the recorder without the process it must stop, and the validator
ran on. These tests send a real SIGINT to this process from inside a ``Popen``
stand-in that has already started a real child, and require the capture to stop
and reap that child.
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from typing import Any

import pytest

import tools.facet_receipt_evidence as evidence

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX signals and process groups")

_SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


def _alive(pid: int) -> bool:
    """Return whether a process with this id still exists, a zombie included."""
    return Path(f"/proc/{pid}").exists()


def _interrupting_popen(started: list[int], fail: bool = False) -> Any:
    """Return a ``Popen`` stand-in that starts the child, then interrupts its caller.

    Parameters
    ----------
    started : list of int
        Receives the started child's process id.
    fail : bool
        Raise ``OSError`` after the interrupt, as a failed start would.

    Returns
    -------
    callable
        The stand-in.
    """
    real = subprocess.Popen

    def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        if fail:
            os.kill(os.getpid(), signal.SIGINT)
            raise OSError("the start failed")
        process: subprocess.Popen[bytes] = real(*args, **kwargs)
        started.append(process.pid)
        os.kill(os.getpid(), signal.SIGINT)
        return process

    return popen


def test_an_interrupt_during_the_start_stops_and_reaps_the_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The held interrupt is raised once the child is owned, and cleanup reaps it.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces ``subprocess.Popen``, which the evidence module calls.
    tmp_path : Path
        Working directory of the child.
    """
    started: list[int] = []
    monkeypatch.setattr(subprocess, "Popen", _interrupting_popen(started))
    before = signal.getsignal(signal.SIGINT)
    captured = evidence.capture_execution(
        _SLEEPER, repo_root=tmp_path, environment=dict(os.environ), timeout=30
    )
    assert isinstance(captured.interruption, KeyboardInterrupt)
    assert captured.exit_code == -signal.SIGKILL
    assert len(started) == 1 and not _alive(started[0])
    assert signal.getsignal(signal.SIGINT) is before


def test_an_interrupt_during_a_failed_start_reaches_the_caller(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A start that fails after an interrupt gives the interrupt, not the start error.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces ``subprocess.Popen``, which the evidence module calls.
    tmp_path : Path
        Working directory of the child.
    """
    monkeypatch.setattr(subprocess, "Popen", _interrupting_popen([], fail=True))
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        evidence.capture_execution(
            _SLEEPER, repo_root=tmp_path, environment=dict(os.environ), timeout=30
        )
    assert signal.getsignal(signal.SIGINT) is before


def test_a_failed_start_without_an_interrupt_is_a_start_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A start that fails on its own still names the start, and restores the handler.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces ``subprocess.Popen``, which the evidence module calls.
    tmp_path : Path
        Working directory of the child.
    """

    def refuse(*_args: Any, **_kwargs: Any) -> subprocess.Popen[bytes]:
        raise OSError("no such program")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(evidence.ExecutionStartError):
        evidence.capture_execution(
            _SLEEPER, repo_root=tmp_path, environment=dict(os.environ), timeout=30
        )
    assert signal.getsignal(signal.SIGINT) is before


def test_another_start_failure_passes_through_and_restores_the_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A start failure that is not an OSError reaches the caller unchanged.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces ``subprocess.Popen``, which the evidence module calls.
    tmp_path : Path
        Working directory of the child.
    """

    def refuse(*_args: Any, **_kwargs: Any) -> subprocess.Popen[bytes]:
        raise ValueError("bad arguments")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(ValueError, match="bad arguments"):
        evidence.capture_execution(
            _SLEEPER, repo_root=tmp_path, environment=dict(os.environ), timeout=30
        )
    assert signal.getsignal(signal.SIGINT) is before


def test_a_clean_run_restores_the_handler_and_holds_nothing(tmp_path: Path) -> None:
    """Without an interrupt the child runs to its end and SIGINT is the caller's again.

    Parameters
    ----------
    tmp_path : Path
        Working directory of the child.
    """
    before = signal.getsignal(signal.SIGINT)
    captured = evidence.capture_execution(
        [sys.executable, "-c", "pass"], repo_root=tmp_path, environment=dict(os.environ), timeout=30
    )
    assert captured.exit_code == 0 and captured.interruption is None
    assert signal.getsignal(signal.SIGINT) is before


def test_an_ignored_sigint_is_left_alone(tmp_path: Path) -> None:
    """A caller that ignores SIGINT keeps ignoring it; nothing is held or installed.

    Parameters
    ----------
    tmp_path : Path
        Working directory of the child.
    """
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        captured = evidence.capture_execution(
            [sys.executable, "-c", "pass"],
            repo_root=tmp_path,
            environment=dict(os.environ),
            timeout=30,
        )
        assert captured.exit_code == 0
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


def test_a_worker_thread_starts_the_child_without_touching_sigint(tmp_path: Path) -> None:
    """Off the main thread no handler may be installed, and none is.

    Parameters
    ----------
    tmp_path : Path
        Working directory of the child.
    """
    before = signal.getsignal(signal.SIGINT)
    results: list[evidence.CapturedExecution] = []
    worker = threading.Thread(
        target=lambda: results.append(
            evidence.capture_execution(
                [sys.executable, "-c", "pass"],
                repo_root=tmp_path,
                environment=dict(os.environ),
                timeout=30,
            )
        )
    )
    worker.start()
    worker.join(timeout=30)
    assert len(results) == 1 and results[0].exit_code == 0
    assert signal.getsignal(signal.SIGINT) is before
