# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the launcher service stops when asked and never outlives its test

"""Bind how a test-started launcher service stops and where its sockets live.

The service must stop on SIGTERM even when the process that started it handed
down a mask blocking that signal; a service that does not stop is killed and the
failure reported; a service ends when the thread that started it ends; and its
sockets fit the Unix path limit under any temporary folder.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from tests import studio_storage_launcher_support as support
from tests.studio_storage_launcher_support import (
    BASE_NAME,
    LONGEST_SOCKET_SUFFIX,
    UNIX_SOCKET_PATH_LIMIT,
    Launcher,
    launcher_base,
    short_root,
    shutdown,
    start,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the launcher runs on Linux only")

_ROOT = Path(__file__).resolve().parents[1]


def _blocked_signals(pid: int) -> int:
    """Return the blocked-signal mask of a process, read from ``/proc``."""
    for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
        if line.startswith("SigBlk:"):
            return int(line.split()[1], 16)
    raise AssertionError(f"no SigBlk line for process {pid}")


def _gone(pid: int, seconds: float) -> bool:
    """Return whether a process has ended (or is a zombie) within a time."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            state = (
                Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()[0]
            )
        except FileNotFoundError:
            return True
        if state == "Z":
            return True
        time.sleep(0.05)
    return False


def test_the_service_stops_on_sigterm_although_its_starter_blocked_it() -> None:
    """A mask blocking SIGTERM and SIGINT, handed down at start, does not keep the service running."""
    handled = {signal.SIGTERM, signal.SIGINT}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
    try:
        with launcher_base() as base:
            launcher = start(base)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            blocked = _blocked_signals(launcher.process.pid)
            shutdown(launcher, timeout=10.0)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    assert blocked & ((1 << (signal.SIGTERM - 1)) | (1 << (signal.SIGINT - 1))) == 0
    assert launcher.process.returncode == 0


def test_a_launcher_that_ignores_sigterm_is_killed_and_reported() -> None:
    """The stop does not wait forever and does not leave the process running."""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(600)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stdout.readline() == b"ready\n"
    stubborn = Launcher(process, Path("unused.sock"), Path("unused-spool"))
    with pytest.raises(AssertionError, match="did not stop on SIGTERM within 0.5 s and was killed"):
        shutdown(stubborn, timeout=0.5)
    assert process.returncode == -signal.SIGKILL
    assert process.stdout.closed and process.stderr is not None and process.stderr.closed


def test_a_started_launcher_ends_when_its_starter_dies() -> None:
    """A test process killed outright takes its launcher with it."""
    script = textwrap.dedent(
        """
        import sys, time
        from tests.studio_storage_launcher_support import launcher_base, start
        with launcher_base() as base:
            launcher = start(base)
            print(launcher.process.pid, base, flush=True)
            time.sleep(600)
        """
    )
    starter = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(_ROOT), str(_ROOT / "src")])},
        stdout=subprocess.PIPE,
    )
    assert starter.stdout is not None
    pid, base = starter.stdout.readline().decode().split(maxsplit=1)
    service = int(pid)
    try:
        assert not _gone(service, 0.2)
        starter.kill()
        starter.wait(timeout=30.0)
        assert _gone(service, 10.0), f"launcher {service} outlived the process that started it"
    finally:
        # The killed starter could not remove its base folder.
        shutil.rmtree(base.strip(), ignore_errors=True)
        starter.stdout.close()


def test_the_base_lies_under_the_temporary_folder_when_it_is_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A short temporary folder is used as it is."""
    short = Path(tempfile.mkdtemp(prefix="t", dir=short_root()))
    try:
        monkeypatch.setattr(tempfile, "tempdir", str(short))
        assert short_root() == short
        with launcher_base() as base:
            assert base.parent == short
            assert len(os.fsencode(str(base) + LONGEST_SOCKET_SUFFIX)) < UNIX_SOCKET_PATH_LIMIT
    finally:
        short.rmdir()


def test_a_long_temporary_folder_gives_way_to_the_runtime_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Under a temporary folder too long for the sockets, the user's runtime folder is used."""
    long_folder = tmp_path / ("x" * 60)
    long_folder.mkdir()
    runtime = Path(tempfile.mkdtemp(prefix="r", dir=short_root()))
    try:
        monkeypatch.setattr(tempfile, "tempdir", str(long_folder))
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
        assert short_root() == runtime
        with launcher_base() as base:
            assert base.parent == runtime
            assert len(os.fsencode(str(base) + LONGEST_SOCKET_SUFFIX)) < UNIX_SOCKET_PATH_LIMIT
    finally:
        runtime.rmdir()


def test_no_short_folder_is_refused_with_the_length_that_fits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a folder short enough, the run is told how short TMPDIR must be."""
    long_folder = tmp_path / ("x" * 60)
    long_folder.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(long_folder))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(long_folder))
    room = UNIX_SOCKET_PATH_LIMIT - 1 - len(BASE_NAME) - len(LONGEST_SOCKET_SUFFIX)
    with pytest.raises(AssertionError, match=f"no temporary folder of at most {room} bytes"):
        short_root()


def test_the_longest_socket_path_is_the_grant_endpoint_of_a_generation() -> None:
    """The reserve counts the spool, a job, a generation and the grant endpoint's name."""
    assert LONGEST_SOCKET_SUFFIX == "/spool/sj_" + "0" * 16 + "/" + "0" * 32 + "/grant.sock"
    assert len(BASE_NAME) == len("/scl") + 8


def _exit_code(call: Callable[[], None], monkeypatch: pytest.MonkeyPatch) -> int | None:
    """Run a child-side step with ``os._exit`` raising, and return the code it exits with."""

    def raise_exit(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", raise_exit)
    try:
        call()
    except SystemExit as stopped:
        return int(stopped.code) if isinstance(stopped.code, int) else -1
    return None


def test_the_child_asks_to_be_killed_with_its_starter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child requests SIGKILL on the starter's end and continues while the starter lives."""
    calls: list[tuple[int, int]] = []

    def prctl(option: int, value: int) -> int:
        calls.append((option, value))
        return 0

    monkeypatch.setattr(support, "_PRCTL", prctl)
    assert _exit_code(lambda: support._end_with_starter(os.getppid()), monkeypatch) is None
    assert calls == [(1, int(signal.SIGKILL))]


def test_a_child_whose_starter_already_ended_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A starter gone before the request leaves nobody to stop the child, so it exits."""
    monkeypatch.setattr(support, "_PRCTL", lambda option, value: 0)
    assert _exit_code(lambda: support._end_with_starter(os.getppid() + 1), monkeypatch) == 70


def test_a_child_that_cannot_ask_to_end_with_its_starter_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused request is not ignored: the child exits instead of running unguarded."""
    monkeypatch.setattr(support, "_PRCTL", lambda option, value: -1)
    assert _exit_code(lambda: support._end_with_starter(os.getppid()), monkeypatch) == 70


def test_without_prctl_the_child_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off Linux there is no request to make; the stop on SIGTERM remains."""
    monkeypatch.setattr(support, "_PRCTL", None)
    assert _exit_code(lambda: support._end_with_starter(1), monkeypatch) is None
