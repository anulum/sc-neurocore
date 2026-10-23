# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop startup refusal and real child custody

"""Exercise public admission cancellation and deadlines with actual stopped children."""

from __future__ import annotations

import fcntl
import os
import signal
import threading
import time
from contextlib import suppress
from pathlib import Path

import pytest

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.runtime import DesktopBackend
from tests.test_studio_desktop_runtime import _environment, _request
from tests.test_studio_jobs_guard_cleanup import _children, _runs


def test_pre_cancelled_start_does_not_spawn_or_retain_an_origin(tmp_path: Path) -> None:
    """Keep existing user bytes when the owner cancels before child admission.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Private desktop profile containing an existing user document.
    """
    env = _environment(tmp_path)
    document = tmp_path / "home" / "retained.txt"
    document.write_bytes(b"existing desktop work\n")
    backend = DesktopBackend(env)
    assert backend.returncode is None and backend.close() is None
    backend.cancel_startup()
    with pytest.raises(DesktopStartRefused, match="startup was cancelled"):
        backend.start()
    assert backend.returncode is None and backend.close() is None
    assert not (backend.paths.config / "desktop.json").exists()
    assert document.read_bytes() == b"existing desktop work\n"
    with (backend.paths.state / "backend.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize(
    "setting,value,message",
    [
        ("SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE", "production", "development profile"),
        ("SC_NEUROCORE_STUDIO_STORAGE_MODE", "external", "embedded local storage"),
    ],
)
def test_incompatible_profile_refusal_preserves_user_data(
    tmp_path: Path, setting: str, value: str, message: str
) -> None:
    """Refuse incompatible deployment settings before creating an owned child.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable user profile, independent from the live desktop application.
    setting, value : str
        Actual inherited configuration incompatible with the local launcher.
    message : str
        Public refusal explaining the required deployment or storage profile.
    """
    env = _environment(tmp_path)
    env[setting] = value
    retained = tmp_path / "home" / "retained.txt"
    retained.write_bytes(b"do not migrate this user profile\n")
    backend = DesktopBackend(env)
    with pytest.raises(DesktopStartRefused, match=message):
        backend.start()
    assert backend.returncode is None and backend.close() is None
    assert not (backend.paths.config / "desktop.json").exists()
    assert retained.read_bytes() == b"do not migrate this user profile\n"
    with (backend.paths.state / "backend.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("cancel", [False, True], ids=["deadline", "cancel"])
def test_interrupted_admission_collects_its_actual_child_and_allows_restart(
    tmp_path: Path, cancel: bool
) -> None:
    """Retain custody until a real stopped admission child can process termination.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Fresh persistent profile whose origin must remain unwritten on failure.
    cancel : bool
        Request public cancellation, or let the actual two-second deadline expire.
    """
    env = _environment(tmp_path)
    retained = tmp_path / "home" / "retained.txt"
    retained.write_bytes(b"retain this work through interrupted startup\n")
    backend = DesktopBackend(env)
    outcomes: list[BaseException] = []

    def start() -> None:
        """Run normal admission and retain its actual terminal result."""
        try:
            backend.start(timeout_seconds=30.0 if cancel else 2.0)
        except BaseException as error:
            outcomes.append(error)

    worker = threading.Thread(target=start, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 30
        children: list[int] = []
        while not children:
            children = [
                pid
                for pid in _children(os.getpid())
                if _runs(pid, "sc_neurocore.studio.desktop.backend")
            ]
            assert worker.is_alive() and time.monotonic() < deadline
            time.sleep(0.005)
        (pid,) = children
        with os.fdopen(os.pidfd_open(pid), "rb", buffering=0) as handle:
            try:
                signal.pidfd_send_signal(handle.fileno(), signal.SIGSTOP)
                stopped = os.waitid(os.P_PIDFD, handle.fileno(), os.WSTOPPED | os.WNOWAIT)
                assert stopped is not None and stopped.si_code == os.CLD_STOPPED
                assert stopped.si_status == signal.SIGSTOP
                status = Path(f"/proc/{pid}/status")
                assert not outcomes and backend.returncode is None
                if cancel:
                    backend.cancel_startup()
                while True:
                    pending = [
                        int(line.split()[1], 16)
                        for line in status.read_text().splitlines()
                        if line.startswith(("SigPnd:", "ShdPnd:"))
                    ]
                    if any(mask & (1 << (signal.SIGTERM - 1)) for mask in pending):
                        break
                    assert worker.is_alive() and time.monotonic() < deadline
                    time.sleep(0.005)
                assert worker.is_alive() and backend.returncode is None
                assert not (backend.paths.config / "desktop.json").exists()
                with (
                    (backend.paths.state / "backend.lock").open("a+b") as lock,
                    pytest.raises(BlockingIOError),
                ):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                with suppress(ProcessLookupError):
                    signal.pidfd_send_signal(handle.fileno(), signal.SIGCONT)
    finally:
        backend.cancel_startup()
        worker.join(timeout=30)
        assert not worker.is_alive()
        report = backend.close()
        assert report is not None and report.reaped
    (outcome,) = outcomes
    assert isinstance(outcome, DesktopStartRefused), outcome
    assert ("startup was cancelled" if cancel else "did not become ready") in str(outcome)
    assert backend.returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)
    assert not (backend.paths.config / "desktop.json").exists()
    assert retained.read_bytes() == b"retain this work through interrupted startup\n"
    with (backend.paths.state / "backend.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resumed = DesktopBackend(env)
    try:
        instance = resumed.start()
        assert instance.pid != pid
        assert _request(instance.origin, "/api/health") == {"status": "ok"}
        assert (resumed.paths.config / "desktop.json").exists()
        assert retained.read_bytes() == b"retain this work through interrupted startup\n"
    finally:
        recovered = resumed.close()
        assert recovered is not None and recovered.reaped and recovered.returncode == 0


def test_invalid_backend_settings_release_the_profile_for_a_corrected_restart(
    tmp_path: Path,
) -> None:
    """Collect a real child that exits during application configuration.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Private profile with an existing user document and invalid inherited limit.
    """
    env = _environment(tmp_path)
    env["SC_NEUROCORE_STUDIO_MAX_REQUEST_BODY_BYTES"] = "not-an-integer"
    document = tmp_path / "home" / "retained.txt"
    document.write_bytes(b"retain this document after application refusal\n")
    backend = DesktopBackend(env)
    with pytest.raises(DesktopStartRefused, match="The Studio backend"):
        backend.start()
    report = backend.close()
    assert report is not None and report.reaped and report.returncode != 0
    assert (
        "Studio request body limit must be an integer"
        in (backend.paths.state / "backend.log").read_text()
    )
    assert not (backend.paths.config / "desktop.json").exists()
    with (backend.paths.state / "backend.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env.pop("SC_NEUROCORE_STUDIO_MAX_REQUEST_BODY_BYTES")
    resumed = DesktopBackend(env)
    try:
        instance = resumed.start()
        assert _request(instance.origin, "/api/health") == {"status": "ok"}
        assert document.read_bytes() == b"retain this document after application refusal\n"
    finally:
        recovered = resumed.close()
        assert recovered is not None and recovered.reaped and recovered.returncode == 0


def test_configuration_write_failure_retains_the_previous_admitted_origin(
    tmp_path: Path,
) -> None:
    """Report a real read-only configuration directory after successful admission.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Persistent private profile whose existing configuration must survive refusal.
    """
    env = _environment(tmp_path)
    previous = DesktopBackend(env)
    try:
        first = previous.start()
        assert _request(first.origin, "/api/health") == {"status": "ok"}
        configuration = previous.paths.config / "desktop.json"
        original = configuration.read_bytes()
    finally:
        closed = previous.close()
        assert closed is not None and closed.reaped and closed.returncode == 0
    backend = DesktopBackend(env)
    outcomes: list[BaseException] = []

    def start() -> None:
        """Run actual admission through its final persistent-origin write."""
        try:
            backend.start()
        except BaseException as error:
            outcomes.append(error)

    worker = threading.Thread(target=start, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 30
        children: list[int] = []
        while not children:
            children = [
                pid
                for pid in _children(os.getpid())
                if _runs(pid, "sc_neurocore.studio.desktop.backend")
            ]
            assert worker.is_alive() and time.monotonic() < deadline
            time.sleep(0.005)
        (pid,) = children
        backend.paths.config.chmod(0o500)
        worker.join(timeout=40)
        assert not worker.is_alive()
        (outcome,) = outcomes
        assert isinstance(outcome, DesktopStartRefused), outcome
        assert "could not save its desktop configuration" in str(outcome)
        assert configuration.read_bytes() == original
        assert backend.returncode == 0
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        with (backend.paths.state / "backend.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        backend.paths.config.chmod(0o700)
        backend.cancel_startup()
        worker.join(timeout=30)
        assert not worker.is_alive()
        report = backend.close()
        assert report is not None and report.reaped
    resumed = DesktopBackend(env)
    try:
        second = resumed.start()
        assert second.origin == first.origin
        assert configuration.read_bytes() == original
        assert _request(second.origin, "/api/health") == {"status": "ok"}
    finally:
        recovered = resumed.close()
        assert recovered is not None and recovered.reaped and recovered.returncode == 0


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_deadline_does_not_create_a_profile_or_spawn_a_backend(
    tmp_path: Path, seconds: float
) -> None:
    """Refuse unusable readiness limits before touching persistent user state.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated user home and XDG directory bases.
    seconds : float
        Zero, negative or nonfinite requested startup deadline.
    """
    backend = DesktopBackend(_environment(tmp_path))
    document = tmp_path / "home" / "retained.txt"
    document.write_bytes(b"keep existing work before validating launch inputs\n")
    with pytest.raises(ValueError, match="finite and positive"):
        backend.start(timeout_seconds=seconds)
    assert backend.returncode is None and backend.close() is None
    assert not backend.paths.config.exists() and not backend.paths.state.exists()
    assert document.read_bytes() == b"keep existing work before validating launch inputs\n"
