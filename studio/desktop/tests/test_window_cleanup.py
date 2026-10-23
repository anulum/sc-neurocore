# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native window cleanup failure and retry

"""Retain backend custody through a real kernel observation failure and native retry."""

from __future__ import annotations

import errno
import fcntl
import os
import select
import signal
import threading
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QThread
from PySide6.QtWebEngineCore import QWebEngineProfile

from sc_neurocore.studio.desktop.runtime import DesktopBackend
from studio.desktop.tests.test_window import (
    _close_button,
    _health,
    _interface_loaded,
    _message,
    _shown,
    _wait,
)
from tests.studio_seccomp_support import Refusal, install_refusals
from tests.test_studio_jobs_guard_cleanup import _children, _runs

pytestmark = pytest.mark.usefixtures("application")


def test_failed_cleanup_retains_the_live_backend_until_native_close_retry(
    desktop_backend: DesktopBackend, caplog: pytest.LogCaptureFixture
) -> None:
    """Report unreadable process membership and retry without losing the backend.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Real child server and private durable profile owned by this native window.
    caplog : pytest.LogCaptureFixture
        Actual cleanup-thread exception, independent of the visible retry message.
    """
    with _shown(desktop_backend) as window:
        _wait(lambda: _interface_loaded(window))
        instance = window.instance
        assert instance is not None
        _health(desktop_backend, instance.origin)
        config = desktop_backend.paths.config / "desktop.json"
        original = config.read_bytes()
        filtered_threads: list[int] = []
        native_start = QThread.start

        def start_restricted(
            thread: QThread, priority: QThread.Priority = QThread.Priority.InheritPriority
        ) -> None:
            """Retain Qt dispatch while applying the fault before the real worker runs."""
            actual_run = thread.run

            def run_restricted() -> None:
                """Refuse directory reads only in this native thread's actual callback."""
                flags = os.O_RDONLY | os.O_NONBLOCK | os.O_DIRECTORY | os.O_CLOEXEC
                install_refusals([Refusal("openat", errno.EMFILE, flags, argument_index=2)])
                filtered_threads.append(threading.get_native_id())
                actual_run()

            native.setattr(thread, "run", run_restricted)
            native_start(thread, priority)

        pidfd = os.pidfd_open(instance.pid)
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGSTOP)
            status = Path(f"/proc/{instance.pid}/status")
            _wait(lambda: "\nState:\tT" in status.read_text())
            with pytest.MonkeyPatch.context() as native:
                native.setattr(QThread, "start", start_restricted)
                window.close()
                _wait(
                    lambda: _message(window, "Studio could not confirm that its backend stopped.")
                )
                _wait(
                    lambda: all(not thread.isRunning() for thread in window.findChildren(QThread))
                )
            assert len(filtered_threads) == 1
            assert filtered_threads[0] != threading.get_native_id()
            assert os.listdir("/proc")
            assert desktop_backend.returncode is None and window.isVisible()
            assert config.read_bytes() == original
            assert any(
                record.exc_info
                and isinstance(record.exc_info[1], OSError)
                and record.exc_info[1].errno == errno.EMFILE
                and record.exc_info[1].filename == "/proc"
                for record in caplog.records
            )
            with (
                (desktop_backend.paths.state / "backend.lock").open("a+b") as lock,
                pytest.raises(BlockingIOError),
            ):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGCONT)
            finally:
                os.close(pidfd)
        _wait(lambda: desktop_backend.returncode is not None)
        _close_button(window)
        assert desktop_backend.returncode == 0
        assert config.read_bytes() == original
        with (desktop_backend.paths.state / "backend.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_queued_renderer_loss_does_not_replace_the_native_close_state(
    desktop_backend: DesktopBackend,
) -> None:
    """Finish an actual close when renderer loss is awaiting main-thread delivery.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Real server whose window owns a separate native renderer process.
    """
    with _shown(desktop_backend) as window:
        _wait(lambda: _interface_loaded(window))
        instance = window.instance
        assert instance is not None and window.view is not None
        config = desktop_backend.paths.config / "desktop.json"
        original = config.read_bytes()
        page = window.view.page()
        renderer = page.renderProcessPid()
        assert renderer > 0 and renderer not in (os.getpid(), instance.pid)
        assert b"QtWebEngineProcess" in Path(f"/proc/{renderer}/cmdline").read_bytes()
        losses: list[int] = []
        disposed: list[bool] = []
        page.renderProcessTerminated.connect(lambda _status, code: losses.append(code))
        page.destroyed.connect(lambda: disposed.append(True))

        def close_before_renderer_notification() -> bool:
            """Stop the renderer and request native close within one real Qt callback."""
            pidfd = os.pidfd_open(renderer)
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
                ready, _, _ = select.select([pidfd], [], [], 5)
                assert ready == [pidfd]
            finally:
                os.close(pidfd)
            window.close()
            _wait(lambda: bool(losses))
            assert not _message(window, "The Studio interface stopped.")
            return True

        _wait(close_before_renderer_notification)
        _wait(lambda: not window.isVisible())
        assert disposed and desktop_backend.returncode == 0
        assert config.read_bytes() == original


def test_backend_cleanup_waits_for_actual_browser_profile_disposal(
    desktop_backend: DesktopBackend,
) -> None:
    """Keep the window until queued profile deletion follows real group collection.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Fresh real backend with its actual Qt renderer and persistent profile.
    """
    with _shown(desktop_backend) as window:
        _wait(lambda: _interface_loaded(window))
        config = desktop_backend.paths.config / "desktop.json"
        original = config.read_bytes()
        (startup,) = window.findChildren(QThread)
        window.close()
        (cleanup,) = [thread for thread in window.findChildren(QThread) if thread is not startup]
        assert cleanup.wait(30_000)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.MetaCall)
        assert desktop_backend.returncode == 0
        assert window.isVisible() and window.findChildren(QWebEngineProfile)
        _wait(lambda: not window.isVisible())
        assert config.read_bytes() == original


def test_close_during_active_admission_waits_for_the_startup_owner(
    desktop_backend: DesktopBackend,
) -> None:
    """Cancel actual admission while its stopped child still requires collection.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Fresh owner whose real child is observed through the parent's task list.
    """
    with _shown(desktop_backend) as window:
        (startup,) = window.findChildren(QThread)
        children: list[int] = []
        deadline = time.monotonic() + 30
        while not children:
            children = [
                pid
                for pid in _children(os.getpid())
                if _runs(pid, "sc_neurocore.studio.desktop.backend")
            ]
            assert time.monotonic() < deadline, "No real admission child was observed."
            time.sleep(0.005)
        (pid,) = children
        pidfd = os.pidfd_open(pid)
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGSTOP)
            status = Path(f"/proc/{pid}/status")
            while "\nState:\tT" not in status.read_text():
                assert time.monotonic() < deadline, "The admission child did not stop."
                time.sleep(0.005)
            assert startup.isRunning() and window.instance is None
            window.close()
            assert window.isVisible() and startup.isRunning()
            assert window.view is None
        finally:
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGCONT)
            finally:
                os.close(pidfd)
        _wait(lambda: not window.isVisible())
        report = desktop_backend.close()
        assert report is not None and report.reaped
        with (desktop_backend.paths.state / "backend.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
