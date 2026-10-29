# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native window failure and shutdown integration

"""Exercise actual Qt events, child loss and configuration refusal without substitutes."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QThread, Qt, QTimer, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QLabel, QPushButton

from sc_neurocore.studio.desktop.runtime import DesktopBackend
from sc_neurocore.studio.desktop.configuration import DesktopConfiguration, write_configuration
from sc_neurocore.studio.desktop.window import StudioWindow

pytestmark = pytest.mark.usefixtures("application")


def _wait(predicate: Callable[[], bool]) -> None:
    loop = QEventLoop()
    poll = QTimer(loop)
    poll.setSingleShot(True)
    deadline = QTimer(loop)
    deadline.setSingleShot(True)
    completed = False
    errors: list[BaseException] = []

    def check() -> None:
        nonlocal completed
        try:
            completed = predicate()
        except BaseException as error:
            errors.append(error)
        if completed or errors:
            loop.quit()
        else:
            poll.start(25)

    poll.timeout.connect(check)
    deadline.timeout.connect(loop.quit)
    poll.start(25)
    deadline.start(90_000)
    loop.exec()
    poll.stop()
    deadline.stop()
    if errors:
        raise errors[0]
    assert completed, "The native window did not reach its expected state."


@contextmanager
def _shown(backend: DesktopBackend) -> Iterator[StudioWindow]:
    window = StudioWindow(backend)
    window.show()
    try:
        yield window
    finally:
        window.close()
        _wait(lambda: not window.isVisible())
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QTest.qWait(25)


def _message(window: StudioWindow, text: str) -> bool:
    panel = window.centralWidget()
    assert panel is not None
    return any(text in label.text() for label in panel.findChildren(QLabel) if label.isVisible())


def _interface_loaded(window: StudioWindow) -> bool:
    if window.view is None:
        panel = window.centralWidget()
        assert panel is not None
        messages = [label.text() for label in panel.findChildren(QLabel) if label.isVisible()]
        assert messages == ["Starting Studio…"], "Studio startup failed: " + "; ".join(messages)
        return False
    values: list[object] = []
    window.view.page().runJavaScript(
        "document.readyState === 'complete' && "
        "!!document.querySelector('input[aria-label=\"Search models\"]')",
        values.append,
    )
    _wait(lambda: bool(values))
    return values[0] is True


def _close_button(window: StudioWindow) -> None:
    panel = window.centralWidget()
    assert panel is not None
    buttons = [button for button in panel.findChildren(QPushButton) if button.isVisible()]
    assert len(buttons) == 1
    assert buttons[0].text() == "Close"
    QTest.mouseClick(buttons[0], Qt.MouseButton.LeftButton)
    _wait(lambda: not window.isVisible())


def _health(backend: DesktopBackend, origin: str) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(origin + "/api/health", timeout=5) as response:
        assert json.loads(response.read(1024)) == {"status": "ok"}
    assert backend.returncode is None


def test_cached_previous_interface_cannot_replace_current_ui_or_erase_session(
    desktop_backend: DesktopBackend,
) -> None:
    """Load current Studio over a retained old HTTP cache at the same origin.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Actual application backend using the primed browser storage and port.
    """

    class PreviousInterface(BaseHTTPRequestHandler):
        """Serve the cacheable HTML left by a previous installed application."""

        def do_GET(self) -> None:
            """Return an old page whose explicit freshness outlives this server."""
            body = b'<html><body id="previous-interface">Previous Studio</body></html>'
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep fixture request logging out of the application's output.

            Parameters
            ----------
            format : str
                Standard HTTP server log template.
            *args : object
                Standard request details supplied by the HTTP server.
            """
            del format, args

    paths = desktop_backend.paths
    paths.prepare()
    server = ThreadingHTTPServer(("127.0.0.1", 0), PreviousInterface)
    port = server.server_port
    origin = f"http://127.0.0.1:{port}"
    worker = Thread(target=server.serve_forever)
    worker.start()
    legacy_cache = paths.cache / "webengine"
    profile = QWebEngineProfile("sc-neurocore-studio")
    profile.setPersistentStoragePath(str(paths.data / "webengine"))
    profile.setCachePath(str(legacy_cache))
    view = QWebEngineView()
    page = QWebEnginePage(profile, view)
    view.setPage(page)
    values: list[object] = []
    view.load(QUrl(origin + "/studios/sc-neurocore/"))
    view.show()
    try:

        def previous_loaded() -> bool:
            """Observe the actual cached predecessor through the native renderer."""
            values.clear()
            page.runJavaScript("document.body?.id === 'previous-interface'", values.append)
            _wait(lambda: bool(values))
            return values[0] is True

        _wait(previous_loaded)
        values.clear()
        page.runJavaScript(
            "localStorage.setItem('desktop-cache-session', 'retained');"
            "localStorage.getItem('desktop-cache-session')",
            values.append,
        )
        _wait(lambda: bool(values))
        assert values == ["retained"]
    finally:
        view.close()
        view.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        disposed: list[bool] = []
        profile.destroyed.connect(lambda: disposed.append(True))
        profile.deleteLater()
        _wait(lambda: bool(disposed))
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive()
    retained_cache = {p: p.read_bytes() for p in legacy_cache.rglob("*") if p.is_file()}
    assert retained_cache
    write_configuration(
        paths.config / "desktop.json", DesktopConfiguration(port, paths.data / "projects")
    )
    with _shown(desktop_backend) as window:
        _wait(lambda: _interface_loaded(window))
        assert window.instance is not None and window.instance.origin == origin
        assert window.view is not None
        values.clear()
        window.view.page().runJavaScript(
            "localStorage.getItem('desktop-cache-session')", values.append
        )
        _wait(lambda: bool(values))
        assert values == ["retained"]
        assert {p: p.read_bytes() for p in retained_cache} == retained_cache


def test_backend_loss_is_visible_and_close_releases_the_profile(
    desktop_backend: DesktopBackend,
) -> None:
    """Show actual server loss and let the user close the window safely.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Fresh owner launching the real installed backend module.
    """
    with _shown(desktop_backend) as window:
        _wait(lambda: _interface_loaded(window))
        instance = window.instance
        assert instance is not None
        _health(desktop_backend, instance.origin)
        config = desktop_backend.paths.config / "desktop.json"
        original = config.read_bytes()
        os.kill(instance.pid, signal.SIGTERM)
        _wait(lambda: _message(window, "The Studio backend stopped."))
        _close_button(window)
        assert desktop_backend.returncode == 0
        assert config.read_bytes() == original
        with (desktop_backend.paths.state / "backend.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_renderer_loss_keeps_the_backend_owned_until_window_close(
    desktop_backend: DesktopBackend,
) -> None:
    """Show real renderer failure while retaining the server until user closure.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Fresh isolated backend and browser storage namespace.
    """
    with _shown(desktop_backend) as window:
        _wait(lambda: _interface_loaded(window))
        assert window.view is not None
        instance = window.instance
        assert instance is not None
        renderer = window.view.page().renderProcessPid()
        assert renderer > 0 and renderer != os.getpid() and renderer != instance.pid
        assert b"QtWebEngineProcess" in Path(f"/proc/{renderer}/cmdline").read_bytes()
        os.kill(renderer, signal.SIGTERM)
        _wait(lambda: _message(window, "The Studio interface stopped."))
        _health(desktop_backend, instance.origin)
        _close_button(window)
        assert desktop_backend.returncode == 0


def test_future_configuration_is_visible_and_preserved_when_closed(
    desktop_backend: DesktopBackend,
) -> None:
    """Keep unsupported user data intact through an actual native startup refusal.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Fresh profile into which a future configuration is written.
    """
    desktop_backend.paths.prepare()
    config = desktop_backend.paths.config / "desktop.json"
    original = json.dumps(
        {"schema_version": "future", "port": 18031, "project_root": "/retained"}
    ).encode()
    config.write_bytes(original)
    with _shown(desktop_backend) as window:
        _wait(lambda: _message(window, "configuration version is not supported"))
        assert window.instance is None
        _close_button(window)
    assert config.read_bytes() == original


def test_closing_during_startup_waits_for_process_custody_to_finish(
    desktop_backend: DesktopBackend,
) -> None:
    """Cancel real asynchronous admission and close only after owned cleanup.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Fresh owner whose normal startup is cancelled by a native close event.
    """
    with _shown(desktop_backend) as window:
        window.close()
        _wait(lambda: not window.isVisible())
        report = desktop_backend.close()
        assert report is None or report.reaped, report
        lock_path = desktop_backend.paths.state / "backend.lock"
        if lock_path.exists():
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_missing_page_is_visible_and_the_backend_remains_owned_until_close(
    desktop_backend: DesktopBackend,
) -> None:
    """Report a real failed navigation while preserving backend ownership and config.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Fresh owner serving the actual interface and an ordinary missing API page.
    """
    with _shown(desktop_backend) as window:
        _wait(lambda: _interface_loaded(window))
        instance = window.instance
        assert instance is not None and window.view is not None
        _health(desktop_backend, instance.origin)
        config = desktop_backend.paths.config / "desktop.json"
        original = config.read_bytes()
        window.view.load(QUrl(instance.origin + "/api/no-such-studio-page"))
        _wait(lambda: _message(window, "The Studio interface could not load."))
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        _wait(lambda: not window.findChildren(QWebEngineProfile))
        _health(desktop_backend, instance.origin)
        assert config.read_bytes() == original
        _close_button(window)
        assert desktop_backend.returncode == 0
        assert config.read_bytes() == original
        with (desktop_backend.paths.state / "backend.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_cache_file_collision_reports_startup_failure_and_preserves_the_file(
    desktop_backend: DesktopBackend, caplog: pytest.LogCaptureFixture
) -> None:
    """Keep retained bytes intact when a cache path becomes a file before startup.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Fresh public owner with its actual prepared profile directories.
    caplog : pytest.LogCaptureFixture
        Actual worker-thread diagnostic, separate from the visible failure message.
    """
    desktop_backend.paths.prepare()
    cache = desktop_backend.paths.cache
    cache.rmdir()
    retained = b"retained file at the cache location"
    cache.write_bytes(retained)
    with _shown(desktop_backend) as window:
        _wait(lambda: _message(window, "Studio could not start. See the desktop log"))
        assert window.instance is None
        assert cache.read_bytes() == retained
        assert any(
            record.exc_info and isinstance(record.exc_info[1], FileExistsError)
            for record in caplog.records
        )
        _close_button(window)
    assert cache.read_bytes() == retained
    assert desktop_backend.close() is None


def test_close_before_queued_admission_never_mounts_the_browser(
    desktop_backend: DesktopBackend,
) -> None:
    """Close after actual admission finishes but before Qt delivers its queued signal.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Real owner whose startup runs while the main event queue is not processed.
    """
    with _shown(desktop_backend) as window:
        (startup,) = window.findChildren(QThread)
        config = desktop_backend.paths.config / "desktop.json"
        deadline = time.monotonic() + 30
        while startup.isRunning() or not config.is_file():
            assert time.monotonic() < deadline, "Actual backend admission did not finish."
            time.sleep(0.01)
        original = config.read_bytes()
        window.close()
        _wait(lambda: not window.isVisible())
        assert window.instance is not None and window.view is None
        assert desktop_backend.returncode == 0
        assert config.read_bytes() == original
