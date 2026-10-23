# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Studio window and backend lifetime

"""Show the actual Studio interface while keeping process control off the UI thread."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QStandardPaths, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QCloseEvent, QColor, QKeySequence
from PySide6.QtWebEngineCore import QWebEngineDownloadRequest, QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.desktop.runtime import DesktopBackend, DesktopInstance
from sc_neurocore.studio.desktop.webview import DesktopPage, DesktopRequestPolicy

_LOGGER = logging.getLogger(__name__)


class _StartThread(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, backend: DesktopBackend, parent: QWidget) -> None:
        super().__init__(parent)
        self.backend = backend

    def run(self) -> None:
        """Admit the owned backend and report startup without blocking Qt events."""
        try:
            instance = self.backend.start()
        except AuthoredRefusal as exc:
            self.failed.emit(str(exc))
        except Exception:
            _LOGGER.exception("Desktop backend startup failed")
            self.failed.emit("Studio could not start. See the desktop log for details.")
        else:
            self.ready.emit(instance)


class _CloseThread(QThread):
    failed = Signal(str)

    def __init__(self, backend: DesktopBackend, parent: QWidget) -> None:
        super().__init__(parent)
        self.backend = backend
        self.clean = False

    def run(self) -> None:
        """Collect the backend's real group cleanup outcome off the Qt thread."""
        try:
            report = self.backend.close()
            self.clean = report is None or report.reaped
        except Exception:
            _LOGGER.exception("Desktop backend cleanup failed")
        if not self.clean:
            self.failed.emit("Studio could not confirm that its backend stopped. Close to retry.")


class StudioWindow(QMainWindow):
    """Own one persistent browser profile and the backend serving its interface."""

    def __init__(self, backend: DesktopBackend) -> None:
        """Show startup immediately and begin actual backend admission asynchronously.

        Parameters
        ----------
        backend : DesktopBackend
            A fresh owner for this user's desktop profile and child processes.
        """
        super().__init__()
        self.backend = backend
        self.instance: DesktopInstance | None = None
        self.view: QWebEngineView | None = None
        self._profile: QWebEngineProfile | None = None
        self._request_policy: DesktopRequestPolicy | None = None
        self._closing = False
        self._can_close = False
        self._close_thread: _CloseThread | None = None
        self.setWindowTitle("SNN Studio")
        self.resize(1440, 960)
        self._message("Starting Studio…")
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        self.menuBar().addMenu("File").addAction(quit_action)
        self._monitor = QTimer(self)
        self._monitor.setInterval(250)
        self._monitor.timeout.connect(self._check_backend)
        self._startup = _StartThread(backend, self)
        self._startup.ready.connect(self._ready)
        self._startup.failed.connect(self._startup_failed)
        self._startup.finished.connect(self._startup_finished)
        self._startup.start()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Keep Qt alive until startup and owned backend cleanup have finished.

        Parameters
        ----------
        event : PySide6.QtGui.QCloseEvent
            Actual window-close event; delayed while process custody remains.
        """
        if self._can_close:
            super().closeEvent(event)
            return
        event.ignore()
        self._closing = True
        self._monitor.stop()
        self.backend.cancel_startup()
        self._message("Closing Studio…")
        if not self._startup.isRunning():
            self._begin_close()

    def _message(self, text: str) -> None:
        """Display a real startup or process failure state with a close action."""
        if self.view is not None and self._profile is not None:
            self.view.destroyed.connect(self._profile.deleteLater)
            self._profile.destroyed.connect(self._browser_disposed)
            self.view = None
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        label = QLabel(text, panel)
        label.setWordWrap(True)
        layout.addWidget(label)
        button = QPushButton("Close", panel)
        button.clicked.connect(self.close)
        layout.addWidget(button)
        self.setCentralWidget(panel)

    def _ready(self, instance: DesktopInstance) -> None:
        """Mount the admitted real UI with persistent, profile-specific browser storage."""
        self.instance = instance
        if self._closing:
            return
        profile = QWebEngineProfile("sc-neurocore-studio", self)
        profile.setPersistentStoragePath(str(self.backend.paths.data / "webengine"))
        profile.setCachePath(str(self.backend.paths.cache / "webengine"))
        self._request_policy = DesktopRequestPolicy(instance.origin, profile)
        profile.setUrlRequestInterceptor(self._request_policy)
        profile.downloadRequested.connect(self._download)
        view = QWebEngineView(self)
        page = DesktopPage(profile, instance.origin, view)
        page.setBackgroundColor(QColor("#0d1117"))
        page.renderProcessTerminated.connect(self._renderer_stopped)
        view.setPage(page)
        view.loadFinished.connect(self._loaded)
        self._profile = profile
        self.view = view
        self.setCentralWidget(view)
        view.load(QUrl(instance.origin + "/studios/sc-neurocore/"))
        self._monitor.start()

    def _startup_failed(self, message: str) -> None:
        """Retain an actionable failure while the start thread completes its cleanup."""
        if not self._closing:
            self._message(message)

    def _startup_finished(self) -> None:
        """Complete a close requested while admission was still running."""
        if self._closing:
            self._begin_close()

    def _begin_close(self) -> None:
        """Start at most one active cleanup thread for this backend owner."""
        if self._close_thread is not None and self._close_thread.isRunning():
            return
        thread = _CloseThread(self.backend, self)
        thread.failed.connect(self._message)
        thread.finished.connect(self._closed)
        self._close_thread = thread
        thread.start()

    def _closed(self) -> None:
        """Accept window close only after verified backend group cleanup."""
        if self._close_thread is not None and self._close_thread.clean:
            if self._profile is None:
                self._can_close = True
                self.close()

    def _browser_disposed(self) -> None:
        """Let every page finish destruction before its profile and window disappear."""
        self._profile = None
        self._request_policy = None
        if self._closing:
            self._closed()

    def _check_backend(self) -> None:
        """Surface actual backend loss instead of leaving an apparently live interface."""
        if self.backend.returncode is not None:
            self._monitor.stop()
            self._message("The Studio backend stopped. Close and relaunch Studio to recover.")

    def _loaded(self, successful: bool) -> None:
        """Surface a failed real page load without replacing it with a substitute page."""
        if not successful and not self._closing:
            self._message("The Studio interface could not load. Close and relaunch Studio.")

    def _renderer_stopped(
        self, _status: QWebEnginePage.RenderProcessTerminationStatus, _exit_code: int
    ) -> None:
        """Retain a visible renderer failure until the owner closes and reaps the backend."""
        if not self._closing:
            self._message("The Studio interface stopped. Close and relaunch Studio to recover.")

    def _download(self, request: QWebEngineDownloadRequest) -> None:
        """Save actual browser exports only after the user selects their destination."""
        if self.instance is None or not (
            request.url().toString().startswith(self.instance.origin + "/")
            or request.url().toString().startswith("blob:" + self.instance.origin + "/")
        ):
            request.cancel()
            return
        directory = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DownloadLocation
        )
        proposed = str(Path(directory) / Path(request.downloadFileName()).name)
        selected, _filter = QFileDialog.getSaveFileName(self, "Save Studio export", proposed)
        if not selected:
            request.cancel()
            return
        target = Path(selected)
        request.setDownloadDirectory(str(target.parent))
        request.setDownloadFileName(target.name)
        request.accept()
