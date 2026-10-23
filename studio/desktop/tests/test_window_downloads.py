# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual window download admission

"""Keep unsupported real browser downloads away from retained user files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PySide6.QtCore import QStandardPaths, QTimer, QUrl
from PySide6.QtWebEngineCore import QWebEngineDownloadRequest
from PySide6.QtWidgets import QApplication, QFileDialog

from sc_neurocore.studio.desktop.runtime import DesktopBackend
from studio.desktop.tests.test_window import _health, _interface_loaded, _shown, _wait

pytestmark = pytest.mark.usefixtures("application")


def test_inline_download_is_cancelled_without_a_save_dialog_or_file_change(
    desktop_backend: DesktopBackend, tmp_path: Path
) -> None:
    """Cancel an actual unsupported inline download while retaining backend custody.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Actual process owner with a private profile and research storage.
    tmp_path : pathlib.Path
        Durable observation of the real download and retained-file outcome.
    """
    with _shown(desktop_backend) as window:
        _wait(lambda: _interface_loaded(window))
        instance = window.instance
        assert instance is not None and window.view is not None
        directory = Path(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        )
        directory.mkdir(parents=True, exist_ok=True)
        retained = directory / "inline.txt"
        retained.write_bytes(b"retained experiment data")
        before_names = set(directory.iterdir())
        before = {str(path): path.read_bytes() for path in directory.glob("*") if path.is_file()}
        requests: list[dict[str, str]] = []
        unexpected_dialogs: list[str] = []

        def observe(request: QWebEngineDownloadRequest) -> None:
            """Retain the native download's actual state transitions."""
            row = {"url": request.url().toString(), "state": request.state().name}
            requests.append(row)
            request.stateChanged.connect(lambda state: row.update(state=state.name))

        def cancel_unexpected_dialog() -> None:
            """Recover an unexpected real dialog so a regression can fail cleanly."""
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, QFileDialog):
                    unexpected_dialogs.append(widget.windowTitle())
                    widget.reject()

        timer = QTimer(window)
        timer.timeout.connect(cancel_unexpected_dialog)
        timer.start(25)
        window.view.page().profile().downloadRequested.connect(observe)
        try:
            window.view.page().download(
                QUrl("data:text/plain,unsupported-inline-export"), "inline.txt"
            )
            _wait(lambda: bool(requests) and requests[0]["state"] == "DownloadCancelled")
            _wait(lambda: set(directory.iterdir()) == before_names)
        finally:
            timer.stop()
        assert not unexpected_dialogs
        assert len(requests) == 1 and requests[0]["url"].startswith("data:")
        assert {
            str(path): path.read_bytes() for path in directory.glob("*") if path.is_file()
        } == before
        _health(desktop_backend, instance.origin)
    assert desktop_backend.returncode == 0
    assert set(directory.iterdir()) == before_names
    assert {
        str(path): path.read_bytes() for path in directory.glob("*") if path.is_file()
    } == before
    (tmp_path / "unsupported-download.json").write_text(
        json.dumps(
            {
                "requests": requests,
                "unexpected_dialogs": unexpected_dialogs,
                "retained_sha256": hashlib.sha256(retained.read_bytes()).hexdigest(),
                "backend_exit": desktop_backend.returncode,
            },
            indent=2,
        )
        + "\n"
    )
