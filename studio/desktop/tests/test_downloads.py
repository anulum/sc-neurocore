# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual desktop export and save-dialog integration

"""Keep real simulation exports and user cancellation under native window control."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QStandardPaths, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWebEngineCore import QWebEngineDownloadRequest
from PySide6.QtWidgets import QApplication

from sc_neurocore.studio.desktop.runtime import DesktopBackend
from sc_neurocore.studio.desktop.window import StudioWindow

pytestmark = pytest.mark.usefixtures("application", "window_manager")


def _wait(predicate: Callable[[], bool]) -> None:
    """Process actual Qt events until the requested observation or a bounded failure.

    Parameters
    ----------
    predicate : callable
        Read-only observation of the real window, download or filesystem.
    """
    loop = QEventLoop()
    poll = QTimer(loop)
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

    poll.timeout.connect(check)
    deadline.timeout.connect(loop.quit)
    poll.start(25)
    deadline.start(90_000)
    loop.exec()
    poll.stop()
    deadline.stop()
    if errors:
        raise errors[0]
    assert completed, "The native export did not reach its expected state."


def _script(window: StudioWindow, expression: str) -> object:
    """Run an observation or user action in the actual rendered application.

    Parameters
    ----------
    window : StudioWindow
        Shown application with its real WebEngine page.
    expression : str
        JavaScript observation or DOM action, without substituted API responses.

    Returns
    -------
    object
        The real renderer's callback value.
    """
    assert window.view is not None
    values: list[object] = []
    window.view.page().runJavaScript(expression, values.append)
    _wait(lambda: bool(values))
    return values[0]


@contextmanager
def _dialog_action(action: str, destination: Path) -> Iterator[subprocess.Popen[str]]:
    """Operate this process's save dialog from an independent real X controller.

    Parameters
    ----------
    action : str
        Save the selected file or cancel with Escape.
    destination : pathlib.Path
        Actual user-selected path and parent of its private control receipts.

    Yields
    ------
    subprocess.Popen
        Owned controller, reaped before the application fixture is disposed.
    """
    with (destination.parent / "native-dialog.log").open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).with_name("native_dialog_controller.py")),
                str(os.getpid()),
                action,
                str(destination),
                str(destination.parent / "native-dialog.json"),
            ],
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


@pytest.mark.parametrize("action", ["save", "cancel"])
def test_real_simulation_export_obeys_the_native_save_decision(
    desktop_backend: DesktopBackend, tmp_path: Path, action: str
) -> None:
    """Save or cancel two consecutive actual exports without changing retained data.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Actual application backend with private project and browser storage.
    tmp_path : pathlib.Path
        User-selected save location and retained research file.
    action : str
        Native Save or Cancel decision made through actual keyboard events.
    """
    assert not QApplication.testAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)
    window = StudioWindow(desktop_backend)
    window.show()
    try:
        _wait(lambda: window.view is not None)
        _wait(
            lambda: (
                _script(
                    window, "Boolean(document.querySelector('input[aria-label=\"Search models\"]'))"
                )
                is True
            )
        )
        instance = window.instance
        assert instance is not None
        _script(
            window,
            "(() => { const original = window.fetch.bind(window); window.__nativeRequests = []; window.fetch = async (...args) => {"
            " const request = {url: String(args[0]), pending: true}; window.__nativeRequests.push(request);"
            " let response; try { response = await original(...args); request.status = response.status; }"
            " finally { request.pending = false; }"
            " if (String(args[0]).includes('/simulate') && String(args[1]?.body).includes('9223372036854775807'))"
            " window.__nativeExportObservation = {status: response.status, request: args[1].body, response: await response.clone().text()};"
            " return response; }; Array.from(document.querySelectorAll('button')).find(button => button.innerText === 'Skip tour')?.click(); return true; })()",
        )
        workspace = {
            "name": "native-download-input",
            "state": {
                "sourceMode": "model",
                "selectedModelName": "SCStochasticRateAdaptationNeuron",
                "modelParams": {"tau_a": 37},
                "dt": 0.5,
                "duration": 20,
                "current": 10,
                "protocol": "constant",
                "frequencyHz": 37,
                "seed": 9223372036854775807,
                "trial": "replay",
            },
        }
        request = urllib.request.Request(
            instance.origin + "/api/project/save",
            data=json.dumps(workspace).encode(),
            headers={"Origin": instance.origin, "Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=5) as response:
            assert response.status == 200
        assert (
            _script(
                window,
                "(() => { document.querySelector('[aria-label=\"Refresh projects\"]').click(); return true; })()",
            )
            is True
        )
        selector = "Array.from(document.querySelectorAll('button')).find(button => button.getAttribute('aria-label')?.startsWith('Open project native-download-input revision'))"
        _wait(lambda: _script(window, "Boolean(" + selector + ")") is True)
        assert _script(window, "(() => { (" + selector + ").click(); return true; })()") is True
        _wait(lambda: _script(window, "Boolean(window.__nativeExportObservation)") is True)
        raw = _script(window, "JSON.stringify(window.__nativeExportObservation)")
        assert isinstance(raw, str)
        observation = json.loads(raw)
        assert observation["status"] == 200
        assert "9223372036854775807" in observation["request"]
        expected = json.loads(observation["response"])
        assert (
            expected["model_name"] == "SCStochasticRateAdaptationNeuron"
            and expected["n_steps"] == 40
        )
        assert window.view is not None
        downloads: list[dict[str, str]] = []

        def requested(download: QWebEngineDownloadRequest) -> None:
            record = {"url": download.url().toString(), "state": download.state().name}
            downloads.append(record)
            download.stateChanged.connect(lambda state: record.update(state=state.name))

        window.view.page().profile().downloadRequested.connect(requested)
        exports: list[dict[str, object]] = []
        for export_index in range(2):
            directory = tmp_path / f"export-{export_index}"
            directory.mkdir()
            destination = directory / "retained experiment % 63.json"
            original = b'{"retained":"user research data"}'
            if action == "cancel":
                destination.write_bytes(original)
            default = Path(
                QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
            )
            before = {str(path) for path in default.glob("*") if path.is_file()}
            with _dialog_action(action, destination) as controller:
                assert (
                    _script(
                        window,
                        "(() => { Array.from(document.querySelectorAll('button')).find(button => button.title === 'Download the current run as JSON').click(); return true; })()",
                    )
                    is True
                )

                def _download_finished(index: int = export_index) -> bool:
                    """Observe this export's actual terminal download state."""
                    return len(downloads) > index and downloads[index]["state"] in {
                        "DownloadCompleted",
                        "DownloadCancelled",
                    }

                def _controller_finished(child: subprocess.Popen[str] = controller) -> bool:
                    """Observe the real controller's bounded termination."""
                    return child.poll() is not None

                _wait(_download_finished)
                _wait(_controller_finished)
            assert controller.returncode == 0, (directory / "native-dialog.log").read_text()
            controlled = json.loads((directory / "native-dialog.json").read_text())
            assert controlled["owner_pid"] == os.getpid() and controlled["action"] == action
            assert controlled["window"] and len(downloads) == export_index + 1
            assert downloads[export_index]["url"].startswith("blob:" + instance.origin + "/")
            if action == "save":
                assert downloads[export_index]["state"] == "DownloadCompleted"
                assert json.loads(destination.read_text()) == expected
            else:
                assert downloads[export_index]["state"] == "DownloadCancelled"
                assert destination.read_bytes() == original
            assert {str(path) for path in default.glob("*") if path.is_file()} == before
            exports.append(
                {
                    "download": downloads[export_index].copy(),
                    "dialog": controlled,
                    "export_or_retained_file_sha256": hashlib.sha256(
                        destination.read_bytes()
                    ).hexdigest(),
                }
            )
        assert desktop_backend.returncode is None
        (tmp_path / "native-requests-at-close.json").write_text(
            str(_script(window, "JSON.stringify(window.__nativeRequests)")) + "\n"
        )
    finally:
        window.close()
        _wait(lambda: not window.isVisible())
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QTest.qWait(25)
    assert desktop_backend.returncode == 0
    (tmp_path / "native-export-observation.json").write_text(
        json.dumps(
            {
                "action": action,
                "origin": instance.origin,
                "ui_sha256": instance.ui_sha256,
                "model": expected["model_name"],
                "n_steps": expected["n_steps"],
                "request": observation["request"],
                "actual_response_sha256": hashlib.sha256(
                    observation["response"].encode()
                ).hexdigest(),
                "exports": exports,
                "backend_exit": desktop_backend.returncode,
                "scope": "Current-source native UI export/save/cancel; no installed-release or scientific/full-state acceptance.",
            },
            indent=2,
        )
        + "\n"
    )
