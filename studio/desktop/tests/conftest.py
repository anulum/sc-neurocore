# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native desktop integration environment

"""Provide real Qt and isolated backend profiles for the Linux desktop lane."""

from __future__ import annotations

import os
import subprocess
import time
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from sc_neurocore.studio.desktop.runtime import DesktopBackend

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def application(tmp_path_factory: pytest.TempPathFactory) -> Iterator[QApplication]:
    """Retain Qt until every actual window and renderer has been disposed.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Per-run private namespace keeping Qt away from the user's desktop bus.

    Yields
    ------
    PySide6.QtWidgets.QApplication
        Native application whose events the integration tests process.
    """
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    private = tmp_path_factory.mktemp("qt-desktop-runtime")
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=" + str(private / "bus"))
        environment.setenv("XDG_CURRENT_DESKTOP", "X-Generic")
        app = QApplication([])
        app.setQuitOnLastWindowClosed(False)
        bootstrap = threading.gettrace()
        if bootstrap is not None:
            start_thread = QThread.start

            def start_measured(
                thread: QThread, priority: QThread.Priority = QThread.Priority.InheritPriority
            ) -> None:
                """Attach tracing while preserving the real native start and priority."""
                actual_run = thread.run

                def run_measured() -> None:
                    """Install the thread bootstrap before the unchanged worker callback."""
                    sys.settrace(bootstrap)
                    actual_run()

                environment.setattr(thread, "run", run_measured)
                start_thread(thread, priority)

            environment.setattr(QThread, "start", start_measured)
        yield app
        app.quit()


@pytest.fixture
def desktop_backend(tmp_path: Path) -> Iterator[DesktopBackend]:
    """Keep each real child server's durable data outside the installation.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable parent of the backend's home and XDG directories.

    Yields
    ------
    DesktopBackend
        Fresh process owner, closed after the window's own cleanup finishes.
    """
    home = tmp_path / "home"
    temporary = tmp_path / "tmp"
    home.mkdir()
    temporary.mkdir()
    environ = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SC_NEUROCORE_STUDIO_")
    }
    environ.update(
        HOME=str(home),
        TMPDIR=str(temporary),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        XDG_STATE_HOME=str(tmp_path / "state"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    backend = DesktopBackend(environ)
    try:
        yield backend
    finally:
        backend.cancel_startup()
        report = backend.close()
        assert report is None or report.reaped, report


@pytest.fixture
def window_manager(tmp_path: Path) -> Iterator[None]:
    """Run the existing real Openbox input on this test's private X display.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Destination of the private manager log.

    Yields
    ------
    None
        The actual window manager is ready for native focus and keyboard actions.
    """
    environment = dict(os.environ)
    for key in ("SESSION_MANAGER", "DBUS_SESSION_BUS_ADDRESS", "DESKTOP_AUTOSTART_ID"):
        environment.pop(key, None)
    command = ["openbox", "--sm-disable"]
    root = os.environ.get("SC_NEUROCORE_DESKTOP_TEST_WM_ROOT")
    if root is not None:
        prefix = Path(root)
        environment["LD_LIBRARY_PATH"] = str(prefix / "usr/lib/x86_64-linux-gnu")
        environment["XDG_DATA_DIRS"] = str(prefix / "usr/share") + ":/usr/share"
        command = [
            str(prefix / "usr/bin/openbox"),
            "--sm-disable",
            "--config-file",
            str(prefix / "etc/xdg/openbox/rc.xml"),
        ]
    with (tmp_path / "window-manager.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 8
            while True:
                assert process.poll() is None, "The private window manager exited."
                probe = subprocess.run(["wmctrl", "-m"], capture_output=True, text=True, timeout=3)
                if probe.returncode == 0 and "Openbox" in probe.stdout:
                    break
                assert time.monotonic() < deadline, "The private window manager was not ready."
                time.sleep(0.05)
            yield
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
