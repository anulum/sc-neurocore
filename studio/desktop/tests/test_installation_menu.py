# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Registered native application lifecycle


"""Open the installed native application through its actual OS menu entry."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from sc_neurocore.studio.desktop.installation import DesktopInstallation
from sc_neurocore.studio.platform.jobs_ledger_supervisor import (
    supervisor_identity,
    supervisor_is_alive,
)

pytest_plugins = ["studio.desktop.tests.installation_fixtures"]

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication


def test_menu_entry_launches_owned_backend_and_closes_normally(
    installation: DesktopInstallation,
    installation_environment: dict[str, str],
    desktop_bundles: tuple[Path, Path],
    application: QApplication,
    tmp_path: Path,
) -> None:
    """Use the actual OS entry, HTTP backend, live lock and native Quit shortcut.

    Parameters
    ----------
    installation : DesktopInstallation
        Installed application observer in a disposable user namespace.
    installation_environment : dict of str to str
        Private HOME/XDG and X11 display inherited by actual OS commands.
    desktop_bundles : tuple[pathlib.Path, pathlib.Path]
        Complete real research runtimes used by installation and refused upgrade.
    application : QApplication
        Parent Qt application used only for a screenshot of the actual X11 window.
    tmp_path : pathlib.Path
        Native application/window-manager diagnostic output directory.
    """
    installation_environment = {
        key: value
        for key, value in installation_environment.items()
        if not key.startswith("SC_NEUROCORE_STUDIO_")
        and key
        not in {
            "PYTHONPATH",
            "PYTHONHOME",
            "LD_LIBRARY_PATH",
            "QT_PLUGIN_PATH",
            "QT_QPA_PLATFORM_PLUGIN_PATH",
            "QTWEBENGINEPROCESS_PATH",
            "QML_IMPORT_PATH",
            "QML2_IMPORT_PATH",
        }
    }
    first = installation.install(desktop_bundles[0])
    entry = installation.registration.paths[2]
    valid = subprocess.run(
        ["desktop-file-validate", str(entry)],
        env=installation_environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert valid.returncode == 0, valid.stdout + valid.stderr
    prefix = Path(installation_environment["SC_NEUROCORE_DESKTOP_TEST_WM_ROOT"])
    wm_environment = dict(installation_environment)
    for key in ("SESSION_MANAGER", "DESKTOP_AUTOSTART_ID"):
        wm_environment.pop(key, None)
    wm_environment["LD_LIBRARY_PATH"] = str(prefix / "usr/lib/x86_64-linux-gnu")
    wm_environment["XDG_DATA_DIRS"] = str(prefix / "usr/share") + ":/usr/share"
    identity: str | None = None
    backend_identity: str | None = None
    observations: dict[str, object] = {"build_id": first.build_id}
    with (tmp_path / "window-manager.txt").open("wb") as log:
        manager = subprocess.Popen(
            [
                str(prefix / "usr/bin/openbox"),
                "--sm-disable",
                "--config-file",
                str(prefix / "etc/xdg/openbox/rc.xml"),
            ],
            env=wm_environment,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                wm = subprocess.run(
                    ["wmctrl", "-m"],
                    env=installation_environment,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if wm.returncode == 0 and "Openbox" in wm.stdout:
                    break
                assert manager.poll() is None, "The private window manager stopped."
                time.sleep(0.1)
            else:
                raise AssertionError("The private window manager did not become ready.")
            with (tmp_path / "gio-launch.txt").open("wb") as output:
                launched = subprocess.run(
                    ["gio", "launch", str(entry)],
                    env=installation_environment,
                    cwd=tmp_path,
                    stdout=output,
                    stderr=output,
                    timeout=30,
                )
            assert launched.returncode == 0
            configuration = installation.profile.config / "desktop.json"
            windows: list[str] = []
            deadline = time.monotonic() + 75
            while time.monotonic() < deadline:
                if identity is None:
                    for proc in Path("/proc").iterdir():
                        if not proc.name.isdecimal():
                            continue
                        try:
                            args = (proc / "cmdline").read_bytes().split(b"\0")
                        except OSError:
                            continue
                        if (
                            args[0]
                            == str(installation.root / "current/runtime/bin/python3.12").encode()
                            and b"sc_neurocore.studio.desktop" in args
                        ):
                            identity = supervisor_identity(int(proc.name))
                found = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--name", "^SNN Studio$"],
                    env=installation_environment,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                windows = found.stdout.split()
                if windows and configuration.is_file():
                    break
                time.sleep(0.1)
            assert len(windows) == 1 and configuration.is_file()
            pid_result = subprocess.run(
                ["xdotool", "getwindowpid", windows[0]],
                env=installation_environment,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            pid = int(pid_result.stdout)
            command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
            assert command[0] == str(installation.root / "current/runtime/bin/python3.12").encode()
            assert b"sc_neurocore.studio.desktop" in command and b"-I" in command
            identity = supervisor_identity(pid)
            assert identity is not None
            for proc in Path("/proc").iterdir():
                if not proc.name.isdecimal():
                    continue
                try:
                    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                    args = (proc / "cmdline").read_bytes().split(b"\0")
                except OSError:
                    continue
                if int(fields[1]) == pid and b"sc_neurocore.studio.desktop.backend" in args:
                    backend_identity = supervisor_identity(int(proc.name))
            assert backend_identity is not None
            settings = json.loads(configuration.read_text())
            origin = "http://127.0.0.1:" + str(settings["port"])
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(origin + "/api/health", timeout=5) as reply:
                assert json.load(reply) == {"status": "ok"}
            refused = subprocess.run(
                [str(installation.registration.paths[1]), "install", str(desktop_bundles[1])],
                env=installation_environment,
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=180,
            )
            assert refused.returncode == 1 and "Close Studio" in refused.stderr
            assert (installation.root / "current").resolve().name == first.build_id
            assert supervisor_is_alive(identity) and supervisor_is_alive(backend_identity)
            initial_window = windows[0]
            current = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--pid", str(pid), "--name", "^SNN Studio$"],
                env=installation_environment,
                capture_output=True,
                text=True,
                timeout=5,
            )
            windows = current.stdout.split()
            assert current.returncode == 0 and len(windows) == 1
            observations["initial_native_window"] = initial_window
            observations.update(
                origin=origin,
                launcher_identity=identity,
                backend_identity=backend_identity,
                native_window=windows[0],
                qt_platform=application.platformName(),
                live_upgrade_refused=True,
            )
            screen = application.primaryScreen()
            assert screen is not None
            assert screen.grabWindow(0).save(str(tmp_path / "installed-window.png"))
            subprocess.run(
                ["xdotool", "windowactivate", "--sync", windows[0]],
                env=installation_environment,
                check=True,
                timeout=5,
            )
            subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "ctrl+q"],
                env=installation_environment,
                check=True,
                timeout=5,
            )
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and (
                supervisor_is_alive(identity) or supervisor_is_alive(backend_identity)
            ):
                time.sleep(0.1)
            assert screen.grabWindow(0).save(str(tmp_path / "after-quit.png"))
            assert supervisor_is_alive(identity) is False
            assert supervisor_is_alive(backend_identity) is False
            observations.update(
                origin=origin,
                launcher_identity=identity,
                backend_identity=backend_identity,
                native_window=windows[0],
            )
        finally:
            if identity is not None and supervisor_is_alive(identity):
                os.kill(int(identity.split(":")[1]), 15)
            if identity is not None:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and supervisor_is_alive(identity):
                    time.sleep(0.1)
            manager.terminate()
            manager.wait(timeout=10)
            (tmp_path / "menu-observation.json").write_text(json.dumps(observations, indent=2))
