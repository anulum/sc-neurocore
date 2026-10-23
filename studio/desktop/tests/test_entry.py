# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native desktop module entry integration

"""Launch the actual application and exercise missing-runtime/profile refusals."""

from __future__ import annotations

import fcntl
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import venv
from pathlib import Path

import coverage
import pytest

from sc_neurocore.studio.desktop.runtime import DesktopBackend
from sc_neurocore.studio.platform.jobs_ledger_supervisor import (
    supervisor_identity,
    supervisor_is_alive,
)
from tests.test_studio_jobs_guard_cleanup import _children, _runs


def _window_identifier(pid: int) -> str | None:
    windows = subprocess.run(["wmctrl", "-lp"], capture_output=True, text=True, timeout=3)
    assert windows.returncode in (0, 1), windows.stderr
    identifiers = []
    for line in windows.stdout.splitlines():
        fields = line.split(None, 4)
        if len(fields) == 5 and fields[2] == str(pid) and fields[4] == "SNN Studio":
            identifiers.append(fields[0])
    assert len(identifiers) <= 1, windows.stdout
    return identifiers[0] if identifiers else None


def _renderer_process(pid: int) -> int | None:
    pending = _children(pid)
    while pending:
        child = pending.pop()
        try:
            command = Path(f"/proc/{child}/cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if (
            b"QtWebEngineProcess" in command
            and b"--type=renderer" in command.replace(b"\0", b" ").split()
        ):
            return child
        pending.extend(_children(child))
    return None


def test_missing_qt_runtime_explains_refusal_before_creating_user_state(
    desktop_backend: DesktopBackend, tmp_path: Path
) -> None:
    """Run the real module in an actual dependency-free interpreter prefix.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Private profile whose untouched directories are observed after refusal.
    tmp_path : pathlib.Path
        Disposable interpreter prefix, outside the retained user profile.
    """
    prefix = tmp_path / "without-qt"
    venv.EnvBuilder(with_pip=False).create(prefix)
    site = prefix / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    source = Path(inspect.getfile(DesktopBackend)).resolve().parents[3]
    (site / "sc_neurocore_source.pth").write_text(str(source) + "\n")
    measured = Path(coverage.__file__).resolve().parent
    shutil.copytree(
        measured, site / "coverage", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copyfile(measured.parent / "a1_coverage.pth", site / "a1_coverage.pth")
    result = subprocess.run(
        [str(prefix / "bin/python"), "-I", "-B", "-m", "sc_neurocore.studio.desktop"],
        env=desktop_backend.environ,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr.strip() == "The Studio desktop runtime is missing Qt WebEngine."
    assert not desktop_backend.paths.config.exists()
    assert not desktop_backend.paths.state.exists()


@pytest.mark.parametrize("obstruction", ["relative", "file", "readonly"])
def test_profile_refusal_precedes_native_application_start(
    desktop_backend: DesktopBackend, tmp_path: Path, obstruction: str
) -> None:
    """Preserve real user bytes when profile resolution or directory preparation fails.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Disposable user environment passed unchanged except for the stated input.
    tmp_path : pathlib.Path
        Parent of an existing file that must not become a configuration directory.
    obstruction : str
        Relative base, an existing file, or a real unwritable configuration directory.
    """
    env = dict(desktop_backend.environ)
    retained = tmp_path / "existing research.txt"
    retained.write_bytes(b"keep this user work\n")
    readonly = tmp_path / "readonly configuration"
    readonly.mkdir()
    env["XDG_CONFIG_HOME"] = {
        "relative": "relative",
        "file": str(retained),
        "readonly": str(readonly),
    }[obstruction]
    readonly.chmod(0o500)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                "from sc_neurocore.studio.desktop.__main__ import main; raise SystemExit(main())",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        readonly.chmod(0o700)
    assert result.returncode == 1, result.stdout + result.stderr
    expected = {
        "relative": "absolute",
        "file": "could not be resolved",
        "readonly": "Studio could not prepare its desktop directories.",
    }
    assert expected[obstruction] in result.stderr
    assert "Traceback" not in result.stderr
    assert retained.read_bytes() == b"keep this user work\n"
    assert not (desktop_backend.paths.state / "backend.lock").exists()


@pytest.mark.usefixtures("window_manager")
@pytest.mark.parametrize("entrypoint", ["module", "gui-script"])
def test_module_entry_opens_its_real_window_and_closes_the_owned_backend(
    desktop_backend: DesktopBackend, tmp_path: Path, entrypoint: str
) -> None:
    """Close the actual separate application through its PID-matched native window.

    Parameters
    ----------
    desktop_backend : DesktopBackend
        Private retained environment used by the independent application process.
    tmp_path : pathlib.Path
        Destination for actual application diagnostics and native window receipt.
    entrypoint : str
        Package module or the GUI script produced by the real editable install.
    """
    identity: str | None = None
    backend_pid: int | None = None
    configuration = desktop_backend.paths.config / "desktop.json"
    command = [sys.executable, "-I", "-B", "-m", "sc_neurocore.studio.desktop"]
    if entrypoint == "gui-script":
        launcher = Path(sys.executable).parent / "snn-studio"
        assert launcher.is_file(), "The declared GUI script was not installed."
        command = [str(launcher)]
    with (tmp_path / "application.log").open("wb") as log:
        application = subprocess.Popen(
            command,
            env=desktop_backend.environ,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 40
            identifier: str | None = None
            while identifier is None or not configuration.is_file():
                assert application.poll() is None, (tmp_path / "application.log").read_text()
                identifier = _window_identifier(application.pid)
                assert time.monotonic() < deadline, (
                    "The actual application window was not admitted."
                )
                time.sleep(0.025)
            initial_identifier = identifier
            original = configuration.read_bytes()
            settings = json.loads(original)
            origin = f"http://127.0.0.1:{settings['port']}"
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(origin + "/api/health", timeout=5) as response:
                assert json.loads(response.read(4097)) == {"status": "ok"}
            children = [
                pid
                for pid in _children(application.pid)
                if _runs(pid, "sc_neurocore.studio.desktop.backend")
            ]
            (backend_pid,) = children
            identity = supervisor_identity(backend_pid)
            assert supervisor_is_alive(identity) is True
            (tmp_path / "startup-receipt.json").write_text(
                json.dumps(
                    {
                        "application_pid": application.pid,
                        "backend_pid": backend_pid,
                        "display": os.environ["DISPLAY"],
                    }
                )
            )
            renderer = _renderer_process(application.pid)
            identifier = _window_identifier(application.pid)
            while renderer is None or identifier is None:
                assert application.poll() is None, (tmp_path / "application.log").read_text()
                assert time.monotonic() < deadline, "The actual browser renderer did not start."
                time.sleep(0.025)
                renderer = _renderer_process(application.pid)
                identifier = _window_identifier(application.pid)
            (tmp_path / "before-close.json").write_text(
                json.dumps(
                    {
                        "window": identifier,
                        "initial_window": initial_identifier,
                        "renderer_pid": renderer,
                        "application_pid": application.pid,
                        "backend_pid": backend_pid,
                        "origin": origin,
                        "display": os.environ["DISPLAY"],
                    }
                )
            )
            subprocess.run(["wmctrl", "-ic", identifier], check=True, timeout=5)
            try:
                result = application.wait(timeout=30)
            except subprocess.TimeoutExpired:
                windows = subprocess.run(
                    ["wmctrl", "-lp"], capture_output=True, text=True, timeout=3
                )
                (tmp_path / "close-timeout.json").write_text(
                    json.dumps(
                        {
                            "windows": windows.stdout,
                            "backend_live": supervisor_is_alive(identity),
                            "application_live": application.poll() is None,
                        }
                    )
                )
                raise
            assert result == 0, (tmp_path / "application.log").read_text()
            assert supervisor_is_alive(identity) is False
            assert configuration.read_bytes() == original
            with (desktop_backend.paths.state / "backend.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (tmp_path / "entry-receipt.json").write_text(
                json.dumps(
                    {
                        "window": identifier,
                        "application_pid": application.pid,
                        "backend_pid": backend_pid,
                        "origin": origin,
                        "exit": application.returncode,
                    }
                )
            )
        finally:
            if application.poll() is None:
                application.kill()
            application.wait(timeout=5)
            if identity is not None and backend_pid is not None:
                deadline = time.monotonic() + 8
                while supervisor_is_alive(identity) is True and time.monotonic() < deadline:
                    time.sleep(0.025)
                if supervisor_is_alive(identity) is True:
                    os.killpg(backend_pid, signal.SIGKILL)
