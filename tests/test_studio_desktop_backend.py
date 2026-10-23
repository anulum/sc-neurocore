# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop parent loss and real worker process custody

"""Stop an actual backend and numerical job after its launcher is killed."""

from __future__ import annotations

import fcntl
import json
import os
import selectors
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from sc_neurocore.studio.desktop.runtime import DesktopBackend
from sc_neurocore.studio.platform.jobs_ledger_supervisor import (
    supervisor_identity,
    supervisor_is_alive,
)
from tests.test_studio_desktop_runtime import _environment, _request
from tests.test_studio_jobs_guard_cleanup import _children, _runs


def test_launcher_loss_stops_owned_backend_and_real_analysis_worker(tmp_path: Path) -> None:
    """Kill the launcher while a real public analysis job has an observed live worker.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Disposable user profile, actual backend log, configuration and ledger.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SC_NEUROCORE_STUDIO_")
    }
    env.update(
        HOME=str(tmp_path),
        TMPDIR=str(tmp_path),
        PYTHONDONTWRITEBYTECODE="1",
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        XDG_STATE_HOME=str(tmp_path / "state"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    driver = tmp_path / "launcher.py"
    driver.write_text('''# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real desktop backend launcher under parent-loss observation

"""Keep the actual launcher alive until its owner terminates it."""
import json
import sys
from sc_neurocore.studio.desktop.runtime import DesktopBackend
backend = DesktopBackend()
try:
    instance = backend.start()
    print(json.dumps({"pid":instance.pid,"origin":instance.origin}), flush=True)
    sys.stdin.buffer.read(1)
finally:
    backend.close()
''')
    identities: dict[int, str] = {}
    with (tmp_path / "launcher.log").open("wb") as log:
        parent = subprocess.Popen(
            [sys.executable, "-I", "-B", str(driver)],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            start_new_session=True,
        )
        try:
            assert parent.stdout is not None
            with selectors.DefaultSelector() as selector:
                selector.register(parent.stdout, selectors.EVENT_READ)
                assert selector.select(30), (tmp_path / "launcher.log").read_text()
                instance = json.loads(parent.stdout.readline())
            backend_pid = int(instance["pid"])
            identities[backend_pid] = supervisor_identity(backend_pid)
            origin = str(instance["origin"])
            request = urllib.request.Request(
                origin + "/api/analysis/jobs",
                data=json.dumps(
                    {
                        "analysis": "bifurcation",
                        "payload": {
                            "equations": ["dv/dt = (-v + I)/tau_m"],
                            "params": {"tau_m": 10.0},
                            "init": {"v": 0.0},
                            "sweep_param": "tau_m",
                            "sweep_min": 5.0,
                            "sweep_max": 15.0,
                            "sweep_steps": 5,
                            "duration": 10000.0,
                            "dt": 0.1,
                            "current": 10.0,
                        },
                    }
                ).encode(),
                headers={"Origin": origin, "Content-Type": "application/json"},
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=5) as response:
                receipt = json.load(response)
            assert receipt["execution_mode"] == "async_job"
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                for proc in Path("/proc").iterdir():
                    if not proc.name.isdecimal():
                        continue
                    try:
                        command = (proc / "cmdline").read_bytes()
                        fields = (proc / "stat").read_text().rsplit(")", 1)[-1].split()
                    except (OSError, IndexError):
                        continue
                    if (
                        int(fields[1]) == backend_pid
                        and b"sc_neurocore.studio.platform.process_worker" in command
                    ):
                        pid = int(proc.name)
                        identities[pid] = supervisor_identity(pid)
                if len(identities) > 1:
                    break
                time.sleep(0.01)
            assert len(identities) > 1, "The public job never produced an observable real worker."
            assert all(supervisor_is_alive(identity) is True for identity in identities.values())
            parent.kill()
            parent.wait(timeout=5)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and any(
                supervisor_is_alive(identity) is not False for identity in identities.values()
            ):
                time.sleep(0.05)
            assert all(
                supervisor_is_alive(identity) is False for identity in identities.values()
            ), identities
        finally:
            if parent.poll() is None:
                parent.kill()
            parent.wait(timeout=5)
            for pid, identity in identities.items():
                if supervisor_is_alive(identity) is True:
                    os.killpg(pid, signal.SIGKILL)
            if parent.stdin is not None:
                parent.stdin.close()
            if parent.stdout is not None:
                parent.stdout.close()


def test_normal_close_cancels_an_actual_worker_and_retains_its_record(tmp_path: Path) -> None:
    """Collect a live analysis worker and reopen its durable cancellation after quitting.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Private user profile shared only by this backend and its later restart.
    """
    env = _environment(tmp_path)
    backend = DesktopBackend(env)
    identities: dict[int, str] = {}
    try:
        instance = backend.start()
        config = backend.paths.config / "desktop.json"
        original = config.read_bytes()
        receipt = _request(
            instance.origin,
            "/api/analysis/jobs",
            {
                "analysis": "bifurcation",
                "payload": {
                    "equations": ["dv/dt = (-v + I)/tau_m"],
                    "params": {"tau_m": 10.0},
                    "init": {"v": 0.0},
                    "sweep_param": "tau_m",
                    "sweep_min": 5.0,
                    "sweep_max": 15.0,
                    "sweep_steps": 5,
                    "duration": 10000.0,
                    "dt": 0.1,
                    "current": 10.0,
                },
            },
        )
        assert isinstance(receipt, dict) and receipt["execution_mode"] == "async_job"
        job_id, route = receipt["job_id"], receipt["status_route"]
        assert isinstance(job_id, str) and isinstance(route, str)
        deadline = time.monotonic() + 5
        while not identities:
            identities = {
                pid: supervisor_identity(pid)
                for pid in _children(instance.pid)
                if _runs(pid, "sc_neurocore.studio.platform.process_worker")
            }
            assert time.monotonic() < deadline, "No real analysis worker was observed."
            time.sleep(0.01)
        live = _request(instance.origin, route)
        assert isinstance(live, dict) and live["status"] == "running"
        assert live["lease_owner"] == supervisor_identity(instance.pid)
        assert all(supervisor_is_alive(identity) is True for identity in identities.values())
        report = backend.close()
        assert report is not None and report.reaped and report.returncode == 0
        assert all(supervisor_is_alive(identity) is False for identity in identities.values())
        assert config.read_bytes() == original
        with (backend.paths.state / "backend.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        report = backend.close()
        assert report is None or report.reaped, report
        for pid, identity in identities.items():
            if supervisor_is_alive(identity) is True:
                os.killpg(pid, signal.SIGKILL)
    resumed = DesktopBackend(env)
    try:
        reopened = resumed.start()
        assert reopened.origin == instance.origin and reopened.pid != instance.pid
        stored = _request(reopened.origin, route)
        assert isinstance(stored, dict) and stored["job_id"] == job_id
        assert stored["status"] == "cancelled" and stored["error"] is None
        assert stored["finished_at_utc"] is not None and stored["lease_owner"] is None
        assert config.read_bytes() == original
    finally:
        report = resumed.close()
        assert report is not None and report.reaped and report.returncode == 0
