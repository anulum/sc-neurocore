# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — worker launcher test support

"""Real launcher processes, spool preparation and requests shared by launcher tests."""

from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_process_protocol import _process_worker_environment
from sc_neurocore.studio.platform.storage_launcher_client import (
    exchange_launcher_request,
    new_launcher_request,
)
from sc_neurocore.studio.platform.storage_launcher_protocol import (
    LauncherOperation,
    LauncherResponse,
)


@contextmanager
def launcher_base() -> Iterator[Path]:
    """Short private base directory so every socket path fits the Unix limit."""
    path = Path(tempfile.mkdtemp(prefix="scl"))
    (path / "sock").mkdir(mode=0o700)
    (path / "spool").mkdir(mode=0o750)
    try:
        yield path
    finally:
        shutil.rmtree(path)


SRC = Path(__file__).resolve().parents[1] / "src"


ACCOUNT_PROCESS_CEILING = resource.getrlimit(resource.RLIMIT_NPROC)[0]


@dataclass
class Launcher:
    """One running launcher process and the paths it was configured with."""

    process: subprocess.Popen[bytes]
    socket_path: Path
    spool_root: Path


def configuration(base: Path, **changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "api_uid": os.getuid(),
        "worker_uid": os.getuid(),
        "worker_gid": os.getgid(),
        "spool_root": str(base / "spool"),
        "socket_path": str(base / "sock" / "launcher.sock"),
        "python_executable": sys.executable,
        "python_path": [str(SRC)],
        "max_workers": 4,
        "max_records": 16,
        "max_memory_bytes": 4 << 30,
        "max_cpu_seconds": 600,
        "max_open_files": 256,
        "max_file_bytes": 64 << 20,
        # RLIMIT_NPROC counts every process and thread of the UID, not one job;
        # a same-identity run must inherit the account's own ceiling.
        "max_processes": ACCOUNT_PROCESS_CEILING,
        "stop_rounds": 20,
        "transfer_timeout_seconds": 0.5,
    }
    body.update(changes)
    return body


def start(base: Path, **changes: object) -> Launcher:
    config_path = base / "launcher.json"
    config_path.write_text(json.dumps(configuration(base, **changes)))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sc_neurocore.studio.platform.storage_launcher_service",
            "--configuration",
            str(config_path),
        ],
        env=_process_worker_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    ready = process.stdout.readline()
    if ready != b"ready\n":
        process.wait(timeout=30.0)
        raise AssertionError(process.stderr.read().decode() if process.stderr else "")
    return Launcher(process, base / "sock" / "launcher.sock", base / "spool")


def shutdown(launcher: Launcher) -> None:
    launcher.process.send_signal(signal.SIGTERM)
    launcher.process.wait(timeout=30.0)
    for stream in (launcher.process.stdout, launcher.process.stderr):
        if stream is not None:
            stream.close()


def prepare(spool_root: Path, job_id: str, generation: str) -> Path:
    directory = spool_root / job_id / generation
    (directory / "input").mkdir(parents=True)
    (directory / job_id).mkdir()
    descriptor = {
        "version": "studio.worker.descriptor.v1",
        "job_id": job_id,
        "generation": generation,
        "task_name": "analysis.run",
        "authorized_route": "/api/analysis/jobs",
        "supervisor": supervisor_identity(),
        "max_artifact_bytes": 1024,
    }
    (directory / "input" / "descriptor.json").write_text(json.dumps(descriptor))
    (directory / "input" / "descriptor.json").chmod(0o640)
    (directory / "input" / "payload.json").write_text("{}")
    return directory


def send(
    launcher: Launcher, operation: LauncherOperation, job_id: str, generation: str
) -> LauncherResponse:
    request = new_launcher_request(operation, job_id=job_id, generation=generation)
    return exchange_launcher_request(
        launcher.socket_path,
        request,
        launcher_uid=os.getuid(),
        deadline=time.monotonic() + 30.0,
    )


def await_state(launcher: Launcher, job_id: str, generation: str, state: str) -> LauncherResponse:
    deadline = time.monotonic() + 60.0
    while True:
        response = send(launcher, "status", job_id, generation)
        if response.state == state or time.monotonic() > deadline:
            return response
        time.sleep(0.1)


JOB_A = "sj_" + "1" * 16


JOB_B = "sj_" + "2" * 16


GEN_A = "a" * 32


GEN_B = "b" * 32
