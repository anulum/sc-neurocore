# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — worker launcher test support

"""Real launcher processes, spool preparation and requests shared by launcher tests.

A launcher started here ends with the test thread that started it, and a launcher
that does not stop on SIGTERM is killed and reported, so no test leaves one running.
Its sockets live under a base short enough for the Unix socket path limit, whatever
the temporary folder of the run.

Workers run the interpreter of the test with ``-s``, so the packages they import must
be installed in that interpreter's own environment (a project virtual environment);
with packages only in the user site, a launched worker exits before it asks for its
grant.
"""

from __future__ import annotations

import ctypes
import functools
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
from sc_neurocore.studio.platform.storage_worker_grant import GRANT_ENDPOINT_NAME

# Bytes of a Unix socket address's path, its terminating zero included.
UNIX_SOCKET_PATH_LIMIT = 108
# The longest socket path below a base: a generation's grant endpoint in the spool.
LONGEST_SOCKET_SUFFIX = f"/spool/sj_{'0' * 16}/{'0' * 32}/{GRANT_ENDPOINT_NAME}"
# The base folder's own name: its prefix and the eight random characters of mkdtemp.
BASE_NAME = "/scl" + "0" * 8
# prctl option that asks the kernel to signal a process when the thread that started it ends.
_PR_SET_PDEATHSIG = 1
_PRCTL = ctypes.CDLL(None, use_errno=True).prctl if sys.platform == "linux" else None


def short_root() -> Path:
    """Return a temporary folder under which every launcher socket path fits the Unix limit.

    Returns
    -------
    Path
        The run's temporary folder when it is short enough, else ``XDG_RUNTIME_DIR``
        (the user's private runtime folder) when that is.

    Raises
    ------
    AssertionError
        If neither folder is short enough; the message names the largest length that
        fits, so the run can set ``TMPDIR`` to a shorter folder.
    """
    room = UNIX_SOCKET_PATH_LIMIT - 1 - len(BASE_NAME) - len(LONGEST_SOCKET_SUFFIX)
    candidates = (tempfile.gettempdir(), os.environ.get("XDG_RUNTIME_DIR", ""))
    for candidate in candidates:
        if candidate and os.path.isdir(candidate) and len(os.fsencode(candidate)) <= room:
            return Path(candidate)
    raise AssertionError(
        f"no temporary folder of at most {room} bytes for the launcher's sockets; "
        f"TMPDIR is {tempfile.gettempdir()!r} and XDG_RUNTIME_DIR "
        f"{os.environ.get('XDG_RUNTIME_DIR', '')!r}; set TMPDIR to a shorter folder"
    )


@contextmanager
def launcher_base() -> Iterator[Path]:
    """Yield a private base folder short enough that every socket path fits the Unix limit.

    Yields
    ------
    Path
        A new folder under :func:`short_root` with ``sock`` (mode 0700) and ``spool``
        (mode 0750); it is removed afterwards.
    """
    path = Path(tempfile.mkdtemp(prefix="scl", dir=short_root()))
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
    """One running launcher process and the paths it was configured with.

    Attributes
    ----------
    process : subprocess.Popen
        The service process, its output and error streams piped.
    socket_path : Path
        The service's request endpoint.
    spool_root : Path
        The spool its workers run in.
    """

    process: subprocess.Popen[bytes]
    socket_path: Path
    spool_root: Path


def configuration(base: Path, **changes: object) -> dict[str, object]:
    """Return a launcher configuration for one base, with any field replaced.

    Parameters
    ----------
    base : Path
        The base folder of :func:`launcher_base`.
    **changes : object
        Fields that replace the defaults.

    Returns
    -------
    dict of str to object
        The configuration: this process's identity for the API and the worker, the
        base's spool and socket, this interpreter and the checkout's ``src``.
    """
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


def _end_with_starter(parent: int) -> None:
    """In a new child: be killed when the starting thread ends; exit if it already has.

    Parameters
    ----------
    parent : int
        Process id of the starting process, read before the fork.
    """
    if _PRCTL is None:
        return
    if _PRCTL(_PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        os._exit(70)
    if os.getppid() != parent:
        # The starter ended between the fork and the request; nobody would stop this one.
        os._exit(70)


def start(base: Path, **changes: object) -> Launcher:
    """Start a launcher service process and wait until it is ready.

    On Linux the kernel kills the service when the thread that called this ends, so
    a test process that dies, even by SIGKILL, leaves no service behind; call it from
    the thread that outlives the service.

    Parameters
    ----------
    base : Path
        The base folder of :func:`launcher_base`.
    **changes : object
        Configuration fields that replace the defaults.

    Returns
    -------
    Launcher
        The running service and its socket and spool paths.

    Raises
    ------
    AssertionError
        If the service does not print ``ready``; the message is its error output.
    """
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
        preexec_fn=functools.partial(_end_with_starter, os.getpid()),
    )
    assert process.stdout is not None
    ready = process.stdout.readline()
    if ready != b"ready\n":
        process.wait(timeout=30.0)
        raise AssertionError(process.stderr.read().decode() if process.stderr else "")
    return Launcher(process, base / "sock" / "launcher.sock", base / "spool")


def shutdown(launcher: Launcher, *, timeout: float = 30.0) -> None:
    """Stop a launcher with SIGTERM; kill and report one that does not stop in time.

    Parameters
    ----------
    launcher : Launcher
        The running service.
    timeout : float
        Seconds to wait for the orderly stop, and again after a kill.

    Raises
    ------
    AssertionError
        If the service did not stop on SIGTERM; it has been killed by then.
    """
    try:
        launcher.process.send_signal(signal.SIGTERM)
        try:
            launcher.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            launcher.process.kill()
            launcher.process.wait(timeout=timeout)
            raise AssertionError(
                f"launcher {launcher.process.pid} did not stop on SIGTERM within "
                f"{timeout} s and was killed"
            ) from None
    finally:
        for stream in (launcher.process.stdout, launcher.process.stderr):
            if stream is not None:
                stream.close()


def prepare(spool_root: Path, job_id: str, generation: str) -> Path:
    """Write the input of one generation into the spool, as the API would.

    Parameters
    ----------
    spool_root : Path
        The launcher's spool.
    job_id, generation : str
        The job and generation names.

    Returns
    -------
    Path
        The generation's folder, with an analysis task descriptor and an empty payload.
    """
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
    """Send one request to a launcher and return its answer.

    Parameters
    ----------
    launcher : Launcher
        The running service.
    operation : LauncherOperation
        The request's operation.
    job_id, generation : str
        The generation addressed.

    Returns
    -------
    LauncherResponse
        The answer, within 30 seconds.
    """
    request = new_launcher_request(operation, job_id=job_id, generation=generation)
    return exchange_launcher_request(
        launcher.socket_path,
        request,
        launcher_uid=os.getuid(),
        deadline=time.monotonic() + 30.0,
    )


def await_state(launcher: Launcher, job_id: str, generation: str, state: str) -> LauncherResponse:
    """Poll a generation's status until it reaches a state or 60 seconds pass.

    Parameters
    ----------
    launcher : Launcher
        The running service.
    job_id, generation : str
        The generation addressed.
    state : str
        The state waited for.

    Returns
    -------
    LauncherResponse
        The last answer, in that state or not.
    """
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
