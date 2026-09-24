# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — launcher-started worker grant test support

"""Real worker processes and spool helpers shared by the grant test modules."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_process_protocol import _process_worker_environment
from sc_neurocore.studio.platform.storage_worker_grant import (
    ExpectedWorker,
)


@contextmanager
def held_grant_directory() -> Iterator[tuple[int, Path]]:
    """Hold a short private directory so the socket path fits the Unix limit."""
    path = Path(tempfile.mkdtemp(prefix="scg"))
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        yield descriptor, path
    finally:
        os.close(descriptor)
        shutil.rmtree(path)


NAME = "grant.sock"


WORKER = (
    "import sys\n"
    "from pathlib import Path\n"
    "from sc_neurocore.studio.platform.storage_worker_grant import receive_socket_grant\n"
    "print('imported', flush=True)\n"
    "try:\n"
    "    receive_socket_grant(Path(sys.argv[1]), expected_server_uid=int(sys.argv[2]))\n"
    "except BaseException as exc:\n"
    "    print(type(exc).__name__, flush=True)\n"
    "    raise SystemExit(3)\n"
    "print('granted', flush=True)\n"
)


def spawn_worker(endpoint: Path, server_uid: int) -> subprocess.Popen[str]:
    """Start a real worker and return once its imports finished, just before connect."""
    child = subprocess.Popen(
        [sys.executable, "-c", WORKER, str(endpoint), str(server_uid)],
        env=_process_worker_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "imported"
    return child


def expected_for(child: subprocess.Popen[str]) -> ExpectedWorker:
    _, pid, token = supervisor_identity(child.pid).split(":", 2)
    return ExpectedWorker(uid=os.getuid(), pid=int(pid), start_token=token)


def finish(child: subprocess.Popen[str]) -> tuple[int, str]:
    """Return exit status and final line, keeping text already buffered by ``readline``.

    ``communicate`` reads the raw descriptor and would drop lines the text
    wrapper prefetched while waiting for ``imported``.
    """
    assert child.stdout is not None and child.stderr is not None
    stdout = child.stdout.read()
    child.stderr.read()
    child.wait(timeout=10.0)
    child.stdout.close()
    child.stderr.close()
    return child.returncode, stdout.strip().splitlines()[-1]


def task_module(root: Path) -> Path:
    module = root / "grant_probe.py"
    module.write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_suffix('.imported').write_text('imported')\n"
        "def run(context,payload):\n"
        "    return {'granted_execution': payload['value']}\n"
    )
    return module


def process_worker(
    root: Path, extra: list[str], *, supervisor: bool = True
) -> subprocess.Popen[bytes]:
    payload = root / "payload.json"
    payload.write_text(json.dumps({"value": 7}))
    environment = _process_worker_environment()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment["PYTHONPATH"]
    command = [
        sys.executable,
        "-m",
        "sc_neurocore.studio.platform.process_worker",
        "--task",
        "grant_probe:run",
        "--payload",
        str(payload),
        "--result",
        str(root / "result.json"),
        "--work-dir",
        str(root / "work"),
        "--max-artifact-bytes",
        "1024",
        *extra,
    ]
    if supervisor:
        command += ["--supervisor", supervisor_identity()]
    return subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
