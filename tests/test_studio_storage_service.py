# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority service entry

"""The storage service starts, recovers, reconciles, refuses strangers and stops.

The service runs as its own process through its command-line entry over a
real namespace this test prepares. The test process is not the configured
API identity, so its connection is the real refusal of a foreign peer; the
service keeps running. Abandoned work is left by real processes that exit.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_service import StorageService, load_service_configuration
from tests.studio_seccomp_support import REPOSITORY
from tests.studio_storage_supervision_support import JOB, admit

DEAD_JOB = "sj_" + "d" * 16


@pytest.fixture
def base() -> Iterator[Path]:
    """Short private base so the service socket path fits the Unix limit."""
    path = Path(tempfile.mkdtemp(prefix="scs"))
    (path / "authority").mkdir(mode=0o700)
    (path / "endpoint").mkdir(mode=0o2750)
    (path / "endpoint").chmod(0o2750)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _configuration(base: Path) -> dict[str, object]:
    return {
        "boundary": {
            "storage_uid": os.getuid(),
            "api_uid": os.getuid() + 1,
            "worker_uid": os.getuid() + 2,
            "authority_root": str(base / "authority"),
            "spool_root": str(base / "spool"),
            "socket_path": str(base / "endpoint" / "storage.sock"),
            "workspace": "default",
            "frame_max_bytes": 8192,
            "max_metadata_bytes": 4096,
            "max_seed_bytes": 8192,
            "max_seed_entries": 16,
            "max_manifest_bytes": 1024,
            "max_artifact_bytes": 65536,
            "max_artifact_entries": 16,
            "transfer_timeout_seconds": 0.2,
            "max_connections": 2,
        },
        "max_concurrent": 2,
        "max_queued": 0,
        "audit_log_path": str(base / "audit.jsonl"),
        "reconcile_seconds": 0.2,
    }


def _dead_identity() -> str:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    identity = supervisor_identity(child.pid)
    child.wait(timeout=30)
    return identity


def test_configuration_is_exact(base: Path) -> None:
    """Only the complete, unambiguous configuration loads."""
    path = base / "service.json"
    path.write_text(json.dumps(_configuration(base)))
    assert load_service_configuration(path).max_concurrent == 2
    for text in (
        json.dumps({**_configuration(base), "extra": 1}),
        '{"max_queued": 0, "max_queued": 1}',
        "[" * 100000,
    ):
        path.write_text(text)
        with pytest.raises((ValueError, ValidationError)):
            load_service_configuration(path)


def test_the_service_recovers_reconciles_refuses_strangers_and_stops(base: Path) -> None:
    """Startup finishes a left purge; idle reconciliation resolves an abandoned job."""
    ledger = StudioJobLedger(root=base / "authority")
    try:
        # A committed purge of a record already deleted, left by an owner that
        # exited before cleanup; there was no directory to stage.
        with ledger.transaction() as connection:
            connection.execute(
                "INSERT INTO job_purges VALUES(?,?,?,?,?)",
                (DEAD_JOB, _dead_identity(), None, None, "committed"),
            )
        # A running job delegated to an API generation that has exited.
        live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        admit(ledger, supervisor=supervisor_identity(live.pid))
        ledger.transition(JOB, "running")
        live.kill()
        live.wait(timeout=30)
    finally:
        ledger.close()
    config = base / "service.json"
    config.write_text(json.dumps(_configuration(base)))
    service = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sc_neurocore.studio.platform.storage_service",
            "--configuration",
            str(config),
        ],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY / "src")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert service.stdout is not None
        assert service.stdout.readline() == b"ready\n"
        for _ in range(2):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stranger:
                stranger.settimeout(10)
                stranger.connect(str(base / "endpoint" / "storage.sock"))
                assert stranger.recv(1) == b""
        observer = StudioJobLedger(root=base / "authority")
        try:
            deadline = time.monotonic() + 30
            while observer.record(JOB).status == "running":
                assert time.monotonic() < deadline
                time.sleep(0.05)
            assert observer.record(JOB).status == "interrupted"
            purges = observer.connection().execute("SELECT COUNT(*) FROM job_purges")
            assert purges.fetchone()[0] == 0
        finally:
            observer.close()
        assert service.poll() is None
    finally:
        service.send_signal(signal.SIGTERM)
        assert service.wait(timeout=30) == 0, service.stderr.read() if service.stderr else ""
    assert not (base / "endpoint" / "storage.sock").exists()


def test_idle_reconciliation_waits_for_its_interval(base: Path) -> None:
    """Between intervals an idle service only waits; the next interval resolves."""
    payload = {**_configuration(base), "reconcile_seconds": 0.5}
    path = base / "service.json"
    path.write_text(json.dumps(payload))
    service = StorageService(load_service_configuration(path))
    try:
        with service.listener:
            service.reconcile()
            live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            admit(service.ledger, supervisor=supervisor_identity(live.pid))
            service.ledger.transition(JOB, "running")
            live.kill()
            live.wait(timeout=30)
            service.serve_once()
            assert service.ledger.record(JOB).status == "running"
            time.sleep(0.5)
            service.serve_once()
            assert service.ledger.record(JOB).status == "interrupted"
    finally:
        service.ledger.close()
