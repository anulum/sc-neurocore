# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job finish test support

"""A started job with a real worker, and a real finish exchange over a socket pair."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import os
import socket
import subprocess
import sys
import threading
import time

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_finish import serve_finish
from sc_neurocore.studio.platform.storage_finish_client import exchange_finish
from sc_neurocore.studio.platform.storage_finish_protocol import (
    FINISH_SCHEMA_VERSION,
    FinishArtifact,
    FinishOutcome,
    StorageFinishRequest,
    StorageFinishResponse,
)
from tests.studio_storage_supervision_support import (
    JOB,
    _run_before_acting_write,
    admit,
    exchange,
    start,
)

FRAME = 4096
FILES = {"reports/summary.json": b'{"ok": true}', "weights.bin": b"\x00\x01", "empty.txt": b""}


def reserved(ledger: StudioJobLedger) -> int:
    """Count admission reservations in the authority ledger."""
    return int(
        ledger.connection().execute("SELECT COUNT(*) FROM admission_reservations").fetchone()[0]
    )


def started(ledger: StudioJobLedger) -> subprocess.Popen[bytes]:
    """Admit the job to this API generation and start it with a real worker."""
    (ledger.path.parent).chmod(0o700)
    admit(ledger, supervisor=supervisor_identity())
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    response = exchange(ledger, start(supervisor_identity(worker.pid)))
    assert response.outcome == "started"
    return worker


def stop(worker: subprocess.Popen[bytes]) -> None:
    """Stop the worker for real, as the launcher does before finish."""
    if worker.poll() is None:
        worker.kill()
    worker.wait(timeout=10.0)


def request(
    files: dict[str, bytes],
    *,
    outcome: FinishOutcome = "completed",
    request_id: str = "f" * 32,
    workspace: str = "default",
    digests: dict[str, str] | None = None,
    worker_reaped: bool = True,
) -> StorageFinishRequest:
    """Declare ``files`` with their real digests unless ``digests`` overrides one."""
    digests = digests or {}
    return StorageFinishRequest(
        schema_version=FINISH_SCHEMA_VERSION,
        operation="finish",
        request_id=request_id,
        workspace=workspace,
        job_id=JOB,
        outcome=outcome,
        result={"answer": 42} if outcome == "completed" else None,
        error=None if outcome == "completed" else "worker failed",
        artifacts=tuple(
            FinishArtifact(
                relative_path=name,
                size_bytes=len(payload),
                sha256=digests.get(name, hashlib.sha256(payload).hexdigest()),
            )
            for name, payload in files.items()
        ),
        worker_reaped=worker_reaped,
    )


def finish(
    ledger: StudioJobLedger,
    sent: StorageFinishRequest,
    payloads: Sequence[bytes],
    *,
    competing: Callable[[], None] | None = None,
    frame_max_bytes: int = FRAME,
) -> StorageFinishResponse:
    """Run the real authority handler and the real client over one socket pair.

    ``competing`` runs as a concurrent writer when the handler begins its
    second write transaction, after the ownership check and before commit.
    """
    service, client = socket.socketpair()
    failures: list[BaseException] = []
    ran: list[bool] = []

    def serve() -> None:
        try:
            if competing is not None:
                _run_before_acting_write(ledger, competing, ran, failures)
            serve_finish(
                service,
                ledger=ledger,
                workspace="default",
                expected_api_uid=os.getuid(),
                frame_max_bytes=frame_max_bytes,
                max_artifact_bytes=65536,
                max_artifact_entries=16,
                deadline=time.monotonic() + 10.0,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            ledger.close()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        return exchange_finish(
            client,
            sent,
            payloads,
            expected_service_uid=os.getuid(),
            max_bytes=frame_max_bytes,
            deadline=time.monotonic() + 10.0,
        )
    finally:
        thread.join(timeout=10.0)
        assert not thread.is_alive()
        assert failures == []
        assert ran == ([] if competing is None else [True])
