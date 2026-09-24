# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job supervision refusals and races

"""Foreign owners, malformed frames and ledger races never grant a worker.

Each race runs a real competing writer on its own SQLite connection, or exits
a real worker process, after the authority's ownership check and before its
acting write; each case asserts the safe outcome.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_peer import write_verified_frame
from sc_neurocore.studio.platform.storage_supervision import serve_supervision
from sc_neurocore.studio.platform.storage_supervision_client import (
    exchange_supervision_request,
    supervision_heartbeat_request,
    supervision_start_request,
)
from tests.studio_storage_supervision_support import *


def test_heartbeat_by_another_generation_is_refused(ledger: StudioJobLedger, worker: str) -> None:
    """A job delegated to another live generation is not renewed for this one."""
    admit(ledger, supervisor=worker)
    response = exchange(ledger, heartbeat())
    assert (response.outcome, response.reason) == ("refused", "not_owner")


def test_job_ending_before_heartbeat_is_not_live(ledger: StudioJobLedger) -> None:
    """A job cancelled after the ownership check is reported, not renewed."""
    admit(ledger, supervisor=supervisor_identity())
    response = exchange(ledger, heartbeat(), competing=lambda: _cancel(ledger))
    assert (response.outcome, response.reason) == ("refused", "not_live")
    assert ledger.record(JOB).status == "cancelled"


def test_job_ending_before_start_transition_is_not_live(
    ledger: StudioJobLedger, worker: str
) -> None:
    """A job that ends between the ownership check and the transition gets no worker."""
    admit(ledger, supervisor=supervisor_identity())
    response = exchange(ledger, start(worker), competing=lambda: _cancel(ledger))
    assert (response.outcome, response.reason) == ("refused", "not_live")
    assert ledger.record(JOB).status == "cancelled"
    assert worker_rows(ledger) == []


def test_worker_exiting_before_registration_is_unverified(ledger: StudioJobLedger) -> None:
    """A worker that exits after its liveness check is never registered.

    The job is left running without a worker, as in the embedded supervisor
    when a spawned child fails registration; the API then withholds the grant
    and stops the generation.
    """
    admit(ledger, supervisor=supervisor_identity())
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )

    def exit_worker() -> None:
        child.kill()
        child.wait(timeout=10.0)

    try:
        response = exchange(ledger, start(supervisor_identity(child.pid)), competing=exit_worker)
    finally:
        child.kill()
        child.wait(timeout=10.0)
    assert (response.outcome, response.reason) == ("refused", "worker_unverified")
    assert ledger.record(JOB).status == "running"
    assert worker_rows(ledger) == []


def _cancel(ledger: StudioJobLedger) -> None:
    """Cancel the pending job as a competing authority write would."""
    ledger.transition(JOB, "cancelled")


def _raw(ledger: StudioJobLedger, payload: bytes, *, workspace: str = "default") -> BaseException:
    service, client = socket.socketpair()
    failures: list[BaseException] = []

    def serve() -> None:
        try:
            serve_supervision(
                service,
                ledger=ledger,
                workspace=workspace,
                expected_api_uid=os.getuid(),
                max_bytes=4096,
                deadline=time.monotonic() + 10.0,
            )
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    with client:
        write_verified_frame(
            client,
            payload,
            expected_uid=os.getuid(),
            max_bytes=4096,
            deadline=time.monotonic() + 10.0,
        )
        client.settimeout(10.0)
        assert client.recv(16) == b""
    thread.join(timeout=10.0)
    assert len(failures) == 1
    return failures[0]


@pytest.mark.parametrize(
    "payload,match",
    [
        (b'{"schema_version":"studio.storage.supervision.v1"}', "validation error"),
        (b"{", "invalid storage supervision JSON"),
    ],
)
def test_malformed_requests_close_without_reply(
    ledger: StudioJobLedger, payload: bytes, match: str
) -> None:
    """A request that does not decode receives no outcome."""
    failure = _raw(ledger, payload)
    assert isinstance(failure, ValueError) and match in str(failure)


def test_foreign_workspace_request_closes_without_reply(
    ledger: StudioJobLedger, worker: str
) -> None:
    """The authority never serves a workspace other than its configured one."""
    request = start(worker, workspace="elsewhere")
    failure = _raw(ledger, request.model_dump_json().encode())
    assert isinstance(failure, ValueError)
    assert "does not match configured workspace" in str(failure)


def test_unconfigured_workspace_refuses(ledger: StudioJobLedger, worker: str) -> None:
    """An empty service workspace is a configuration error before any read."""
    service, client = socket.socketpair()
    with client, pytest.raises(ValueError, match="nonempty"):
        serve_supervision(
            service,
            ledger=ledger,
            workspace="",
            expected_api_uid=os.getuid(),
            max_bytes=4096,
            deadline=time.monotonic() + 1.0,
        )


def test_endpoint_client_refuses_before_connecting(tmp_path: Path, worker: str) -> None:
    """Another workspace or a process that is not the API identity sends nothing."""
    configuration = StorageBoundaryConfiguration(
        storage_uid=os.getuid() + 1,
        api_uid=os.getuid() + 2,
        worker_uid=os.getuid() + 3,
        authority_root=tmp_path / "authority",
        spool_root=tmp_path / "spool",
        socket_path=tmp_path / "endpoint" / "storage.sock",
        workspace="default",
        frame_max_bytes=4096,
        max_metadata_bytes=4096,
        max_seed_bytes=0,
        max_seed_entries=0,
        max_manifest_bytes=64,
        max_artifact_bytes=65536,
        max_artifact_entries=16,
        transfer_timeout_seconds=1.0,
        max_connections=1,
    )
    other = configuration.model_copy(update={"workspace": "elsewhere"})
    with pytest.raises(ValueError, match="does not match configured workspace"):
        exchange_supervision_request(
            configuration, supervision_start_request(other, job_id=JOB, worker=worker)
        )
    with pytest.raises(PermissionError, match="configured Linux API identity"):
        exchange_supervision_request(
            configuration, supervision_heartbeat_request(configuration, job_id=JOB)
        )
