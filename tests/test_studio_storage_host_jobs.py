# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — real isolated hosted job custody tests

"""Run hosted adapters through actual authority, launcher, workers and artifacts.

Same-UID fixtures establish functional wiring only, not a distinct-role
deployment grade. A real transport deliberately drops a cancellation request
before authority dispatch; no production cancellation or worker is mocked.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from sc_neurocore.studio.api.host_job_tasks import HOST_ANALYSIS_TASK, HOST_COMPILER_TASK
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.storage_requester import delegated
from tests.studio_storage_generation_runs import (
    api_runtime,
    base as base,
    launcher as launcher,
    ledger as ledger,
)
from tests.studio_storage_generation_support import FRAME, Authority
from tests.studio_storage_launcher_support import Launcher
from tests.test_studio_host_job_tasks import build_request

ADMIN = Principal("operator", frozenset({"studio.admin"}))
HOST_ROUTE = "/api/studio/host-jobs"
CANCEL_ROUTE = "/api/studio/jobs/{job_id}/cancel"


class RequestDropAuthority(Authority):
    """Use real storage dispatch, optionally losing one cancel before it acts."""

    drop_cancel = False

    def _handle(self, channel: socket.socket, frame: bytes) -> None:
        """Drop the exact selected wire request; forward every other operation."""
        if self.drop_cancel and json.loads(frame)["operation"] == "cancel":
            self.drop_cancel = False
            return
        super()._handle(channel, frame)


@pytest.fixture
def authority(ledger: StudioJobLedger) -> Iterator[RequestDropAuthority]:
    """Own all actual server connections and reap every serving thread."""
    served = RequestDropAuthority(ledger)
    try:
        yield served
    finally:
        served.join()


@pytest.fixture
def manager(base: Path, launcher: Launcher, authority: RequestDropAuthority) -> IsolatedJobManager:
    """Bind actual authority and launcher using the normal isolated facade."""
    configuration = StorageBoundaryConfiguration(
        storage_uid=os.getuid(),
        api_uid=os.getuid() + 1,
        worker_uid=os.getuid() + 2,
        authority_root=base / "authority",
        spool_root=base / "spool",
        socket_path=base / "sock" / "storage.sock",
        workspace="default",
        frame_max_bytes=FRAME,
        max_metadata_bytes=FRAME,
        max_seed_bytes=FRAME,
        max_seed_entries=8,
        max_manifest_bytes=4096,
        max_artifact_bytes=1 << 20,
        max_artifact_entries=64,
        transfer_timeout_seconds=10,
        max_connections=4,
    )
    return IsolatedJobManager(
        api_runtime(base, authority),
        configuration,
        allowed_kinds=frozenset({"analysis", "compiler"}),
        default_timeout_seconds=90,
    )


@pytest.mark.parametrize("compiler", [False, True])
def test_hosted_named_task_runs_and_seals_result_at_actual_authority(
    manager: IsolatedJobManager, compiler: bool
) -> None:
    """Both task families cross the real launcher/process/storage artifact boundary."""
    envelope = (
        {"operation": "ir.build", "request": build_request()}
        if compiler
        else {
            "operation": "sensitivity",
            "request": {
                "equations": ["dv/dt=I-z*v+a"],
                "params": {"z": 1, "a": 2},
                "init": {"v": 0},
                "duration": 1,
            },
        }
    )
    with delegated(ADMIN, method="POST", route=HOST_ROUTE, request_id="isolated-host"):
        submitted = manager.submit_process_task(
            kind="compiler" if compiler else "analysis",
            owner="studio-host-compiler" if compiler else "studio-host-analysis",
            request_id="isolated-host",
            task_path=HOST_COMPILER_TASK if compiler else HOST_ANALYSIS_TASK,
            payload={"envelope_json": json.dumps(envelope)},
        )
        outcome = manager.wait(submitted.job_id, timeout_seconds=90)
    assert outcome.status == "completed", outcome.error
    assert outcome.result is not None
    if compiler:
        assert outcome.result["accepted"] is True
    else:
        rows = outcome.result["sensitivities"]
        assert isinstance(rows, list)
        assert [row["param"] for row in rows] == ["z", "a"]
    with delegated(
        ADMIN,
        method="GET",
        route="/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}",
        request_id="isolated-host",
    ):
        download = manager.read_artifact(submitted.job_id, "host/result.json")
    assert download.artifact.sha256 == hashlib.sha256(download.payload).hexdigest()
    assert json.loads(download.payload) == outcome.result
    with delegated(
        ADMIN, method="GET", route="/api/studio/jobs/status", request_id="isolated-host"
    ):
        assert manager.unreaped_workers == ()
    assert manager.generation_failures == {}


@pytest.mark.parametrize("failure", ["denied", "lost_request", "lost_reply", "none"])
def test_general_cancel_needs_authority_before_stopping_actual_worker(
    manager: IsolatedJobManager,
    authority: RequestDropAuthority,
    ledger: StudioJobLedger,
    failure: str,
) -> None:
    """Policy denial and a lost request cannot fall through to legacy local stop."""
    with delegated(ADMIN, method="POST", route=HOST_ROUTE, request_id="isolated-host"):
        submitted = manager.submit_process_task(
            kind="analysis",
            owner="studio-host-analysis",
            request_id=None,
            task_path=HOST_ANALYSIS_TASK,
            payload={
                "envelope_json": json.dumps(
                    {
                        "operation": "simulate",
                        "request": {
                            "equations": ["dv/dt=(I-v)/10"],
                            "duration": 100_000,
                            "dt": 0.1,
                        },
                    }
                )
            },
        )
        deadline = time.monotonic() + 30
        while (current := manager.record(submitted.job_id)).status != "running":
            assert current.status == "pending", current.error
            assert time.monotonic() < deadline, "Worker never started"
            time.sleep(0.025)
    try:
        principal = Principal("reader", frozenset()) if failure == "denied" else ADMIN
        if failure == "lost_request":
            authority.drop_cancel = True
        if failure == "lost_reply":
            authority.lose["cancel"] = 1
        with delegated(principal, method="POST", route=CANCEL_ROUTE, request_id="isolated-host"):
            if failure == "denied":
                with pytest.raises(PermissionError):
                    manager.cancel(submitted.job_id)
            elif failure in {"lost_request", "lost_reply"}:
                with pytest.raises(EOFError):
                    manager.cancel(submitted.job_id)
            else:
                assert manager.cancel(submitted.job_id).status in {"cancelling", "cancelled"}
        if failure in {"denied", "lost_request"}:
            assert ledger.record(submitted.job_id).status == "running"
            with delegated(
                ADMIN, method="GET", route="/api/studio/jobs/{job_id}", request_id="isolated-host"
            ):
                assert manager.wait(submitted.job_id, timeout_seconds=0.1).status == "running"
    finally:
        with delegated(ADMIN, method="POST", route=CANCEL_ROUTE, request_id="isolated-host"):
            manager.cancel(submitted.job_id)
            assert manager.wait(submitted.job_id, timeout_seconds=30).status == "cancelled"
            assert manager.cancel(submitted.job_id).status == "cancelled"
        with delegated(
            ADMIN, method="GET", route="/api/studio/jobs/status", request_id="isolated-host"
        ):
            assert manager.unreaped_workers == ()
