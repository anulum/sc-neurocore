# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — isolated Studio jobs facade, end to end

"""The API's job methods work through the storage authority and the launcher.

Each case uses the real launcher process, bootstrap and named worker, the
service's own dispatch with its production named-admission handler over a
real SQLite ledger, and the delegation the security middleware opens for an
authorised request. Same-identity runs establish function only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.training_process import TRAINING_PROCESS_TASK
from sc_neurocore.studio.training_contract import resolve_training_config
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.storage_requester import delegated
from sc_neurocore.studio.platform.studio_job_service import StudioJobService
from tests.studio_storage_finish_support import FILES, finish, request, started, stop
from tests.studio_storage_generation_runs import *
from tests.studio_storage_generation_support import FRAME, Authority
from tests.studio_storage_launcher_support import Launcher
from tests.studio_storage_supervision_support import JOB, admit
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity

ADMIN = Principal("operator", frozenset({"studio.admin"}))
ANALYSIS = "/api/analysis/jobs"
TASK = "sc_neurocore.studio.api.analysis_jobs:execute_analysis_process_task"


@contextmanager
def request_on(route: str, method: str = "POST") -> Iterator[None]:
    """Act inside an authorised request, as the security middleware opens it."""
    with delegated(ADMIN, method=method, route=route, request_id="trace-1"):
        yield


def _configuration(base: Path) -> StorageBoundaryConfiguration:
    return StorageBoundaryConfiguration(
        storage_uid=os.getuid(),
        api_uid=os.getuid() + 1,
        worker_uid=os.getuid() + 2,
        authority_root=base / "authority",
        spool_root=base / "spool",
        socket_path=base / "sock" / "storage.sock",
        workspace="default",
        frame_max_bytes=FRAME,
        max_metadata_bytes=FRAME,
        max_seed_bytes=1 << 16,
        max_seed_entries=8,
        max_manifest_bytes=4096,
        max_artifact_bytes=1 << 20,
        max_artifact_entries=64,
        transfer_timeout_seconds=10.0,
        max_connections=4,
    )


@pytest.fixture
def manager(base: Path, launcher: Launcher, authority: Authority) -> IsolatedJobManager:
    return IsolatedJobManager(
        api_runtime(base, authority),
        _configuration(base),
        allowed_kinds=frozenset({"analysis", "training"}),
        default_timeout_seconds=120.0,
    )


def _submit(
    manager: IsolatedJobManager,
    payload: dict[str, object],
    *,
    kind: str = "analysis",
    owner: str = "studio",
    task_path: str = TASK,
    idempotency_key: str | None = None,
    workspace: str | None = None,
    timeout_seconds: float | None = None,
    training_config: dict[str, object] | None = None,
) -> str:
    return manager.submit_process_task(
        kind=kind,
        owner=owner,
        request_id="trace-1",
        task_path=task_path,
        payload=payload,
        idempotency_key=idempotency_key,
        workspace=workspace,
        timeout_seconds=timeout_seconds,
        training_config=training_config,
    ).job_id


def test_a_submitted_job_completes_and_every_read_view_sees_it(
    manager: IsolatedJobManager, ledger: StudioJobLedger
) -> None:
    """Submit, wait, record, list, status and purge through the authority."""
    service: StudioJobService = manager  # the API depends on this structure only
    assert service is manager
    with request_on(ANALYSIS):
        job_id = _submit(manager, SIMULATE, idempotency_key="once")
        assert _submit(manager, SIMULATE, idempotency_key="once") == job_id
        completed = manager.wait(job_id, timeout_seconds=120)
    assert completed.status == "completed", completed.error
    assert completed.result is not None and completed.result["evidence_receipt"]
    with request_on(ANALYSIS):
        # The identical submission after completion returns the job, runs nothing.
        assert _submit(manager, SIMULATE, idempotency_key="once") == job_id
    assert manager.generation_failures == {}
    with request_on("/api/studio/jobs", "GET"):
        assert manager.record(job_id) == completed
        assert [record.job_id for record in manager.list_records()] == [job_id]
        assert manager.list_records(actor="someone-else") == ()
        assert manager.list_snapshot(workspace="default").records == (completed,)
        assert manager.purge_snapshot(limit=10).purges == ()
    with request_on("/api/studio/jobs/status", "GET"):
        status = manager.status()
        assert (status.completed_count, status.active_count, status.process_count) == (1, 0, 1)
        assert manager.unreaped_workers == ()
    with request_on("/api/studio/audit/quarantine/archive/purge"):
        assert manager.purge_terminal_record(job_id).job_id == job_id
    with pytest.raises(KeyError):
        ledger.record(job_id)


def test_nothing_is_done_for_nobody_or_outside_the_reviewed_contract(
    manager: IsolatedJobManager,
) -> None:
    """No delegation, other kinds, owners, routes, workspaces or bad timeouts refuse."""
    with pytest.raises(PermissionError):
        _submit(manager, SIMULATE)
    refusals: tuple[Callable[[], str], ...] = (
        lambda: _submit(manager, SIMULATE, kind="compiler"),
        lambda: _submit(manager, SIMULATE, owner="someone"),
        lambda: _submit(manager, SIMULATE, task_path="os:system"),
        lambda: _submit(manager, SIMULATE, workspace="elsewhere"),
        lambda: _submit(manager, SIMULATE, timeout_seconds=float("nan")),
        lambda: _submit(manager, SIMULATE, training_config={"epochs": 1}),
    )
    with request_on(ANALYSIS):
        for refused in refusals:
            with pytest.raises(StudioJobRejected):
                refused()
        with pytest.raises(StudioJobRejected):
            manager.submit_process_task(
                kind="unknown", owner="studio", request_id=None, task_path=TASK, payload={}
            )
        with pytest.raises(ValueError):
            manager.wait("sj_" + "0" * 16, timeout_seconds=float("inf"))


def test_a_running_job_takes_control_and_cancellation(
    manager: IsolatedJobManager, base: Path
) -> None:
    """A live job reads its spool, receives a command, and stops on cancel."""
    with request_on(ANALYSIS):
        job_id = _submit(manager, LONG)
        deadline = time.monotonic() + 90
        while manager.record(job_id).status != "running":
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert manager.read_live_artifact_bytes(job_id, "events.jsonl", offset=0) == (b"", 0)
        manager.send_control_command(job_id, command={"action": "pause"})
        (generation,) = (base / "spool" / job_id).iterdir()
        command = generation / job_id / ".studio_control" / "command.json"
        assert json.loads(command.read_text()) == {"action": "pause"}
        cancelled = manager.cancel(job_id)
        assert cancelled.status == "cancelling"
        finished = manager.wait(job_id, timeout_seconds=90)
        assert (finished.status, finished.error) == ("cancelled", None)
        assert manager.cancel(job_id).status == "cancelled"
        with pytest.raises(StudioJobRejected, match="not running"):
            manager.send_control_command(job_id, command={"action": "resume"})
        assert manager.wait(job_id, timeout_seconds=0.0).status == "cancelled"


def test_a_job_the_ledger_cannot_cancel_is_refused(
    manager: IsolatedJobManager, ledger: StudioJobLedger
) -> None:
    """An unknown job is returned by the authority and refused here, as embedded."""
    admit(ledger, supervisor=supervisor_identity())
    ledger.transition(JOB, "unknown")
    with request_on("/api/training/stop"), pytest.raises(StudioJobRejected, match="unknown"):
        manager.cancel(JOB)


def test_sealed_artefacts_are_read_only_on_an_artefact_route(
    manager: IsolatedJobManager, ledger: StudioJobLedger
) -> None:
    """The download route reads the sealed copy; another route may not."""
    stop(started(ledger))
    assert finish(ledger, request(FILES), list(FILES.values())).reply == "sealed"
    with request_on("/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}", "GET"):
        served = manager.read_artifact(JOB, "weights.bin")
    assert served.payload == FILES["weights.bin"]
    with request_on(ANALYSIS), pytest.raises(PermissionError):
        manager.read_artifact(JOB, "weights.bin")


def test_waiting_on_a_job_supervised_elsewhere_polls_the_record(
    manager: IsolatedJobManager, ledger: StudioJobLedger
) -> None:
    """Without a local generation, wait observes the durable record until its deadline."""
    admit(ledger, supervisor=supervisor_identity())
    with request_on("/api/studio/jobs/{job_id}", "GET"):
        assert manager.wait(JOB, timeout_seconds=0.2).status == "pending"
        timer = threading.Timer(0.3, lambda: ledger.transition(JOB, "failed", error="elsewhere"))
        timer.start()
        assert manager.wait(JOB).status == "failed"
        timer.join()


def test_full_capacity_refuses_a_valid_training_submission(manager: IsolatedJobManager) -> None:
    """With every slot held, a matching training snapshot is refused for capacity."""
    config = resolve_training_config({"epochs": 1}).to_public_dict()
    with request_on(ANALYSIS):
        running = [_submit(manager, LONG) for _ in range(2)]
    with request_on("/api/training/start"), pytest.raises(StudioJobQueueFull):
        manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id=None,
            task_path=TRAINING_PROCESS_TASK,
            payload=config,
            training_config=config,
        )
    with request_on("/api/training/stop"):
        for job_id in running:
            manager.cancel(job_id)
        assert {manager.wait(job_id, timeout_seconds=90).status for job_id in running} == {
            "cancelled"
        }


def test_an_unanswered_finish_is_kept_for_operators(
    manager: IsolatedJobManager, authority: Authority
) -> None:
    """A generation whose finish replies were all lost is reported, not hidden."""
    authority.lose["finish"] = 3
    with request_on(ANALYSIS):
        job_id = _submit(manager, SIMULATE)
        completed = manager.wait(job_id, timeout_seconds=120)
    deadline = time.monotonic() + 30
    while job_id not in manager.generation_failures:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert completed.status == "completed"
    assert isinstance(manager.generation_failures[job_id], TimeoutError)
