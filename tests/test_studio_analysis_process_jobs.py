# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Analysis process migration contracts

"""Named workers preserve the existing complete analysis output packets."""

from pathlib import Path
from datetime import datetime
import json
import os
import select
import signal
import sqlite3
import subprocess
import sys
import time

import pytest

from sc_neurocore.studio.api.analysis_jobs import (
    AnalysisJobValidationError,
    run_analysis_job_task,
    submit_analysis_job,
    validate_analysis_job_request,
)
from sc_neurocore.studio.api.schemas import AnalysisJobRequest
from sc_neurocore.studio.platform.jobs_manager import StudioJobManager
from sc_neurocore.studio.platform.jobs_worker_recovery import worker_group_stopped


@pytest.mark.parametrize(
    "analysis", ["simulate", "fi_curve", "bifurcation", "heatmap", "sensitivity"]
)
def test_named_process_matches_existing_thread_analysis(tmp_path: Path, analysis: str) -> None:
    """Every supported operation preserves all results, manifests and owner attribution."""
    payload = {
        "equations": ["dv/dt = (-v + gain*I)/tau"],
        "params": {"tau": 10.0, "gain": 1.0},
        "init": {"v": 0.0},
        "duration": 2.0,
        "dt": 0.1,
        "threshold": "v > 1",
        "reset": "v = 0",
    }
    variants: dict[str, dict[str, object]] = {
        "simulate": {},
        "fi_curve": {"i_min": 0.0, "i_max": 20.0, "i_steps": 3},
        "bifurcation": {
            "sweep_param": "tau",
            "sweep_min": 5.0,
            "sweep_max": 15.0,
            "sweep_steps": 5,
        },
        "heatmap": {
            "param_x": "tau",
            "x_min": 5.0,
            "x_max": 15.0,
            "x_steps": 3,
            "param_y": "gain",
            "y_min": 0.5,
            "y_max": 2.0,
            "y_steps": 3,
        },
        "sensitivity": {},
    }
    request = AnalysisJobRequest.model_validate(
        {"analysis": analysis, "payload": {**payload, **variants[analysis]}}
    )
    kind, normalized, _, _, _ = validate_analysis_job_request(request)
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=30.0
    )
    try:
        reference = manager.submit(
            kind="analysis",
            owner="studio",
            request_id=None,
            task=lambda context: run_analysis_job_task(kind, normalized, context),
        )
        expected = manager.wait(reference.job_id, 30.0)
        assert expected.status == "completed", expected.error
        receipt = submit_analysis_job(manager, request)
        observed = manager.wait(receipt["job_id"], 30.0)
        assert observed.status == "completed", observed.error
        assert observed.execution_model == "process"
        assert observed.owner == expected.owner == "studio"
        # Receipt creation time differs between sequential executions; every
        # numerical value, digest and other provenance field must be identical.
        left = json.loads(json.dumps(observed.result))
        right = json.loads(json.dumps(expected.result))
        for result in (left, right):
            timestamp = result["evidence_receipt"].pop("produced_at_utc")
            # Python 3.10 reads no "Z" suffix; the stored form uses it, so it is spelled out.
            assert datetime.fromisoformat(timestamp.replace("Z", "+00:00")).tzinfo is not None
        assert left == right
        assert receipt["schema_version"] == "studio.analysis.job.v1"
    finally:
        manager._ledger.close()


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_analysis_stop_reaps_worker_and_releases_shared_capacity(tmp_path: Path, stop: str) -> None:
    """A registered analysis stops durably and permits the next real analysis.

    Cancellation comes from a second manager over the same ledger. Observation
    begins at committed worker registration, not at an assumed numerical step;
    these cases qualify process lifetime, not progress inside the integrator.
    """
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=4.0 if stop == "timeout" else 30.0,
        max_concurrent_jobs=1,
        max_queued_jobs=0,
    )
    observer = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=30.0,
        max_concurrent_jobs=1,
        max_queued_jobs=0,
    )
    request = AnalysisJobRequest.model_validate(
        {
            "analysis": "fi_curve",
            "payload": {
                "equations": ["dv/dt = (-v + I)/tau"],
                "params": {"tau": 10.0},
                "init": {"v": 0.0},
                "duration": 2000.0,
                "dt": 0.1,
                "i_steps": 100,
            },
        }
    )
    job_id: str | None = None
    try:
        job_id = str(submit_analysis_job(manager, request)["job_id"])
        deadline = time.monotonic() + 10.0
        group_id = None
        with sqlite3.connect(f"file:{manager.ledger_path}?mode=ro", uri=True) as connection:
            while time.monotonic() < deadline:
                row = connection.execute(
                    "SELECT group_id FROM job_workers WHERE job_id=?", (job_id,)
                ).fetchone()
                if row is not None:
                    group_id = int(row[0])
                    break
                time.sleep(0.01)
        assert group_id is not None, manager.record(job_id)
        assert os.getpgid(group_id) == group_id
        if stop == "cancel":
            observer.cancel(job_id)
        completed = observer.wait(job_id, 15.0)
        assert completed.status == ("cancelled" if stop == "cancel" else "timed_out")
        assert completed.result is None
        assert completed.execution_model == "process"
        assert completed.owner == "studio"
        assert completed.finished_at_utc is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(group_id, 0)
        assert manager.unreaped_workers == ()
        # Terminal status can be visible just before admission release commits.
        deadline = time.monotonic() + 5.0
        while observer.status().admission["running"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert observer.status().admission["running"] == 0
        assert observer.status().admission["queued"] == 0
        transitions = observer.transitions(job_id)
        assert observer.cancel(job_id) == completed
        assert observer.transitions(job_id) == transitions
        observer.reconcile()
        assert observer.record(job_id) == completed
        next_request = AnalysisJobRequest.model_validate(
            {"analysis": "simulate", "payload": {"equations": ["dv/dt = -v"], "duration": 1.0}}
        )
        next_id = str(submit_analysis_job(observer, next_request)["job_id"])
        following = observer.wait(next_id, 30.0)
        assert following.status == "completed", following.error
        assert following.result is not None
        evidence = following.result["evidence_receipt"]
        assert isinstance(evidence, dict)
        assert evidence["status"] == "completed"
    finally:
        for record in manager.list_records():
            if record.status not in {"completed", "failed", "timed_out", "cancelled"}:
                observer.cancel(record.job_id)
                observer.wait(record.job_id, 15.0)
        observer._ledger.close()
        manager._ledger.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"analysis": "simulate", "payload": {}, "unknown": True},
        {"analysis": "unknown", "payload": {}, "parameter_order": []},
        {
            "analysis": "simulate",
            "payload": {"equations": ["dv/dt = I"], "dt": -1.0},
            "parameter_order": [],
        },
        *[
            {
                "analysis": "simulate",
                "payload": {
                    "equations": ["dv/dt = (-v + gain*I)/tau"],
                    "params": {"tau": 10.0, "gain": 1.0},
                    "duration": 2.0,
                },
                "parameter_order": order,
            }
            for order in (None, "tau,gain", ["tau"], ["tau", "tau"], ["tau", {}])
        ],
    ],
)
def test_worker_revalidates_untrusted_analysis_envelope(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    """A directly submitted malformed packet cannot obtain a successful result."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=30.0
    )
    try:
        record = manager.submit_process_task(
            kind="analysis",
            owner="studio",
            request_id=None,
            task_path="sc_neurocore.studio.api.analysis_jobs:execute_analysis_process_task",
            payload=payload,
        )
        completed = manager.wait(record.job_id, 30.0)
        assert completed.status == "failed"
        assert completed.result is None
        assert completed.error
    finally:
        manager._ledger.close()


@pytest.mark.parametrize("pause_worker", [False, True])
def test_analysis_supervisor_crash_recovers_without_reexecution(
    tmp_path: Path, pause_worker: bool
) -> None:
    """A real supervisor death leaves an interrupted receipt, never fabricated success.

    The registered worker must stop without a replacement manager. A fresh
    manager then recovers its slot and runs a different analysis; the original
    job and transition history remain sealed across a second reconciliation.
    """
    driver = """
import sys, threading
from pathlib import Path
from sc_neurocore.studio.api.analysis_jobs import submit_analysis_job
from sc_neurocore.studio.api.schemas import AnalysisJobRequest
from sc_neurocore.studio.platform.jobs_manager import StudioJobManager
manager = StudioJobManager(root=Path(sys.argv[1]), allowed_kinds=frozenset({'analysis'}),
    default_timeout_seconds=30.0, max_concurrent_jobs=1, max_queued_jobs=0)
request = AnalysisJobRequest.model_validate({'analysis': 'fi_curve', 'payload': {
    'equations': ['dv/dt = (-v + I)/tau'], 'params': {'tau': 10.0},
    'duration': 2000.0, 'dt': 0.1, 'i_steps': 100}})
print(submit_analysis_job(manager, request)['job_id'], flush=True)
threading.Event().wait(45.0)
"""
    supervisor = subprocess.Popen(
        [sys.executable, "-c", driver, str(tmp_path)],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    group_id: int | None = None
    identity = boot_id = ""
    try:
        assert supervisor.stdout is not None
        assert select.select([supervisor.stdout], [], [], 10.0)[0], "No submission receipt"
        job_id = supervisor.stdout.readline().decode().strip()
        assert job_id.startswith("sj_"), job_id
        with sqlite3.connect(f"file:{tmp_path / 'job_ledger.sqlite3'}?mode=ro", uri=True) as db:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                row = db.execute(
                    "SELECT worker_identity,boot_id,group_id FROM job_workers WHERE job_id=?",
                    (job_id,),
                ).fetchone()
                if row is not None:
                    identity, boot_id, group_id = str(row[0]), str(row[1]), int(row[2])
                    break
                time.sleep(0.01)
        assert group_id is not None
        assert not worker_group_stopped(identity, boot_id, group_id)
        if pause_worker:
            os.killpg(group_id, signal.SIGSTOP)
        supervisor.kill()
        assert supervisor.wait(timeout=5.0) == -signal.SIGKILL
        if pause_worker:
            held = StudioJobManager(
                root=tmp_path,
                allowed_kinds=frozenset({"analysis"}),
                default_timeout_seconds=30.0,
                max_concurrent_jobs=1,
                max_queued_jobs=0,
            )
            try:
                assert held.record(job_id).status == "interrupted"
                assert held.status().admission["running"] == 1
                assert not worker_group_stopped(identity, boot_id, group_id)
                with pytest.raises(AnalysisJobValidationError, match="analysis_job_rejected"):
                    submit_analysis_job(
                        held,
                        AnalysisJobRequest.model_validate(
                            {"analysis": "simulate", "payload": {"equations": ["dv/dt = -v"]}}
                        ),
                    )
                assert len(held.list_records()) == 1
            finally:
                held._ledger.close()
                os.killpg(group_id, signal.SIGCONT)
        deadline = time.monotonic() + 10.0
        while not worker_group_stopped(identity, boot_id, group_id) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert worker_group_stopped(identity, boot_id, group_id), "Orphan kept executing"
        recovered = StudioJobManager(
            root=tmp_path,
            allowed_kinds=frozenset({"analysis"}),
            default_timeout_seconds=30.0,
            max_concurrent_jobs=1,
            max_queued_jobs=0,
        )
        try:
            original = recovered.record(job_id)
            assert original.status == "interrupted"
            assert original.result is None
            assert original.owner == "studio"
            assert original.execution_model == "process"
            assert recovered.status().admission["running"] == 0
            transitions = recovered.transitions(job_id)
            recovered.reconcile()
            assert recovered.record(job_id) == original
            assert recovered.transitions(job_id) == transitions
            request = AnalysisJobRequest.model_validate(
                {"analysis": "simulate", "payload": {"equations": ["dv/dt = -v"], "duration": 1.0}}
            )
            next_id = str(submit_analysis_job(recovered, request)["job_id"])
            assert next_id != job_id
            following = recovered.wait(next_id, 30.0)
            assert following.status == "completed", following.error
            assert recovered.record(job_id) == original
            assert recovered.transitions(job_id) == transitions
            assert len(recovered.list_records()) == 2
        finally:
            for record in recovered.list_records():
                if record.status not in {
                    "completed",
                    "failed",
                    "timed_out",
                    "cancelled",
                    "interrupted",
                }:
                    recovered.cancel(record.job_id)
                    recovered.wait(record.job_id, 15.0)
            recovered._ledger.close()
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.wait(timeout=5.0)
        if supervisor.stdout is not None:
            supervisor.stdout.close()
        if group_id is not None and not worker_group_stopped(identity, boot_id, group_id):
            try:
                os.killpg(group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5.0
            while (
                not worker_group_stopped(identity, boot_id, group_id)
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert worker_group_stopped(identity, boot_id, group_id), "Test worker survived cleanup"
