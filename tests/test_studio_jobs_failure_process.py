# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real subprocess job failures

"""Exercise the public supervisor, worker CLI and durable process failure projection."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_failures import GENERIC_JOB_FAILURE
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_models import StudioJobRefused
from sc_neurocore.studio.platform.storage_finish_client import spool_finish_request
from tests.studio_storage_finish_support import finish, started, stop
from tests.studio_storage_supervision_support import JOB


@pytest.mark.parametrize(
    ("task", "limit", "public"),
    [
        ("write_unnameable_artifact", 64, GENERIC_JOB_FAILURE),
        ("write_oversized_artifact", 4, "Studio job artifact exceeds configured size limit."),
    ],
)
def test_process_failure_projects_safe_source_message(
    tmp_path: Path, task: str, limit: int, public: str
) -> None:
    """A real worker reports a known code or retains OS diagnostics behind fallback."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=15.0,
        max_artifact_bytes=limit,
    )
    job = manager.submit_process_task(
        kind="analysis",
        owner="operator",
        request_id=None,
        task_path="tests.studio_jobs_failure_tasks:" + task,
        payload={},
    )
    record = manager.wait(job.job_id, 20.0)
    assert record.status == "failed"
    assert record.to_public_dict()["error"] == public
    if task == "write_unnameable_artifact":
        assert record.error is not None and str(tmp_path) in record.error
    assert manager.status().active_count == 0
    assert manager.unreaped_workers == ()
    manager._ledger.close()


def test_analysis_validation_diagnostic_is_private_in_a_real_worker(tmp_path: Path) -> None:
    """The actual analysis task keeps its unmarked diagnostic behind the fallback."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=15.0,
    )
    try:
        job = manager.submit_process_task(
            kind="analysis",
            owner="operator",
            request_id=None,
            task_path="sc_neurocore.studio.api.analysis_jobs:execute_analysis_process_task",
            payload={},
        )
        record = manager.wait(job.job_id, 20.0)
        assert record.status == "failed" and record.result is None
        assert record.error == "invalid_analysis_payload"
        assert record.public_error == GENERIC_JOB_FAILURE
        assert record.to_public_dict()["error"] == GENERIC_JOB_FAILURE
        result = json.loads((tmp_path / job.job_id / ".studio_process_result.json").read_text())
        assert result["error"] == "invalid_analysis_payload"
        assert result["failure_schema"] == "studio.worker.failure.v1"
        assert result["refusal_code"] is None
        assert manager.status().active_count == 0 and manager.unreaped_workers == ()
    finally:
        manager._ledger.close()


@pytest.mark.parametrize(
    ("schema", "code", "public"),
    [
        (None, "artifact_size", GENERIC_JOB_FAILURE),
        ("studio.worker.failure.v1", "invented", GENERIC_JOB_FAILURE),
        (
            "studio.worker.failure.v1",
            "artifact_size",
            "Studio job artifact exceeds configured size limit.",
        ),
    ],
)
def test_untrusted_spool_text_stays_private_through_real_storage_finish(
    tmp_path: Path, schema: str | None, code: str, public: str
) -> None:
    """Read forged metadata and finish through actual peer-checked storage handlers."""
    ledger = StudioJobLedger(root=tmp_path / "authority")
    worker = started(ledger)
    stop(worker)
    spool = tmp_path / "spool"
    spool.mkdir()
    diagnostic = "worker failed at " + str(spool / "private-input.bin")
    (spool / ".studio_process_result.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "result": {},
                "artifacts": [],
                "error": diagnostic,
                "public_error": diagnostic,
                "failure_schema": schema,
                "refusal_code": code,
            }
        )
    )
    descriptor = os.open(spool, os.O_RDONLY | os.O_DIRECTORY)
    try:
        request, payloads = spool_finish_request(
            descriptor,
            workspace="default",
            job_id=JOB,
            outcome=None,
            exit_status=1,
            frame_max_bytes=4096,
            max_artifact_bytes=4096,
            max_artifact_entries=8,
        )
        assert request.error == diagnostic and request.public_error == public
        assert finish(ledger, request, payloads).reply == "sealed"
        record = ledger.record(JOB)
        assert record.error == diagnostic and record.to_public_dict()["error"] == public
        assert finish(ledger, request, payloads).reply == "already_sealed"
        assert (
            finish(
                ledger, request.model_copy(update={"public_error": "Different reason."}), payloads
            ).reply
            == "refused"
        )
        assert ledger.record(JOB) == record
    finally:
        os.close(descriptor)
        stop(worker)
        ledger.close()


@pytest.mark.parametrize(
    ("sealed", "different"),
    [(True, 1), (1.0, 1), (-0.0, 0.0)],
)
def test_storage_finish_retry_preserves_json_numeric_identity(
    tmp_path: Path, sealed: bool | int | float, different: bool | int | float
) -> None:
    """Actual finish refuses a changed JSON value even when Python equality matches."""
    from tests.studio_storage_finish_support import request as finish_request

    ledger = StudioJobLedger(root=tmp_path / "authority")
    worker = started(ledger)
    stop(worker)
    sent = finish_request({}).model_copy(update={"result": {"observation": sealed}})
    try:
        assert finish(ledger, sent, []).reply == "sealed"
        original = ledger.record(JOB)
        history = ledger.transitions(JOB)
        assert finish(ledger, sent, []).reply == "already_sealed"
        changed = sent.model_copy(update={"result": {"observation": different}})
        assert finish(ledger, changed, []).reply == "refused"
        assert ledger.record(JOB) == original and ledger.transitions(JOB) == history
    finally:
        stop(worker)
        ledger.close()


def test_refused_training_status_artifact_keeps_diagnostics_private(tmp_path: Path) -> None:
    """A real CLI training refusal retains safe status and evidence before re-raising."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=20.0,
    )
    submitted = manager.submit_process_task(
        kind="training",
        owner="operator",
        request_id=None,
        task_path="sc_neurocore.studio.platform.training_process:run_training_process_task",
        payload={"dataset": str(tmp_path / "private-data")},
    )
    try:
        record = manager.wait(submitted.job_id, 25.0)
        assert record.status == "failed" and record.error is not None
        assert str(tmp_path) in record.error and record.public_error == GENERIC_JOB_FAILURE
        for name in ("training/status.json", "training/evidence.json"):
            content, end = manager.read_live_artifact_bytes(record.job_id, name, offset=0)
            assert end == len(content) and content
            assert str(tmp_path).encode() not in content
            assert GENERIC_JOB_FAILURE.encode() in content
        assert manager.status().active_count == 0 and manager.unreaped_workers == ()
    finally:
        manager._ledger.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_process_input_is_refused_before_admission(tmp_path: Path, value: float) -> None:
    """The public process submitter refuses values that cannot cross as finite JSON."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=15.0,
        max_artifact_bytes=4,
    )
    admitted = None
    try:
        with pytest.raises(StudioJobRefused, match="payload must be JSON"):
            admitted = manager.submit_process_task(
                kind="analysis",
                owner="operator",
                request_id=None,
                task_path="tests.studio_jobs_failure_tasks:write_oversized_artifact",
                payload={"value": value},
            )
        assert manager.list_records() == () and manager.status().active_count == 0
    finally:
        if admitted is not None:
            assert manager.wait(admitted.job_id, 20.0).status == "failed"
            assert manager.unreaped_workers == ()
        manager._ledger.close()


@pytest.mark.parametrize("form", ["nonfinite", "invalid-utf8", "missing", "nonobject"])
def test_actual_invalid_worker_output_seals_a_failed_record(tmp_path: Path, form: str) -> None:
    """Actual worker output corruption cannot leave a live-looking durable job."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=15.0,
    )
    try:
        submitted = manager.submit_process_task(
            kind="analysis",
            owner="operator",
            request_id=None,
            task_path="tests.studio_jobs_failure_tasks:invalid_worker_output",
            payload={"form": form},
        )
        record = manager.wait(submitted.job_id, 8.0)
        assert record.status == "failed"
        message = {
            "nonfinite": GENERIC_JOB_FAILURE,
            "missing": "Studio process worker did not write a result.",
        }.get(form, "Studio process worker wrote an invalid result.")
        assert record.public_error == message
        assert str(tmp_path) not in str(record.to_public_dict())
        assert manager.status().active_count == 0 and manager.unreaped_workers == ()
        if form == "invalid-utf8":
            assert (
                tmp_path / submitted.job_id / ".studio_process_result.json"
            ).read_bytes() == b"\xff"
        elif form == "missing":
            assert not (tmp_path / submitted.job_id / ".studio_process_result.json").exists()
        elif form == "nonobject":
            assert (
                tmp_path / submitted.job_id / ".studio_process_result.json"
            ).read_bytes() == b"[]"
    finally:
        manager._ledger.close()


@pytest.mark.parametrize("wire_value", ["NaN", "Infinity", "-Infinity", "1e1000"])
def test_worker_cli_refuses_nonfinite_payload_before_task(tmp_path: Path, wire_value: str) -> None:
    """The actual worker entry point rejects nonfinite bytes before an artifact write."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"value":' + wire_value + "}")
    result_path = tmp_path / "result.json"
    command = [
        sys.executable,
        "-m",
        "sc_neurocore.studio.platform.process_worker",
        "--task",
        "tests.studio_jobs_failure_tasks:write_oversized_artifact",
        "--payload",
        str(payload_path),
        "--result",
        str(result_path),
        "--work-dir",
        str(work_dir),
        "--max-artifact-bytes",
        "4096",
    ]
    worker = subprocess.run(command, capture_output=True, text=True, timeout=15.0)
    assert worker.returncode == 1
    result = json.loads(result_path.read_text())
    assert result["status"] == "failed" and result["refusal_code"] is None
    assert "JSON" in result["error"]
    assert not (work_dir / "report.txt").exists()
