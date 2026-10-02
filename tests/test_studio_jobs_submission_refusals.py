# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Process submission and control refusals

"""Refuse inconsistent training snapshots and unavailable confined control paths."""

import threading
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import (
    STUDIO_CONTROL_SEED_DIR,
    StudioJobContext,
    StudioJobManager,
)
from sc_neurocore.studio.platform.jobs_models import StudioJobRefused
from sc_neurocore.studio.training_contract import resolve_training_config
from tests.studio_seccomp_support import run_child


@pytest.mark.parametrize("kind", ["analysis", "training"])
def test_training_snapshot_refusal_happens_before_admission(tmp_path: Path, kind: str) -> None:
    """An unexpected or inconsistent snapshot starts no job and takes no slot."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis", "training"}),
        default_timeout_seconds=10.0,
    )
    config = resolve_training_config({}).to_public_dict()
    try:
        with pytest.raises(StudioJobRefused) as refused:
            manager.submit_process_task(
                kind=kind,
                owner="operator",
                request_id=None,
                task_path="tests.studio_job_tasks:process_echo_task",
                payload={"config": {}},
                training_config=config,
            )
        expected = (
            "Only training jobs may carry a training configuration."
            if kind == "analysis"
            else "Training configuration snapshot does not match the process payload."
        )
        assert str(refused.value) == expected
        assert manager.list_records() == ()
        assert manager.status().active_count == 0
        assert manager._admission.snapshot().running == 0
    finally:
        manager._ledger.close()


def test_oversized_submission_seed_fails_before_worker_start(tmp_path: Path) -> None:
    """A real binary seed exceeding the manager's limit releases admitted capacity."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=10.0,
        max_artifact_bytes=16,
    )
    try:
        with pytest.raises(StudioJobRefused) as refused:
            manager.submit_process_task(
                kind="analysis",
                owner="operator",
                request_id=None,
                task_path="tests.studio_job_tasks:process_echo_task",
                payload={},
                seed_inputs={"input.bin": b"x" * 17},
            )
        assert str(refused.value) == "Studio job seed input exceeds configured size limit."
        records = manager.list_records()
        assert len(records) == 1 and records[0].status == "failed"
        assert records[0].public_error == str(refused.value)
        assert manager._admission.snapshot().running == 0
        assert manager.unreaped_workers == ()
        assert not list(manager.root.glob("*/.studio_process_payload.json"))
    finally:
        manager._ledger.close()


def test_an_untyped_python_caller_cannot_submit_text_as_binary_seed(tmp_path: Path) -> None:
    """The actual public API rejects a dynamic caller's wrong seed type before launch."""
    root = tmp_path / "jobs"
    observed = run_child(
        "import json, sys\n"
        "from pathlib import Path\n"
        "from sc_neurocore.studio.platform.jobs import StudioJobManager\n"
        "from sc_neurocore.studio.platform.jobs_models import StudioJobRefused\n"
        "manager = StudioJobManager(root=Path(sys.argv[1]),\n"
        "    allowed_kinds=frozenset({'analysis'}), default_timeout_seconds=10.0)\n"
        "try:\n"
        "    manager.submit_process_task(kind='analysis', owner='operator', request_id=None,\n"
        "        task_path='tests.studio_job_tasks:process_echo_task', payload={},\n"
        "        seed_inputs={'input.bin': 'text is not binary'})\n"
        "except StudioJobRefused as refusal:\n"
        "    records = manager.list_records()\n"
        "    print(json.dumps({'refusal': str(refusal),\n"
        "        'records': [[r.status, r.public_error] for r in records],\n"
        "        'running': manager.status().active_count,\n"
        "        'unreaped': list(manager.unreaped_workers)}))\n"
        "else:\n"
        "    raise AssertionError('a text seed was accepted')\n",
        arguments=(str(root),),
    )
    reason = "Studio job seed input must be bytes."
    assert observed == {
        "refusal": reason,
        "records": [["failed", reason]],
        "running": 0,
        "unreaped": [],
    }
    assert not list(root.glob("*/.studio_seed/input.bin"))
    assert not list(root.glob("*/.studio_process_payload.json"))


@pytest.mark.parametrize(
    "fault", ["missing-workdir", "escaped-workdir", "escaped-seed-root", "escaped-seed-child"]
)
def test_running_control_refusal_preserves_input_and_capacity(tmp_path: Path, fault: str) -> None:
    """Actual missing paths and symlinks are refused before any control data is written."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=20.0,
    )
    started = threading.Event()
    release = threading.Event()

    def task(context: StudioJobContext) -> dict[str, object]:
        started.set()
        assert release.wait(15.0)
        return {}

    record = manager.submit(kind="analysis", owner="operator", request_id=None, task=task)
    try:
        assert started.wait(5.0)
        work = manager.root / record.job_id
        outside = tmp_path / "outside"
        outside.mkdir()
        original_input = outside / "input.bin"
        original_input.write_bytes(b"retained input")
        if fault in {"missing-workdir", "escaped-workdir"}:
            work.rename(tmp_path / "retained-workdir")
            if fault == "escaped-workdir":
                work.symlink_to(outside, target_is_directory=True)
        elif fault == "escaped-seed-root":
            (work / STUDIO_CONTROL_SEED_DIR).symlink_to(outside, target_is_directory=True)
        else:
            seeds = work / STUDIO_CONTROL_SEED_DIR
            seeds.mkdir()
            (seeds / "input.bin").symlink_to(original_input)
        before = manager.record(record.job_id)
        with pytest.raises(StudioJobRefused) as refused:
            manager.send_control_command(
                record.job_id,
                command={"command": "continue"},
                seed_inputs={"input.bin": b"new input"},
            )
        expected = {
            "missing-workdir": "Studio job work directory is unavailable.",
            "escaped-workdir": "Studio job path escapes the job root.",
            "escaped-seed-root": "Studio job seed-input path escapes the seed directory.",
            "escaped-seed-child": "Studio job seed-input path escapes the seed directory.",
        }[fault]
        assert str(refused.value) == expected
        assert str(tmp_path) not in str(refused.value)
        assert original_input.read_bytes() == b"retained input"
        assert manager.record(record.job_id) == before
        assert manager._admission.snapshot().running == 1
        assert manager.unreaped_workers == ()
    finally:
        release.set()
        outcome = manager.wait(record.job_id, timeout_seconds=10.0)
        assert outcome.status == "completed", outcome.error
        assert manager._admission.snapshot().running == 0
        manager._ledger.close()
