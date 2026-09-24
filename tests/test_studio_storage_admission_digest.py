# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage admission content digest tests

"""Exercise service-derived replay content through real SQLite admission."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import TypedDict, cast

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_admission_replay import StorageAdmissionReplay
from sc_neurocore.studio.platform.jobs_ledger_schema import StudioJobSubmission
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.storage_admission_digest import derive_storage_admission_replay
from sc_neurocore.studio.platform.storage_named_tasks import (
    NamedStudioTask,
    resolve_named_studio_task,
)


class _DigestArgs(TypedDict):
    requester: Principal
    mutation_id: str
    workspace: str
    task: NamedStudioTask
    authorized_route: str
    payload_json: bytes
    seed_inputs: dict[str, bytes]
    execution_timeout_seconds: float
    queue_wait_seconds: float | None
    admission: dict[str, object] | None
    training_config: dict[str, object] | None
    experiment_sha256: str | None
    max_metadata_bytes: int
    max_seed_bytes: int
    max_seed_entries: int


def _content(**overrides: object) -> _DigestArgs:
    fields: dict[str, object] = {
        "requester": Principal("operator", frozenset({"studio.admin"})),
        "mutation_id": "same-mutation",
        "workspace": "default",
        "task": resolve_named_studio_task("analysis.run", authorized_route="/api/analysis/jobs"),
        "authorized_route": "/api/analysis/jobs",
        "payload_json": b'{"model":"lif","parameters":{"a":1,"b":2}}',
        "seed_inputs": {"input/a.bin": b"\x01\x02", "input/b.bin": b"\x03"},
        "execution_timeout_seconds": 30.0,
        "queue_wait_seconds": None,
        "admission": {"budget": 3},
        "training_config": None,
        "experiment_sha256": None,
        "max_metadata_bytes": 4096,
        "max_seed_bytes": 8,
        "max_seed_entries": 2,
    }
    fields.update(overrides)
    return cast(_DigestArgs, fields)


def _admit(
    admission: SharedJobAdmission, replay: StorageAdmissionReplay, job_id: str
) -> StudioJobSubmission:
    return admission.admit(
        job_id=job_id,
        kind="analysis",
        actor="studio",
        workspace="default",
        request_id="trace-one",
        idempotency_key=None,
        experiment_sha256=None,
        admission={"budget": 3},
        execution_model="process",
        replay=replay,
    )


def test_equivalent_content_replays_original_sqlite_admission(tmp_path: Path) -> None:
    """JSON and seed order never turn a retry into a second admission."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    original = derive_storage_admission_replay(**_content())
    equivalent = derive_storage_admission_replay(
        **_content(
            payload_json=b'{"parameters":{"b":2,"a":1},"model":"lif"}',
            seed_inputs={"input/b.bin": b"\x03", "input/a.bin": b"\x01\x02"},
        )
    )
    try:
        assert equivalent == original
        first = _admit(admission, original, "sj_0000000000000001")
        repeated = _admit(admission, equivalent, "sj_0000000000000002")
        assert not first.duplicate
        assert repeated.duplicate
        assert repeated.record == first.record
        assert admission.snapshot().admitted == 1
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "change",
    [
        {"payload_json": b'{"model":"lif","parameters":{"a":1,"b":4}}'},
        {"seed_inputs": {"input/a.bin": b"\x01\x04", "input/b.bin": b"\x03"}},
        {"execution_timeout_seconds": 31.0},
        {"queue_wait_seconds": 0.0},
        {
            "task": resolve_named_studio_task(
                "model.scan", authorized_route="/api/models/scan/jobs"
            ),
            "authorized_route": "/api/models/scan/jobs",
        },
        {
            "task": resolve_named_studio_task(
                "training.attach",
                authorized_route="/api/studio/training/weight-restore/attach",
            ),
            "authorized_route": "/api/studio/training/weight-restore/attach",
        },
        {"requester": Principal("operator", frozenset({"studio.admin", "auditor"}))},
        {"admission": {"budget": 4}},
        {"training_config": {"epochs": 2}},
        {"experiment_sha256": "b" * 64},
    ],
)
def test_changed_content_rejects_same_mutation_in_sqlite(
    tmp_path: Path, change: dict[str, object]
) -> None:
    """Every named admission input participates in the durable replay digest."""
    ledger = StudioJobLedger(root=tmp_path)
    admission = SharedJobAdmission(ledger, max_concurrent=2, max_queued=0)
    original = derive_storage_admission_replay(**_content())
    changed = derive_storage_admission_replay(**_content(**change))
    try:
        assert changed.payload_sha256 != original.payload_sha256
        _admit(admission, original, "sj_0000000000000001")
        with pytest.raises(StudioJobRejected, match="changed content"):
            _admit(admission, changed, "sj_0000000000000002")
        assert admission.snapshot().admitted == 1
        assert len(ledger.list_records()) == 1
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "change",
    [
        {"payload_json": b'{"model":"lif","model":"adex"}'},
        {"payload_json": b'{"parameters":{"a":1,"a":2}}'},
        {"payload_json": b'{"voltage":NaN}'},
        {"payload_json": b"[]"},
        {"payload_json": b"\xff"},
        {"payload_json": b"{" + b"a" * 4096 + b"}"},
        {"seed_inputs": {"bad\nname": b"x"}},
        {"seed_inputs": {"good": bytearray(b"x")}},
        {"seed_inputs": {"a": b"123456789"}},
        {"seed_inputs": {"a": b"", "b": b"", "c": b""}},
        {"seed_inputs": []},
        {"admission": {"budget": float("nan")}},
        {"execution_timeout_seconds": float("inf")},
        {"execution_timeout_seconds": 10**1000},
        {"execution_timeout_seconds": True},
        {"queue_wait_seconds": float("nan")},
        {"queue_wait_seconds": 10**1000},
        {"queue_wait_seconds": -1},
        {"workspace": ""},
        {"task": "analysis.run"},
        {"requester": Principal("", frozenset())},
        {"requester": Principal("operator", frozenset({""}))},
        {"max_metadata_bytes": 100},
        {
            "task": replace(
                resolve_named_studio_task("analysis.run", authorized_route="/api/analysis/jobs"),
                owner="forged",
            )
        },
        {"authorized_route": "/api/models/scan/jobs"},
    ],
)
def test_invalid_content_refuses_before_replay_identity(change: dict[str, object]) -> None:
    """Malformed JSON, seeds and limits never produce a durable mutation key."""
    with pytest.raises(ValueError):
        derive_storage_admission_replay(**_content(**change))
