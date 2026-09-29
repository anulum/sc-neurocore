# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job ledger row and column codecs

"""Rebuild public records from stored rows and refuse malformed stored values.

A stored column that does not decode is reported as ledger corruption rather
than silently defaulted, so a damaged record is never presented as valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, cast

from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifact,
    StudioJobExecutionModel,
    StudioJobRecord,
    StudioJobStatus,
)


class StudioJobLedgerCorrupt(RuntimeError):
    """Raised when the ledger file cannot be read as a Studio job ledger."""


def json_or_none(value: str | None) -> Any:
    """Decode one stored JSON column, or ``None``."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise StudioJobLedgerCorrupt(f"stored JSON column is not valid JSON: {exc}") from exc


def artifacts_from_json(value: str) -> tuple[StudioJobArtifact, ...]:
    """Rebuild an artifact manifest, refusing a malformed one."""
    payload = json_or_none(value)
    if not isinstance(payload, list):
        raise StudioJobLedgerCorrupt("the artifact manifest is not a list")
    artifacts: list[StudioJobArtifact] = []
    for entry in payload:
        if not isinstance(entry, Mapping):
            raise StudioJobLedgerCorrupt("an artifact manifest entry is not an object")
        try:
            artifacts.append(
                StudioJobArtifact(
                    relative_path=str(entry["relative_path"]),
                    size_bytes=int(entry["size_bytes"]),
                    sha256=str(entry["sha256"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StudioJobLedgerCorrupt(f"an artifact manifest entry is malformed: {exc}") from exc
    return tuple(artifacts)


def artifacts_to_json(artifacts: Sequence[StudioJobArtifact]) -> str:
    """Serialise an artifact manifest deterministically."""
    return json.dumps([artifact.to_public_dict() for artifact in artifacts], sort_keys=True)


def record_from_row(row: sqlite3.Row) -> StudioJobRecord:
    """Rebuild one immutable public record from its stored row."""
    return StudioJobRecord(
        job_id=str(row["job_id"]),
        kind=str(row["kind"]),
        owner=str(row["actor"]),
        request_id=None if row["request_id"] is None else str(row["request_id"]),
        status=cast(StudioJobStatus, str(row["status"])),
        execution_model=cast(StudioJobExecutionModel, str(row["execution_model"])),
        created_at_utc=str(row["created_at_utc"]),
        started_at_utc=None if row["started_at_utc"] is None else str(row["started_at_utc"]),
        finished_at_utc=None if row["finished_at_utc"] is None else str(row["finished_at_utc"]),
        error=None if row["error"] is None else str(row["error"]),
        result=json_or_none(row["result"]),
        artifacts=artifacts_from_json(str(row["artifacts"])),
        workspace=str(row["workspace"]),
        idempotency_key=None if row["idempotency_key"] is None else str(row["idempotency_key"]),
        experiment_sha256=(
            None if row["experiment_sha256"] is None else str(row["experiment_sha256"])
        ),
        admission=json_or_none(row["admission"]) or {},
        training_config=training_config_from_json(
            row["training_config"],
            kind=str(row["kind"]),
            event_data=row["training_event_data"],
        ),
        lease_owner=None if row["lease_owner"] is None else str(row["lease_owner"]),
        lease_expires_at_utc=(
            None if row["lease_expires_at_utc"] is None else str(row["lease_expires_at_utc"])
        ),
        heartbeat_at_utc=None if row["heartbeat_at_utc"] is None else str(row["heartbeat_at_utc"]),
    )


#: Validated snapshots keyed by their exact stored bytes and the admission limit.
#: Validation is a pure function of those, and an event declaration can hold a
#: manifest of thousands of recordings: re-validating it on every read made one
#: status read take about a quarter of a second, so a polling wait held a core.
_VALIDATED: OrderedDict[tuple[str, str | None, str | None], str] = OrderedDict()
_VALIDATED_LOCK = threading.Lock()
_VALIDATED_ENTRIES = 64


def training_config_from_json(
    value: str | None, *, kind: str, event_data: str | None = None
) -> dict[str, object] | None:
    """Decode a validated training snapshot, preserving absent legacy values.

    A snapshot whose exact stored bytes were validated before, under the same
    event-input admission limit, is returned as a fresh copy of that result
    without validating again; any other snapshot is validated in full.
    """
    if event_data is not None and not isinstance(event_data, str):
        raise StudioJobLedgerCorrupt("stored event input data is not text")
    if value is None:
        if event_data is not None:
            raise StudioJobLedgerCorrupt("event input data has no training configuration")
        return None
    if kind != "training":
        raise StudioJobLedgerCorrupt("non-training job stores a training configuration")
    if len(value.encode("utf-8")) > 4096:
        raise StudioJobLedgerCorrupt("stored training configuration exceeds 4096 bytes")
    payload = json_or_none(value)
    if not isinstance(payload, dict):
        raise StudioJobLedgerCorrupt("stored training configuration is not an object")
    from sc_neurocore.studio.event_training_budget import EVENT_INPUT_LIMIT_ENV

    key = (
        value,
        None if event_data is None else hashlib.sha256(event_data.encode("utf-8")).hexdigest(),
        os.environ.get(EVENT_INPUT_LIMIT_ENV),
    )
    with _VALIDATED_LOCK:
        cached = _VALIDATED.get(key)
        if cached is not None:
            _VALIDATED.move_to_end(key)
    if cached is not None:
        return cast(dict[str, object], json.loads(cached))
    resolved = _validated_training_config(value, payload, event_data)
    with _VALIDATED_LOCK:
        _VALIDATED[key] = json.dumps(resolved, sort_keys=True)
        while len(_VALIDATED) > _VALIDATED_ENTRIES:
            _VALIDATED.popitem(last=False)
    return resolved


def _validated_training_config(
    value: str, payload: dict[str, Any], event_data: str | None
) -> dict[str, object]:
    """Resolve a stored snapshot and refuse it unless it is canonical."""
    from sc_neurocore.studio.training_contract import resolve_training_config

    try:
        from sc_neurocore.studio.platform.training_config_storage import restore_training_config

        restored = restore_training_config(payload, event_data)
        resolved = resolve_training_config(restored).to_public_dict()
    except ValueError as exc:
        raise StudioJobLedgerCorrupt(f"stored training configuration is invalid: {exc}") from exc
    from sc_neurocore.studio.platform.training_config_storage import prepare_training_config

    try:
        canonical, canonical_event = prepare_training_config(resolved)
    except ValueError as exc:
        raise StudioJobLedgerCorrupt("stored training configuration exceeds its bounds") from exc
    # Legacy inline event declarations remain readable without a migration rewrite.
    if event_data is None:
        canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    if canonical != value or (event_data is not None and canonical_event != event_data):
        raise StudioJobLedgerCorrupt("stored training configuration is not canonical")
    return resolved


__all__ = [
    "StudioJobLedgerCorrupt",
    "artifacts_from_json",
    "artifacts_to_json",
    "json_or_none",
    "record_from_row",
    "training_config_from_json",
]
