# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training event and status codecs

"""Encode persisted training events and adapt platform terminal states."""

from __future__ import annotations

import json
import math
import time
from typing import Any, cast

from sc_neurocore.studio.platform.jobs import StudioJobManager, StudioJobStatus

TRAINING_EVENT_LOG_ARTIFACT_PATH = "training/events.jsonl"

_TRAINING_STATUS_BY_PLATFORM_STATUS: dict[StudioJobStatus, str] = {
    "pending": "running",
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelling": "stopped",
    "cancelled": "stopped",
    "timed_out": "stopped",
}


def _training_status_from_platform_status(platform_status: StudioJobStatus) -> str:
    """Map a platform job status into the Training Monitor vocabulary."""
    return _TRAINING_STATUS_BY_PLATFORM_STATUS[platform_status]


def _event_from_platform_record(
    platform_status: StudioJobStatus,
    platform_error: str | None,
    platform_result: dict[str, object] | None,
) -> dict[str, Any]:
    """Return an SSE event synthesized from a platform job record."""
    training_status = _training_status_from_platform_status(platform_status)
    if training_status == "completed":
        final_metrics = (platform_result or {}).get("final_metrics")
        return {
            "data": final_metrics if isinstance(final_metrics, dict) else {},
            "event": "completed",
            "timestamp": time.time(),
        }
    if training_status == "failed":
        return {
            "data": {"message": platform_error or "Training failed."},
            "event": "error",
            "timestamp": time.time(),
        }
    if training_status == "stopped":
        return {
            "data": {"message": platform_error or "Training stopped."},
            "event": "stopped",
            "timestamp": time.time(),
        }
    return {"event": "heartbeat"}


def _json_event_payload(payload: dict[str, object]) -> dict[str, object]:
    """Return a JSON-compatible copy of a Training Monitor SSE event."""
    converted = {str(key): _json_compatible(value) for key, value in payload.items()}
    return cast(dict[str, object], converted)


def _json_compatible(value: object) -> Any:
    """Return ``value`` converted to JSON-compatible containers."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_compatible(item) for item in value]
    return str(value)


def _read_live_training_events(
    job_manager: StudioJobManager,
    job_id: str,
    *,
    offset: int,
    buffer: str,
) -> tuple[list[dict[str, object]], int, str]:
    """Read complete JSONL Training Monitor events appended by a worker."""
    payload, new_offset = job_manager.read_live_artifact_bytes(
        job_id,
        TRAINING_EVENT_LOG_ARTIFACT_PATH,
        offset=offset,
    )
    if not payload:
        return [], new_offset, buffer
    text = buffer + payload.decode("utf-8")
    lines = text.splitlines(keepends=True)
    next_buffer = ""
    if lines and not lines[-1].endswith("\n"):
        next_buffer = lines.pop()
    events: list[dict[str, object]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(cast(dict[str, object], dict(event)))
    return events, new_offset, next_buffer
