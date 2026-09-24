# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training metric stream

"""Tail training events through the durable manager after API proxy loss."""

from __future__ import annotations

import json
import queue
import time
from collections.abc import Iterator

from sc_neurocore.studio._training_control import _get_registered_job, _sync_proxy_job
from sc_neurocore.studio._training_events import (
    _event_from_platform_record,
    _read_live_training_events,
)
from sc_neurocore.studio.platform.jobs_ledger_schema import TERMINAL_STATUSES
from sc_neurocore.studio.platform.studio_job_service import StudioJobService


def _frame(event: dict[str, object]) -> str:
    """Encode one event as a complete Server-Sent Events frame."""
    return f"data: {json.dumps(event)}\n\n"


def _stream_metrics(
    job_id: str,
    job_manager: StudioJobService | None = None,
) -> Iterator[str]:
    """Yield live and retained events until a training job reaches a terminal state.

    A restarted API has no process-local proxy. Its stream reads the durable
    record and bounded event log on every poll, so a running or unknown job does
    not become a false disconnection after a single heartbeat. The manager's
    workspace and artifact readers retain their existing custody checks.
    """
    job = _get_registered_job(job_id)
    if job is None and job_manager is None:
        yield _frame({"event": "error", "data": {"message": "Job not found"}})
        return

    live_event_offset = 0
    live_event_buffer = ""
    while True:
        if job_manager is not None:
            try:
                record = job_manager.record(job_id)
            except KeyError:
                record = None
            if record is not None and record.kind != "training":
                record = None
            if record is None and job is None:
                yield _frame({"event": "error", "data": {"message": "Job not found"}})
                return
            if record is not None:
                if job is not None:
                    _sync_proxy_job(job, record.status, record.error, record.result)
                live_events, live_event_offset, live_event_buffer = _read_live_training_events(
                    job_manager,
                    job_id,
                    offset=live_event_offset,
                    buffer=live_event_buffer,
                )
                for event in live_events:
                    yield _frame(event)
                    if event.get("event") in ("completed", "stopped", "error"):
                        return
                if record.status in TERMINAL_STATUSES:
                    yield _frame(
                        _event_from_platform_record(record.status, record.error, record.result)
                    )
                    return
                if job is None:
                    yield _frame({"event": "heartbeat"})
                    time.sleep(1.0)
                    continue
        if job is None:
            yield _frame({"event": "error", "data": {"message": "Job not found"}})
            return
        try:
            event = job.metrics.get(timeout=1.0)
            yield _frame(event)
            if event["event"] in ("completed", "stopped", "error"):
                return
        except queue.Empty:
            if job.status in ("completed", "stopped", "failed", "interrupted"):
                return
            yield _frame({"event": "heartbeat"})
