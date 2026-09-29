# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lost complete event purge reply

"""Lose a real committed purge reply without replacing its authority handler."""

from dataclasses import replace
import os
from pathlib import Path
import socket
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_purge_protocol import (
    PURGE_SCHEMA_VERSION,
    decode_purge_request,
    decode_purge_response,
)
from sc_neurocore.studio.platform.storage_view_content import read_view_content, view_content_limit
from tests.studio_storage_generation_runs import api_runtime
from tests.studio_storage_generation_runs import authority as authority
from tests.studio_storage_generation_runs import ledger as ledger
from tests.studio_storage_generation_support import Authority, FRAME
from tests.test_studio_storage_event_stop import exercise_event_stop
from tests.test_studio_storage_event_training import base as base
from tests.test_studio_storage_event_training import launcher as launcher
from tests.test_studio_storage_event_training import event_config as event_config
from tests.test_studio_storage_isolated_jobs import _configuration
from tests.test_studio_storage_isolated_jobs import manager as manager
from tests.test_studio_storage_isolated_jobs import request_on


def exercise_lost_event_purge_reply(
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    authority: Authority,
    event_config: dict[str, object],
    base: Path,
) -> StudioJobRecord:
    """Discard a complete authority reply and recover by public record lookup.

    Parameters
    ----------
    manager, ledger, authority :
        Actual API facade, SQLite authority and production handler connector.
    event_config : dict
        Complete resolved input for the actual worker stopped before purge.
    base : Path
        Trusted API boundary paths of this owned worker generation.

    Returns
    -------
    StudioJobRecord
        Full pre-purge record observed at the relay before deliberately losing it.
    """
    finished = exercise_event_stop(manager, ledger, authority, event_config, "cancel")
    observed: list[StudioJobRecord] = []
    failures: list[BaseException] = []
    threads: list[threading.Thread] = []

    def discard(channel: socket.socket) -> None:
        try:
            with channel, authority.connect() as downstream:
                deadline = time.monotonic() + 10
                raw = read_verified_frame(
                    channel, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
                )
                sent = decode_purge_request(raw, max_bytes=FRAME)
                write_verified_frame(
                    downstream, raw, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
                )
                content = read_view_content(
                    downstream,
                    content_schema=PURGE_SCHEMA_VERSION,
                    request_id=sent.request_id,
                    expected_uid=os.getuid(),
                    frame_max_bytes=FRAME,
                    deadline=deadline,
                )
                reply = decode_purge_response(
                    content, request=sent, max_bytes=view_content_limit(FRAME)
                )
                observed.append(decode_job_snapshot(reply.record or {}))
        except BaseException as exc:
            failures.append(exc)

    def connect() -> socket.socket:
        client, relay = socket.socketpair()
        thread = threading.Thread(target=discard, args=(relay,))
        threads.append(thread)
        thread.start()
        return client

    runtime = replace(api_runtime(base, authority), connect=connect)
    losing = IsolatedJobManager(
        runtime,
        _configuration(base),
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=120.0,
    )
    try:
        with request_on("/api/studio/audit/quarantine/archive/purge"), pytest.raises(EOFError):
            losing.purge_terminal_record(finished.job_id)
    finally:
        for thread in threads:
            thread.join(timeout=15)
            assert not thread.is_alive()
        runtime.live.close()
    assert failures == []
    assert observed == [finished]
    with request_on("/api/studio/jobs", "GET"):
        with pytest.raises(KeyError):
            manager.record(finished.job_id)
        assert manager.list_records() == ()
    with request_on("/api/studio/audit/quarantine/archive/purge"), pytest.raises(KeyError):
        manager.purge_terminal_record(finished.job_id)
    assert ledger.list_records() == ()
    assert not (ledger.path.parent / finished.job_id).exists()
    assert not (ledger.path.parent / f".purge-{finished.job_id}").exists()
    assert ledger.connection().execute("SELECT COUNT(*) FROM job_purges").fetchone()[0] == 0
    return finished


def test_lost_event_purge_reply_is_resolved_without_recreating_the_archive(
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    authority: Authority,
    event_config: dict[str, object],
    base: Path,
) -> None:
    """A genuinely lost full reply leaves one committed purge, confirmed by reads."""
    exercise_lost_event_purge_reply(manager, ledger, authority, event_config, base)
