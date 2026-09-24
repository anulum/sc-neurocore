# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API reader of storage authority views

"""The API refuses pages that would make it loop or read another workspace.

The storage peer here is a real socket endpoint that answers with crafted
but well-formed frames, standing for an authority that misbehaves.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
import os
from pathlib import Path
import socket
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_models import StudioJobExecutionModel
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_query_protocol import (
    StorageQueryRequest,
    StorageQueryResponse,
    decode_query_request,
    encode_query_message,
)
from sc_neurocore.studio.platform.storage_query_client import (
    QueryReader,
)
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester
from tests.studio_storage_generation_support import FRAME
from tests.studio_storage_supervision_support import Clock

ADMIN = StorageRequester(principal_id="operator", roles=("studio.admin",))
VIEWER = StorageRequester(principal_id="viewer", roles=("studio.viewer",))


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[StudioJobLedger]:
    """Authority ledger with a frozen clock, so creation times tie."""
    authority = StudioJobLedger(
        root=tmp_path / "authority", supervisor="storage:1:1", clock=Clock()
    )
    try:
        yield authority
    finally:
        authority.close()


def _admit(
    ledger: StudioJobLedger,
    count: int,
    *,
    workspace: str = "default",
    model: StudioJobExecutionModel = "process",
    first: int = 0,
) -> list[str]:
    """Admit ``count`` jobs whose IDs sort against their creation order."""
    admission = SharedJobAdmission(ledger, max_concurrent=64, max_queued=0)
    ids = [f"sj_{0xFFFF - first - index:016x}" for index in range(count)]
    for job_id in ids:
        admission.admit(
            job_id=job_id,
            kind="analysis",
            actor="operator",
            workspace=workspace,
            request_id=None,
            idempotency_key=None,
            experiment_sha256=None,
            admission=None,
            execution_model=model,
            supervisor=supervisor_identity(),
        )
    return ids


def _peer(answer: Callable[[StorageQueryRequest], bytes]) -> Callable[[], socket.socket]:
    """A storage peer that answers each query with ``answer``'s frame."""

    def connect() -> socket.socket:
        client, service = socket.socketpair()

        def serve() -> None:
            with service:
                deadline = time.monotonic() + 10
                frame = read_verified_frame(
                    service, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
                )
                request = decode_query_request(frame, max_bytes=FRAME)
                write_verified_frame(
                    service,
                    answer(request),
                    expected_uid=os.getuid(),
                    max_bytes=FRAME,
                    deadline=deadline,
                )

        threading.Thread(target=serve, daemon=True).start()
        return client

    return connect


@pytest.mark.parametrize("fault", ["other-workspace", "stalled-cursor", "cursor-without-items"])
def test_a_misbehaving_authority_cannot_make_the_api_loop_or_leak(
    ledger: StudioJobLedger, fault: str
) -> None:
    """Pages from another workspace or with a cursor that does not advance are refused."""
    (job_id,) = _admit(
        ledger, 1, workspace="elsewhere" if fault == "other-workspace" else "default"
    )
    item = json.loads(json.dumps(ledger.record(job_id).to_public_dict()))

    def answer(request: StorageQueryRequest) -> bytes:
        items = [] if fault == "cursor-without-items" else [item]
        cursor = None if fault == "other-workspace" else "sj_" + "0" * 16
        return encode_query_message(
            StorageQueryResponse(
                schema_version="studio.storage.query.v1",
                operation="query",
                request_id=request.request_id,
                view=request.view,
                status="ok",
                items=tuple(items),
                summary=None,
                next_after=cursor,
            )
        )

    reader = QueryReader(
        _peer(answer),
        workspace="default",
        storage_uid=os.getuid(),
        max_bytes=FRAME,
        timeout_seconds=10,
    )
    with pytest.raises(ValueError, match="another workspace|does not advance"):
        reader.records(ADMIN)
