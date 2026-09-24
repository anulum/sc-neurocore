# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sealed artefact read contract and client checks

"""The wire accepts only exact reads; the API verifies every received byte.

The misbehaving storage peer is a real socket endpoint answering with
well-formed frames that name another artefact or carry other bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_models import StudioJobArtifactUnavailable
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink
from sc_neurocore.studio.platform.storage_artifact_client import artifact_request, exchange_artifact
from sc_neurocore.studio.platform.storage_artifact_protocol import (
    StorageArtifactRequest,
    decode_artifact_request,
    decode_artifact_response,
    encode_artifact_message,
)
from sc_neurocore.studio.platform.storage_artifact_read import serve_artifact_read
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester

ADMIN = StorageRequester(principal_id="operator", roles=("studio.admin",))
JOB = "sj_" + "3" * 16
FRAME = 4096
PAYLOAD = b"weights"


def _request() -> StorageArtifactRequest:
    return artifact_request(
        "default", JOB, "weights.bin", route="/api/studio/training/weight-restore", requester=ADMIN
    )


def _reply(request_id: str, changes: dict[str, object] | None = None) -> bytes:
    body: dict[str, object] = {
        "schema_version": "studio.storage.artifact.v1",
        "operation": "artifact",
        "request_id": request_id,
        "status": "ok",
        "artifact": {
            "relative_path": "weights.bin",
            "size_bytes": len(PAYLOAD),
            "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        },
    }
    body.update(changes or {})
    return json.dumps(body).encode()


@pytest.mark.parametrize(
    "changes,payload,error",
    [
        ({}, b"weightz", StudioJobArtifactUnavailable),
        (
            {"artifact": {"relative_path": "other.bin", "size_bytes": 7, "sha256": "0" * 64}},
            None,
            ValueError,
        ),
        ({"request_id": "0" * 32}, None, ValueError),
    ],
    ids=["other-bytes", "other-artefact", "other-request"],
)
def test_the_api_refuses_what_it_did_not_ask_for(
    changes: dict[str, object], payload: bytes | None, error: type[Exception]
) -> None:
    """Bytes that differ from the declaration, or another artefact, never reach a caller."""
    client, service = socket.socketpair()

    def serve() -> None:
        with service:
            deadline = time.monotonic() + 10
            frame = read_verified_frame(
                service, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
            )
            sent = decode_artifact_request(frame, max_bytes=FRAME)
            for part in (_reply(sent.request_id, changes), payload):
                if part is not None:
                    write_verified_frame(
                        service, part, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
                    )

    thread = threading.Thread(target=serve)
    thread.start()
    with pytest.raises(error):
        exchange_artifact(
            client,
            _request(),
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10,
        )
    thread.join(timeout=10)


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"status": "ok", "artifact": None}, ValidationError),
        ({"status": "forbidden"}, ValidationError),
        ({"status": "forbidden", "artifact": None}, PermissionError),
        ({"status": "not_found", "artifact": None}, KeyError),
        ({"status": "unavailable", "artifact": None}, StudioJobArtifactUnavailable),
    ],
)
def test_replies_are_consistent_with_their_status(
    changes: dict[str, object], error: type[Exception]
) -> None:
    """Refusals carry no artefact and raise their own error."""
    request = _request()
    assert decode_artifact_response(_reply(request.request_id), request=request, max_bytes=FRAME)
    with pytest.raises(error):
        decode_artifact_response(
            _reply(request.request_id, changes), request=request, max_bytes=FRAME
        )


@pytest.mark.parametrize(
    "raw",
    [b"", b"\xff", b'{"a":1,"a":2}', b'{"a":NaN}', b"x" * (FRAME + 1)],
)
def test_malformed_or_foreign_route_requests_are_refused(raw: bytes) -> None:
    """Only exact frames naming a reviewed artefact route decode."""
    with pytest.raises(ValueError):
        decode_artifact_request(raw, max_bytes=FRAME)
    body = json.loads(encode_artifact_message(_request()))
    assert decode_artifact_request(
        json.dumps(body).encode(), max_bytes=FRAME
    ) == StorageArtifactRequest.model_validate_json(json.dumps(body), strict=True)
    body["route"] = "/api/studio/jobs"
    with pytest.raises(ValidationError):
        decode_artifact_request(json.dumps(body).encode(), max_bytes=FRAME)


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[StudioJobLedger]:
    authority = StudioJobLedger(root=tmp_path / "authority", supervisor="storage:1:1")
    try:
        yield authority
    finally:
        authority.close()


@pytest.mark.parametrize("workspace", ["default", ""])
def test_the_handler_reads_its_own_frame_and_refuses_no_workspace(
    ledger: StudioJobLedger, workspace: str
) -> None:
    """Called directly, the handler reads the request itself; a blank workspace refuses."""
    client, service = socket.socketpair()
    failures: list[BaseException] = []

    def serve() -> None:
        try:
            serve_artifact_read(
                service,
                ledger=ledger,
                gateway=PolicyGateway(InMemoryAuditSink()),
                workspace=workspace,
                expected_api_uid=os.getuid(),
                max_bytes=FRAME,
                deadline=time.monotonic() + 10,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            ledger.close()

    thread = threading.Thread(target=serve)
    thread.start()
    if workspace:
        with pytest.raises(KeyError):
            exchange_artifact(
                client,
                _request(),
                expected_service_uid=os.getuid(),
                max_bytes=FRAME,
                deadline=time.monotonic() + 10,
            )
    else:
        with client:
            client.settimeout(10)
            assert client.recv(1) == b""
    thread.join(timeout=10)
    assert [str(failure) for failure in failures] == (
        [] if workspace else ["storage workspace must be nonempty"]
    )
