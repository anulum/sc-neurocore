# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete storage view content acceptance

"""Exercise real response frames against the public record and query clients."""

import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_query_client import QueryReader
from sc_neurocore.studio.platform.storage_record_client import read_storage_record
from sc_neurocore.studio.platform.storage_record_protocol import (
    StorageRecordRequest,
    StorageRequester,
)
from sc_neurocore.studio.platform.storage_view_content import StorageViewContent, send_view_content
from tests.studio_storage_generation_support import Authority
from tests.test_studio_training_config_storage import _configuration, _create

FRAME = 512


@pytest.fixture
def record(tmp_path: Path) -> StudioJobRecord:
    """Read a real immutable row containing a thousand-sample SHD declaration."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    try:
        _create(
            ledger, _configuration(tmp_path / "recordings", count=1000), job_id="sj_" + "a" * 16
        )
        return ledger.record("sj_" + "a" * 16)
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "fault",
    [
        "valid",
        "sha",
        "chunk",
        "correlation",
        "schema",
        "limit",
        "snapshot",
        "json",
        "duplicate",
        "nan",
        "array",
        "sender_limit",
        "mutable",
    ],
)
def test_public_record_reader_accepts_full_content_and_refuses_bad_authority_frames(
    record: StudioJobRecord,
    fault: str,
) -> None:
    """Full records survive exact frames; corrupt peers cannot bypass snapshot checks."""
    request = StorageRecordRequest(
        schema_version="studio.storage.record.v3",
        operation="record",
        request_id="view-read",
        job_id=record.job_id,
        workspace=record.workspace,
        requester=StorageRequester(principal_id="operator", roles=("studio.admin",)),
    )
    response: dict[str, object] = {
        "schema_version": "studio.storage.record.v3",
        "request_id": request.request_id,
        "status": "ok",
        "record": record.to_public_dict(),
    }
    if fault == "snapshot":
        response["record"] = {**record.to_public_dict(), "job_id": "another-job"}
    payload = json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    header = StorageViewContent(
        schema_version="studio.storage.view-content.v1",
        content_schema="studio.storage.record.v3",
        request_id=request.request_id,
        content_bytes=len(payload),
        content_sha256=hashlib.sha256(payload).hexdigest(),
    )
    if fault == "correlation":
        header = header.model_copy(update={"request_id": "another-request"})
    if fault == "schema":
        header = header.model_copy(update={"content_schema": "studio.storage.query.v2"})
    client, service = socket.socketpair()
    failures: list[BaseException] = []

    def serve() -> None:
        """Send actual snapshot bytes or deliberately corrupt frames over a real peer."""
        try:
            with service:
                deadline = time.monotonic() + 10
                read_verified_frame(
                    service, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
                )
                if fault in {"valid", "sender_limit", "mutable"}:
                    content = cast(bytes, bytearray(payload)) if fault == "mutable" else payload
                    send_view_content(
                        service,
                        content,
                        content_schema="studio.storage.record.v3",
                        request_id=request.request_id,
                        expected_uid=os.getuid(),
                        frame_max_bytes=FRAME,
                        deadline=deadline,
                        max_content_bytes=FRAME if fault == "sender_limit" else None,
                    )
                    return
                invalid = {
                    "json": b"{bad",
                    "duplicate": b'{"schema_version":"x","schema_version":"x"}',
                    "nan": b'{"value":NaN}',
                    "array": b"[]",
                }
                if fault in invalid:
                    write_verified_frame(
                        service,
                        invalid[fault],
                        expected_uid=os.getuid(),
                        max_bytes=FRAME,
                        deadline=deadline,
                    )
                    return
                write_verified_frame(
                    service,
                    header.model_dump_json().encode(),
                    expected_uid=os.getuid(),
                    max_bytes=FRAME,
                    deadline=deadline,
                )
                if fault in {"correlation", "schema", "limit"}:
                    service.settimeout(10)
                    assert service.recv(1) == b""
                    return
                if fault == "chunk":
                    write_verified_frame(
                        service,
                        payload[:1],
                        expected_uid=os.getuid(),
                        max_bytes=FRAME,
                        deadline=deadline,
                    )
                    return
                content = b"!" + payload[1:] if fault == "sha" else payload
                for offset in range(0, len(content), FRAME):
                    write_verified_frame(
                        service,
                        content[offset : offset + FRAME],
                        expected_uid=os.getuid(),
                        max_bytes=FRAME,
                        deadline=deadline,
                    )
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        arguments = {"max_content_bytes": 1024} if fault == "limit" else {}
        if fault == "valid":
            restored = read_storage_record(
                client,
                request=request,
                expected_service_uid=os.getuid(),
                max_bytes=FRAME,
                deadline=time.monotonic() + 10,
                **arguments,
            )
            assert restored == record
        else:
            expected = EOFError if fault in {"sender_limit", "mutable"} else ValueError
            with pytest.raises(expected):
                read_storage_record(
                    client,
                    request=request,
                    expected_service_uid=os.getuid(),
                    max_bytes=FRAME,
                    deadline=time.monotonic() + 10,
                    **arguments,
                )
    finally:
        client.close()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert [type(error) for error in failures] == (
        [ValueError] if fault in {"sender_limit", "mutable"} else []
    )


@pytest.mark.parametrize(
    "frame,content",
    [(0, None), (True, None), (0x100000000, None), (512, 0), (512, True), (512, 0x100000000)],
)
def test_query_reader_rejects_bad_trusted_limits_before_connecting(
    tmp_path: Path, frame: int, content: int | None
) -> None:
    """Invalid operator limits never open an authority connection or access rows."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    authority = Authority(ledger)
    try:
        with pytest.raises(ValueError, match="limit"):
            QueryReader(
                authority.connect,
                workspace="default",
                storage_uid=os.getuid(),
                max_bytes=frame,
                max_content_bytes=content,
                timeout_seconds=1.0,
            )
        assert authority.seen == []
    finally:
        authority.join()
        ledger.close()


def test_inline_response_obeys_independent_content_budget(record: StudioJobRecord) -> None:
    """A large frame ceiling cannot override a smaller receiver content budget."""
    request = StorageRecordRequest(
        schema_version="studio.storage.record.v3",
        operation="record",
        request_id="inline",
        job_id=record.job_id,
        workspace=record.workspace,
        requester=None,
    )
    body = json.dumps(
        {
            "schema_version": "studio.storage.record.v3",
            "request_id": "inline",
            "status": "ok",
            "record": record.to_public_dict(),
        }
    ).encode()
    client, service = socket.socketpair()
    with service:
        write_verified_frame(
            service,
            body,
            expected_uid=os.getuid(),
            max_bytes=len(body) + 1024,
            deadline=time.monotonic() + 10,
        )
        with pytest.raises(ValueError, match="content limit"):
            read_storage_record(
                client,
                request=request,
                expected_service_uid=os.getuid(),
                max_bytes=len(body) + 1024,
                max_content_bytes=1024,
                deadline=time.monotonic() + 10,
            )
