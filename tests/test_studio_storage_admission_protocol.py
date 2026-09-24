# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — named storage admission request tests

"""Exercise exact admission metadata over the verified Unix framing surface."""

from __future__ import annotations

import json
import os
import socket
import time

import pytest

from sc_neurocore.studio.platform.storage_admission_protocol import (
    decode_named_admission_request,
)
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame


def _request() -> dict[str, object]:
    return {
        "schema_version": "studio.storage.admission.v1",
        "operation": "admit_named",
        "request_id": "trace-one",
        "mutation_id": "retry-one",
        "workspace": "default",
        "requester": {"principal_id": "operator", "roles": ["studio.admin"]},
        "task_name": "analysis.run",
        "authorized_route": "/api/analysis/jobs",
        "payload": {"model": "lif", "params": {"tau": 4}},
        "seed_manifest": {"input/data.bin": 2},
        "execution_timeout_seconds": 30.0,
        "queue_wait_seconds": None,
        "admission": {"max_queue": 2},
        "training_config": None,
        "experiment_sha256": None,
    }


def test_verified_frame_decodes_named_request_without_derived_authority() -> None:
    """The metadata frame carries a name and manifest, not task code or digest."""
    reader, writer = socket.socketpair()
    with reader, writer:
        encoded = json.dumps(_request(), separators=(",", ":")).encode("utf-8")
        deadline = time.monotonic() + 1
        write_verified_frame(
            writer, encoded, expected_uid=os.getuid(), max_bytes=4096, deadline=deadline
        )
        frame = read_verified_frame(
            reader, expected_uid=os.getuid(), max_bytes=4096, deadline=deadline
        )
        request = decode_named_admission_request(frame, max_metadata_bytes=4096)
        assert request.task_name == "analysis.run"
        assert request.authorized_route == "/api/analysis/jobs"
        assert request.seed_manifest == {"input/data.bin": 2}
        assert request.requester is not None
        assert request.requester.roles == ("studio.admin",)
        assert not hasattr(request, "task_path")
        assert not hasattr(request, "payload_sha256")
        assert not hasattr(request, "supervisor")


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"studio.storage.admission.v1","schema_version":"old"}',
        b'{"payload":{"a":1,"a":2}}',
        b'{"payload":{"nested":{"a":1,"a":2}}}',
        b'{"execution_timeout_seconds":NaN}',
        b'{"admission":{"budget":Infinity}}',
        b"\xff",
        b"[]",
        b"{}",
        b"{",
    ],
)
def test_invalid_json_or_incomplete_envelope_refuses(payload: bytes) -> None:
    """Duplicate names and nonfinite constants never reach admission."""
    with pytest.raises(ValueError):
        decode_named_admission_request(payload, max_metadata_bytes=4096)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "studio.storage.admission.v0"},
        {"operation": "run_import_path"},
        {"task_path": "os:system"},
        {"payload_sha256": "a" * 64},
        {"supervisor": "pid:1:1"},
        {"seed_manifest": {"a": True}},
        {"seed_manifest": {"a": -1}},
        {"requester": {"principal_id": "operator", "roles": ["studio.admin"], "uid": 1}},
        {"requester": {"principal_id": "operator", "roles": [1]}},
        {"execution_timeout_seconds": float("nan")},
    ],
)
def test_unknown_or_wrong_typed_metadata_refuses(change: dict[str, object]) -> None:
    """The wire contract cannot select raw code or fabricate authority fields."""
    request = _request()
    request.update(change)
    encoded = json.dumps(request, allow_nan=True).encode("utf-8")
    with pytest.raises(ValueError):
        decode_named_admission_request(encoded, max_metadata_bytes=4096)


@pytest.mark.parametrize("payload,limit", [(b"", 8), (b"x", 0), (b"xxxxxxxxx", 8)])
def test_metadata_byte_ceiling_refuses_before_parse(payload: bytes, limit: int) -> None:
    """Empty, oversized and invalid-budget frames have no decoding path."""
    with pytest.raises(ValueError, match="byte limit"):
        decode_named_admission_request(payload, max_metadata_bytes=limit)
