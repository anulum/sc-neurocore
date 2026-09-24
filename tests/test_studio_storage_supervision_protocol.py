# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job supervision wire contract

"""Supervision frames carry no supervisor, payload or path and must correlate."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.storage_supervision_protocol import (
    SUPERVISION_SCHEMA_VERSION,
    StorageSupervisionResponse,
    SupervisionHeartbeatRequest,
    SupervisionStartRequest,
    decode_supervision_request,
    decode_supervision_response,
    encode_supervision_message,
)

_START = {
    "schema_version": "studio.storage.supervision.v1",
    "operation": "start",
    "request_id": "c" * 32,
    "workspace": "default",
    "job_id": "sj_" + "d" * 16,
    "worker": "host:4242:987654",
}


def _response(**fields: object) -> StorageSupervisionResponse:
    body: dict[str, object] = {
        "schema_version": SUPERVISION_SCHEMA_VERSION,
        "operation": "start",
        "request_id": "c" * 32,
        "job_id": "sj_" + "d" * 16,
        "outcome": "started",
        "reason": None,
    }
    body.update(fields)
    return StorageSupervisionResponse.model_validate(body, strict=True)


def test_start_and_heartbeat_round_trip_by_operation() -> None:
    """The operation selects the exact request type and survives encoding."""
    start = decode_supervision_request(json.dumps(_START).encode(), max_bytes=4096)
    assert isinstance(start, SupervisionStartRequest)
    assert decode_supervision_request(encode_supervision_message(start), max_bytes=4096) == start
    heartbeat = {key: value for key, value in _START.items() if key != "worker"}
    heartbeat["operation"] = "heartbeat"
    decoded = decode_supervision_request(json.dumps(heartbeat).encode(), max_bytes=4096)
    assert isinstance(decoded, SupervisionHeartbeatRequest)


@pytest.mark.parametrize(
    "changes",
    [
        {"worker": None},
        {"operation": "heartbeat"},
        {"operation": "complete"},
        {"supervisor": "host:1:1"},
        {"payload": {}},
        {"job_id": "sj_bad"},
        {"request_id": "C" * 32},
        {"workspace": ""},
        {"worker": "host:0:1"},
        {"worker": "host:1:01"},
        {"schema_version": "studio.storage.supervision.v2"},
    ],
)
def test_requests_refuse_every_other_shape(changes: dict[str, object]) -> None:
    """Supervisor claims, payloads, malformed identities and versions refuse."""
    body = {**_START, **changes}
    with pytest.raises(ValueError):
        decode_supervision_request(json.dumps(body).encode(), max_bytes=4096)


@pytest.mark.parametrize(
    "raw",
    [b"", b"\xff", b"{", b'{"operation":"start","operation":"start"}', b'{"a":NaN}', b" " * 65],
)
def test_malformed_frames_refuse(raw: bytes) -> None:
    """Empty, oversized, non-UTF-8, duplicate-name and non-finite frames refuse."""
    with pytest.raises(ValueError):
        decode_supervision_request(raw, max_bytes=64)


@pytest.mark.parametrize(
    "fields",
    [
        {"outcome": "started"},
        {"outcome": "cancelling"},
        {"outcome": "refused", "reason": "worker_conflict"},
        {"operation": "heartbeat", "outcome": "renewed"},
        {"operation": "heartbeat", "outcome": "cancelling"},
        {"operation": "heartbeat", "outcome": "refused", "reason": "not_owner"},
    ],
)
def test_consistent_responses_decode_against_their_request(fields: dict[str, object]) -> None:
    """Each outcome decodes when it answers the matching request."""
    response = _response(**fields)
    request: SupervisionStartRequest | SupervisionHeartbeatRequest
    if response.operation == "start":
        request = SupervisionStartRequest.model_validate(_START, strict=True)
    else:
        body = {key: value for key, value in _START.items() if key != "worker"}
        request = SupervisionHeartbeatRequest.model_validate(
            {**body, "operation": "heartbeat"}, strict=True
        )
    encoded = encode_supervision_message(response)
    assert decode_supervision_response(encoded, request=request, max_bytes=4096) == response


@pytest.mark.parametrize(
    "fields",
    [
        {"outcome": "refused"},
        {"outcome": "started", "reason": "not_found"},
        {"operation": "heartbeat", "outcome": "started"},
        {"outcome": "renewed"},
        {"operation": "heartbeat", "outcome": "refused", "reason": "worker_unverified"},
    ],
)
def test_inconsistent_responses_refuse(fields: dict[str, object]) -> None:
    """Outcome, reason and operation must agree."""
    with pytest.raises(ValidationError):
        _response(**fields)


@pytest.mark.parametrize(
    "fields", [{"request_id": "e" * 32}, {"job_id": "sj_" + "e" * 16}, {"operation": "heartbeat"}]
)
def test_uncorrelated_response_refuses(fields: dict[str, object]) -> None:
    """A reply for another request, job or operation is refused."""
    request = SupervisionStartRequest.model_validate(_START, strict=True)
    outcome = {"outcome": "renewed"} if fields.get("operation") == "heartbeat" else {}
    other = _response(**fields, **outcome)
    with pytest.raises(ValueError, match="does not answer"):
        decode_supervision_response(
            encode_supervision_message(other), request=request, max_bytes=4096
        )
