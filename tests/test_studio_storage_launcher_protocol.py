# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — worker launcher wire contract

"""The launcher wire accepts only exact job generations and bounded observations."""

from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.storage_launcher_protocol import (
    LAUNCHER_MESSAGE_MAX_BYTES,
    LauncherRequest,
    LauncherResponse,
    decode_launcher_request,
    decode_launcher_response,
    encode_launcher_request,
    encode_launcher_response,
)

_REQUEST = {
    "version": "studio.launcher.v1",
    "request_id": "1" * 32,
    "operation": "launch",
    "job_id": "sj_" + "a" * 16,
    "generation": "b" * 32,
}


def _request(**changes: object) -> LauncherRequest:
    return LauncherRequest.model_validate({**_REQUEST, **changes}, strict=True)


def _response(request: LauncherRequest, **fields: object) -> LauncherResponse:
    body: dict[str, object] = {
        "version": "studio.launcher.v1",
        "request_id": request.request_id,
        "operation": request.operation,
        "job_id": request.job_id,
        "generation": request.generation,
        "state": "running",
        "pid": 4242,
        "start_token": "987654",
        "reason": None,
    }
    body.update(fields)
    if body["state"] == "stopped":
        body.setdefault("exit_status", 0)
    return LauncherResponse.model_validate(body, strict=True)


def test_request_round_trip_is_canonical() -> None:
    """Encoding is sorted compact JSON that decodes to the same request."""
    request = _request()
    encoded = encode_launcher_request(request)
    assert encoded == json.dumps(_REQUEST, sort_keys=True, separators=(",", ":")).encode()
    assert decode_launcher_request(encoded) == request


@pytest.mark.parametrize(
    "mutation",
    [
        {"version": "studio.launcher.v2"},
        {"operation": "exec"},
        {"job_id": "sj_" + "A" * 16},
        {"job_id": "../sj_" + "a" * 16},
        {"generation": "b" * 31},
        {"request_id": 7},
        {"command": "/bin/sh"},
        {"unit": "scn-studio-worker@x.service"},
    ],
)
def test_request_refuses_any_other_shape(mutation: dict[str, object]) -> None:
    """Commands, unit names, other versions and malformed identifiers are refused."""
    body = json.dumps({**_REQUEST, **mutation}).encode()
    with pytest.raises(ValueError):
        decode_launcher_request(body)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\xff",
        b"[]",
        b'{"version":"studio.launcher.v1","version":"studio.launcher.v1"}',
        b'{"pid": NaN}',
        b"{" * 5000,
        b" " * (LAUNCHER_MESSAGE_MAX_BYTES + 1),
    ],
)
def test_request_refuses_malformed_frames(raw: bytes) -> None:
    """Empty, oversized, non-UTF-8, duplicate-name and non-finite frames refuse."""
    with pytest.raises(ValueError):
        decode_launcher_request(raw)


def test_decoder_refuses_non_bytes() -> None:
    """Only a received byte frame is decodable."""
    with pytest.raises(ValueError, match="byte limit"):
        decode_launcher_request(cast(bytes, "{}"))


@pytest.mark.parametrize(
    "operation,fields",
    [
        ("launch", {}),
        ("launch", {"state": "stopped"}),
        ("launch", {"state": "refused", "reason": "capacity", "pid": None, "start_token": None}),
        ("launch", {"state": "refused", "reason": "conflict", "pid": None, "start_token": None}),
        ("status", {}),
        ("status", {"state": "absent", "pid": None, "start_token": None}),
        ("stop", {"state": "stopped"}),
        ("stop", {"state": "stopped", "pid": None, "start_token": None}),
        ("stop", {"state": "absent", "pid": None, "start_token": None}),
        ("stop", {"state": "refused", "reason": "survivors"}),
        ("status", {"state": "stopped", "exit_status": -9}),
        ("status", {"state": "stopped", "exit_status": 255}),
    ],
)
def test_valid_response_states_round_trip(operation: str, fields: dict[str, object]) -> None:
    """Every consistent state decodes against its own request."""
    request = _request(operation=operation)
    response = _response(request, **fields)
    assert decode_launcher_response(encode_launcher_response(response), request=request) == response


@pytest.mark.parametrize(
    "operation,fields",
    [
        ("launch", {"pid": None}),
        ("launch", {"start_token": None}),
        ("launch", {"reason": "capacity"}),
        ("launch", {"state": "absent", "pid": None, "start_token": None}),
        ("status", {"state": "absent"}),
        ("status", {"state": "absent", "pid": None, "start_token": None, "reason": "spool"}),
        ("stop", {}),
        ("stop", {"state": "stopped", "reason": "survivors"}),
        ("stop", {"state": "refused", "pid": None, "start_token": None}),
        ("stop", {"state": "refused", "reason": "survivors", "pid": None, "start_token": None}),
        ("launch", {"state": "refused", "reason": "survivors"}),
        ("launch", {"state": "refused", "reason": "capacity"}),
        ("launch", {"start_token": "0123"}),
        ("launch", {"pid": 0}),
        ("launch", {"pid": True}),
        ("launch", {"exit_status": 0}),
        ("stop", {"state": "absent", "pid": None, "start_token": None, "exit_status": 1}),
        ("status", {"state": "stopped", "exit_status": None}),
        ("status", {"state": "stopped", "exit_status": 256}),
        ("status", {"state": "stopped", "exit_status": -65}),
        ("status", {"state": "stopped", "exit_status": False}),
    ],
)
def test_inconsistent_response_states_refuse(operation: str, fields: dict[str, object]) -> None:
    """Identity, reason and operation must agree with the reported state."""
    request = _request(operation=operation)
    with pytest.raises(ValidationError):
        _response(request, **fields)


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "2" * 32),
        ("operation", "status"),
        ("job_id", "sj_" + "c" * 16),
        ("generation", "d" * 32),
    ],
)
def test_uncorrelated_response_refuses(field: str, value: str) -> None:
    """A reply for another request, operation, job or generation is refused."""
    request = _request()
    other = _response(_request(**{field: value}))
    with pytest.raises(ValueError, match="does not answer"):
        decode_launcher_response(encode_launcher_response(other), request=request)


def test_response_refuses_unknown_fields_and_output() -> None:
    """A launcher cannot smuggle output or file contents through the reply."""
    request = _request()
    body = json.loads(encode_launcher_response(_response(request)))
    body["stdout"] = "secret"
    with pytest.raises(ValueError):
        decode_launcher_response(json.dumps(body).encode(), request=request)


def test_longest_valid_messages_fit_the_frame_bound() -> None:
    """Schema bounds keep every encodable message inside the decoder limit."""
    request = _request(operation="status")
    response = _response(
        request,
        state="refused",
        reason="unavailable",
        pid=None,
        start_token=None,
    )
    widest = _response(
        _request(operation="stop"),
        state="refused",
        reason="survivors",
        pid=0x3FFFFFFF,
        start_token="9" * 20,
    )
    stopped = _response(
        _request(operation="status"),
        state="stopped",
        pid=0x3FFFFFFF,
        start_token="9" * 20,
        exit_status=-64,
    )
    for encoded in (
        encode_launcher_request(request),
        encode_launcher_response(response),
        encode_launcher_response(widest),
        encode_launcher_response(stopped),
    ):
        assert len(encoded) < LAUNCHER_MESSAGE_MAX_BYTES // 2
