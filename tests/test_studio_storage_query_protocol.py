# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage query wire contract

"""Only exact, correlated, view-consistent query messages decode."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.storage_query_protocol import (
    StorageQueryRequest,
    StorageQueryResponse,
    decode_query_request,
    decode_query_response,
    encode_query_message,
)

_REQUEST: dict[str, object] = {
    "schema_version": "studio.storage.query.v1",
    "operation": "query",
    "request_id": "a" * 32,
    "workspace": "default",
    "requester": {"principal_id": "operator", "roles": ["studio.admin"]},
    "view": "records",
    "limit": 10,
    "after": None,
}


def _request(**changes: object) -> StorageQueryRequest:
    return StorageQueryRequest.model_validate_json(json.dumps({**_REQUEST, **changes}), strict=True)


def _response(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "studio.storage.query.v1",
        "operation": "query",
        "request_id": "a" * 32,
        "view": "records",
        "status": "ok",
        "items": [{"job_id": "sj_" + "1" * 16}],
        "summary": None,
        "next_after": None,
    }
    body.update(changes)
    return body


def test_requests_round_trip() -> None:
    """A valid request survives encoding and strict decoding."""
    request = _request(after="sj_" + "b" * 16)
    assert decode_query_request(encode_query_message(request), max_bytes=4096) == request


@pytest.mark.parametrize(
    "changes",
    [
        {"view": "status", "after": "sj_" + "b" * 16},
        {"limit": 0},
        {"limit": 1001},
        {"after": "../x"},
        {"view": "all"},
        {"supervisor": "api:1:1"},
    ],
    ids=["status-cursor", "empty-page", "huge-page", "bad-cursor", "unknown-view", "extra-field"],
)
def test_malformed_requests_are_refused(changes: dict[str, object]) -> None:
    """Views, bounds and cursors follow the grammar; unknown fields never decode."""
    with pytest.raises(ValidationError):
        _request(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "forbidden"},
        {"status": "invalid_cursor", "items": [], "next_after": "sj_" + "1" * 16},
        {"status": "forbidden", "items": [], "summary": {"a": 1}},
        {"view": "status", "items": [], "summary": None},
        {"summary": {"a": 1}},
        {"view": "status", "summary": {"a": 1}},
        {"view": "status", "items": [], "summary": {"a": 1}, "next_after": "sj_" + "1" * 16},
    ],
    ids=[
        "refusal-with-items",
        "refusal-with-cursor",
        "refusal-with-summary",
        "status-without-summary",
        "page-with-summary",
        "status-with-items",
        "status-with-cursor",
    ],
)
def test_inconsistent_responses_are_refused(changes: dict[str, object]) -> None:
    """Items, summary and cursor belong to their view and to an answered query."""
    with pytest.raises(ValidationError):
        StorageQueryResponse.model_validate_json(json.dumps(_response(**changes)), strict=True)


@pytest.mark.parametrize(
    "raw",
    [b"", b"\xff", b"[]", b'{"a": 1, "a": 2}', b'{"a": NaN}', b"[" * 5000, b" " * 5000],
    ids=["empty", "not-utf8", "array", "duplicate", "nan", "deep", "oversized"],
)
def test_malformed_frames_are_refused(raw: bytes) -> None:
    """Empty, oversized, ambiguous, non-finite and non-object frames never decode."""
    with pytest.raises(ValueError):
        decode_query_request(raw, max_bytes=4096)


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"request_id": "b" * 32}, ValueError),
        ({"view": "purges"}, ValueError),
        ({"status": "forbidden", "items": []}, PermissionError),
        ({"status": "invalid_cursor", "items": []}, ValueError),
    ],
    ids=["other-request", "other-view", "forbidden", "invalid-cursor"],
)
def test_responses_must_answer_the_request(
    changes: dict[str, object], error: type[Exception]
) -> None:
    """Uncorrelated replies and refusals raise; an answered page decodes."""
    request = _request()
    answered = json.dumps(_response()).encode()
    assert decode_query_response(answered, request=request, max_bytes=4096).items
    with pytest.raises(error):
        decode_query_response(
            json.dumps(_response(**changes)).encode(), request=request, max_bytes=4096
        )
