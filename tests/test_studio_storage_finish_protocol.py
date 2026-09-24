# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job finish wire contract

"""The finish wire accepts exactly one bounded, correlated shape."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.storage_finish_protocol import (
    FINISH_SCHEMA_VERSION,
    FinishArtifact,
    StorageFinishRequest,
    StorageFinishResponse,
    decode_finish_request,
    decode_finish_response,
    encode_finish_message,
    validate_finish_manifest,
)

JOB = "sj_" + "4" * 16
DIGEST = "c" * 64


def _request(**changes: object) -> StorageFinishRequest:
    fields: dict[str, object] = {
        "schema_version": FINISH_SCHEMA_VERSION,
        "operation": "finish",
        "request_id": "d" * 32,
        "workspace": "default",
        "job_id": JOB,
        "outcome": "completed",
        "result": {"answer": 42, "nested": [1.5, None, "x"]},
        "error": None,
        "worker_reaped": True,
        "artifacts": [
            {"relative_path": "reports/summary.json", "size_bytes": 3, "sha256": DIGEST},
            {"relative_path": "weights.bin", "size_bytes": 0, "sha256": DIGEST},
        ],
    }
    fields.update(changes)
    return StorageFinishRequest.model_validate_json(json.dumps(fields), strict=True)


def test_request_round_trips_through_the_wire() -> None:
    """A valid request survives encoding and strict decoding unchanged."""
    request = _request()
    assert decode_finish_request(encode_finish_message(request), max_bytes=4096) == request


@pytest.mark.parametrize(
    "path",
    ["../escape", "/absolute", "a//b", "a/./b", "a/", "line\nbreak", "", "x" * 513],
)
def test_artefact_paths_stay_canonical_inside_the_job(path: str) -> None:
    """Traversal, absolute, non-canonical, unprintable or oversized paths refuse."""
    with pytest.raises(ValidationError):
        FinishArtifact(relative_path=path, size_bytes=1, sha256=DIGEST)


@pytest.mark.parametrize(
    "changes",
    [
        {"outcome": "failed"},
        {"outcome": "failed", "result": None},
        {"outcome": "timed_out", "result": None},
        {"outcome": "cancelled", "error": "stopped"},
        {"worker_reaped": False},
        {"error": "worker failed"},
        {"artifacts": [{"relative_path": "a", "size_bytes": 1, "sha256": DIGEST}] * 2},
        {"artifacts": [{"relative_path": "a", "size_bytes": -1, "sha256": DIGEST}]},
        {"artifacts": [{"relative_path": "a", "size_bytes": 1, "sha256": "C" * 64}]},
        {"job_id": "sj_short"},
        {"supervisor": "api:1:1"},
    ],
    ids=[
        "failure-with-result",
        "failure-without-error",
        "timeout-without-error",
        "cancellation-with-result",
        "success-unreaped",
        "success-with-error",
        "duplicate-path",
        "negative-size",
        "uppercase-digest",
        "malformed-job",
        "supervisor-field",
    ],
)
def test_inconsistent_requests_are_refused(changes: dict[str, object]) -> None:
    """Outcome/error mismatch, duplicates and unknown fields never decode."""
    with pytest.raises(ValidationError):
        _request(**changes)


def test_unsuccessful_outcome_carries_its_error() -> None:
    """A failed job reports a bounded error and may still declare artefacts."""
    request = _request(outcome="failed", error="worker failed", result=None)
    assert request.error == "worker failed" and len(request.artifacts) == 2


@pytest.mark.parametrize("error", [None, "workers were not reaped"])
def test_cancellation_error_is_optional_and_reaping_may_be_unconfirmed(
    error: str | None,
) -> None:
    """A cancelled job reports an error only when there is one, as embedded does."""
    request = _request(outcome="cancelled", result=None, error=error, worker_reaped=False)
    assert (request.error, request.worker_reaped) == (error, False)


@pytest.mark.parametrize(
    "payload,match",
    [
        (b"", "byte limit"),
        (b"{" + b" " * 5000 + b"}", "byte limit"),
        (b"\xff", "invalid storage finish JSON"),
        (b'{"a":1,"a":2}', "duplicate storage finish field"),
        (b'{"a":NaN}', "nonfinite storage finish constant"),
    ],
)
def test_malformed_frames_are_refused(payload: bytes, match: str) -> None:
    """Size, encoding, duplicate names and nonfinite constants refuse before schema."""
    with pytest.raises(ValueError, match=match):
        decode_finish_request(payload, max_bytes=4096)


def _response(**changes: object) -> bytes:
    fields: dict[str, object] = {
        "schema_version": FINISH_SCHEMA_VERSION,
        "operation": "finish",
        "request_id": "d" * 32,
        "job_id": JOB,
        "reply": "sealed",
        "reason": None,
    }
    fields.update(changes)
    return json.dumps(fields).encode()


def test_responses_must_answer_the_sent_request() -> None:
    """Only a correlated, consistent answer is accepted."""
    request = _request()
    accepted = decode_finish_response(_response(), request=request, max_bytes=4096)
    assert (accepted.reply, accepted.reason) == ("sealed", None)
    refused = decode_finish_response(
        _response(reply="refused", reason="conflict"), request=request, max_bytes=4096
    )
    assert refused.reason == "conflict"
    with pytest.raises(ValueError, match="does not answer"):
        decode_finish_response(_response(request_id="e" * 32), request=request, max_bytes=4096)
    for inconsistent in ({"reason": "bytes"}, {"reply": "refused"}):
        with pytest.raises(ValidationError):
            decode_finish_response(_response(**inconsistent), request=request, max_bytes=4096)
    assert isinstance(
        StorageFinishResponse.model_validate_json(_response(), strict=True), StorageFinishResponse
    )


@pytest.mark.parametrize(
    "limits,match",
    [
        ({"max_artifact_entries": 1}, "entry limit"),
        ({"frame_max_bytes": 2}, "frame limit"),
        ({"max_artifact_bytes": 2}, "aggregate limit"),
    ],
)
def test_manifest_budgets_are_trusted_configuration(limits: dict[str, int], match: str) -> None:
    """Entry, frame and aggregate budgets bound the declared artefacts."""
    budgets = {"frame_max_bytes": 1024, "max_artifact_bytes": 1024, "max_artifact_entries": 8}
    validate_finish_manifest(_request(), **budgets)
    budgets.update(limits)
    with pytest.raises(ValueError, match=match):
        validate_finish_manifest(_request(), **budgets)
