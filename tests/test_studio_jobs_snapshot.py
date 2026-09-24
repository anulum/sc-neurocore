# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public job snapshot contracts

"""Complete public record round trips and strict malformed snapshot refusal."""

import hashlib
import json
from dataclasses import replace

import pytest

from sc_neurocore.studio.platform.jobs_models import StudioJobArtifact, StudioJobRecord
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot
from sc_neurocore.studio.training_contract import resolve_training_config


def _record() -> StudioJobRecord:
    """Represent every optional custody field and nested result value."""
    return StudioJobRecord(
        job_id="sj_0123456789abcdef",
        kind="evidence",
        owner="studio-evidence",
        request_id="request",
        status="completed",
        execution_model="process",
        created_at_utc="2026-09-12T00:00:00Z",
        started_at_utc="2026-09-12T00:00:01Z",
        finished_at_utc="2026-09-12T00:00:02Z",
        error="retained diagnostic",
        result={"nested": [1, 2.5, True, None, {"unicode": "váhy"}]},
        artifacts=(StudioJobArtifact("nested/data.bin", 4, hashlib.sha256(b"data").hexdigest()),),
        workspace="workspace",
        idempotency_key="key",
        experiment_sha256="a" * 64,
        admission={"policy": "admitted", "bytes": 4},
        lease_owner="supervisor",
        lease_expires_at_utc="2026-09-12T00:00:03Z",
        heartbeat_at_utc="2026-09-12T00:00:02Z",
    )


def test_snapshot_roundtrip_preserves_every_public_field() -> None:
    """Decode actual JSON without dropping custody, numeric types or Unicode."""
    record = _record()
    decoded = decode_job_snapshot(json.loads(json.dumps(record.to_public_dict())))
    assert decoded == record
    assert decoded.to_public_dict() == record.to_public_dict()
    assert decoded is not record


def test_training_snapshot_roundtrip_retains_validated_config() -> None:
    """A storage client reconstructs the same training config as the ledger."""
    config = resolve_training_config({"epochs": 1, "hidden": [4]}).to_public_dict()
    record = replace(_record(), kind="training", training_config=config)

    assert decode_job_snapshot(json.loads(json.dumps(record.to_public_dict()))) == record


@pytest.mark.parametrize(
    ("kind", "config"),
    [
        ("evidence", resolve_training_config({"epochs": 1}).to_public_dict()),
        ("training", {"epochs": 0}),
        ("training", {"hidden": [1] * 2000}),
    ],
)
def test_snapshot_refuses_invalid_training_configuration(
    kind: str, config: dict[str, object]
) -> None:
    """Unknown, oversized, or wrong-kind config never crosses the wire."""
    payload = replace(_record(), kind=kind).to_public_dict()
    payload["training_config"] = config
    with pytest.raises(ValueError):
        decode_job_snapshot(payload)


@pytest.mark.parametrize("field", list(_record().to_public_dict()))
def test_snapshot_refuses_each_missing_field(field: str) -> None:
    """Even nullable/defaulted fields are mandatory in a complete snapshot."""
    payload = _record().to_public_dict()
    del payload[field]
    with pytest.raises(ValueError, match="snapshot fields"):
        decode_job_snapshot(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unexpected", "not permitted"),
        ("status", "invented"),
        ("execution_model", "remote"),
        ("owner", 2),
        ("admission", []),
        ("artifacts", None),
        ("artifacts", [None]),
        ("artifacts", [{}]),
        ("artifacts", [{"relative_path": "data", "size_bytes": True, "sha256": "a" * 64}]),
        ("result", {"bad": float("nan")}),
        ("result", {"bad": object()}),
    ],
)
def test_snapshot_refuses_unknown_fields_types_and_non_json(field: str, value: object) -> None:
    """Transport accepts JSON-native record types, not coercions or objects."""
    payload = _record().to_public_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        decode_job_snapshot(payload)
