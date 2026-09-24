# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public job snapshot decoding

"""Decode complete public job snapshots without opening the ledger."""

import json
from collections.abc import Mapping
from dataclasses import fields

from pydantic import TypeAdapter

from sc_neurocore.studio.platform.jobs_models import StudioJobArtifact, StudioJobRecord
from sc_neurocore.studio.training_contract import TrainingConfigError, resolve_training_config

_RECORD = TypeAdapter(StudioJobRecord)
_RECORD_FIELDS = frozenset(field.name for field in fields(StudioJobRecord))
_ARTIFACT_FIELDS = frozenset(field.name for field in fields(StudioJobArtifact))


def decode_job_snapshot(payload: Mapping[str, object]) -> StudioJobRecord:
    """Decode an exact complete snapshot, retaining every custody field.

    Parameters
    ----------
    payload : mapping
        Public record JSON, including nullable fields and artifact declarations.

    Returns
    -------
    StudioJobRecord
        Domain record reconstructed from JSON, without storage or identity claims.

    Raises
    ------
    ValueError
        Fields are missing, unknown, non-JSON or violate native record types.
    """
    if set(payload) != _RECORD_FIELDS:
        raise ValueError("Studio job snapshot fields do not match the public record.")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or any(
        not isinstance(item, dict) or set(item) != _ARTIFACT_FIELDS for item in artifacts
    ):
        raise ValueError("Studio job snapshot artifact fields are invalid.")
    try:
        encoded = json.dumps(dict(payload), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Studio job snapshot must contain finite JSON values.") from exc
    record = _RECORD.validate_json(encoded, strict=True)
    config = record.training_config
    if config is not None:
        if record.kind != "training":
            raise ValueError("Non-training snapshot carries a training configuration.")
        canonical = json.dumps(config, allow_nan=False, sort_keys=True, separators=(",", ":"))
        if len(canonical.encode("utf-8")) > 4096:
            raise ValueError("Training snapshot exceeds the 4096-byte limit.")
        try:
            resolved = resolve_training_config(config).to_public_dict()
        except TrainingConfigError as exc:
            raise ValueError("Training snapshot configuration is invalid.") from exc
        if canonical != json.dumps(
            resolved, allow_nan=False, sort_keys=True, separators=(",", ":")
        ):
            raise ValueError("Training snapshot configuration is not canonical.")
    return record
