# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Training weight restore process adapter

"""Restore verified checkpoint seeds through the original training loader."""

import json
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, JsonValue

from sc_neurocore.studio.platform.jobs_context import StudioJobContext
from sc_neurocore.studio.platform.training_weight_loader import load_training_weight_state_dict
from sc_neurocore.studio.platform.training_weights import (
    TRAINING_WEIGHT_RESTORE_EVIDENCE_ARTIFACT_PATH,
    build_training_weight_restore_evidence,
    materialize_training_weight_payload,
)

RESTORE_METADATA_SEED = "restore-metadata.json"
RESTORE_WEIGHTS_SEED = "restore-weights.pt"


class _RestoreRequest(BaseModel):
    """Exact JSON plan envelope, without paths, loaders or tensor state."""

    model_config = ConfigDict(extra="forbid", strict=True)
    restore_plan: dict[str, JsonValue]
    source_status: str


def execute_training_weight_restore_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Verify binary seeds and emit the existing path-free restore receipt.

    Parameters
    ----------
    context : StudioJobContext
        Registered worker context holding bounded metadata and checkpoint seeds.
    payload : mapping
        Exactly ``restore_plan`` and ``source_status`` from the source job.
        The original materializer validates lengths and digests before loading.

    Returns
    -------
    dict[str, object]
        Restore evidence also written as the canonical JSON artifact. Loaded
        tensor state remains inside this worker and is not returned.

    Raises
    ------
    ValueError
        Envelope, plan, seed integrity or restricted checkpoint loading fails.
    StudioJobArtifactUnavailable
        A required submission seed is absent.
    """
    request = _RestoreRequest.model_validate(dict(payload))
    materialization = materialize_training_weight_payload(
        restore_plan=request.restore_plan,
        metadata_payload=context.read_seed_input(RESTORE_METADATA_SEED),
        weights_payload=context.read_seed_input(RESTORE_WEIGHTS_SEED),
        trusted_loader=load_training_weight_state_dict,
    )
    evidence = build_training_weight_restore_evidence(
        materialization, source_status=request.source_status
    )
    context.write_artifact(
        TRAINING_WEIGHT_RESTORE_EVIDENCE_ARTIFACT_PATH,
        json.dumps(evidence, indent=2, sort_keys=True),
    )
    return dict(evidence)
