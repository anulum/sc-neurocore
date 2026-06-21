# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio analysis result manifests

"""Path-free reproducibility manifests for Studio analysis results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from sc_neurocore.studio.evidence_classification import (
    StudioEvidenceClassification,
    StudioEvidenceStatus,
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)

STUDIO_ANALYSIS_RESULT_SCHEMA_VERSION = "studio.analysis-result.v1"

AnalysisSource: TypeAlias = Literal["ode", "model", "mixed", "unknown"]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class StudioAnalysisResultManifest:
    """Path-free metadata for one Studio analysis response.

    Parameters
    ----------
    analysis_type:
        Stable analysis endpoint identifier such as ``"fi_curve"``.
    source:
        Input surface used to produce the analysis result.
    input_sha256:
        SHA-256 digest of the canonical request payload.
    result_sha256:
        SHA-256 digest of the canonical result payload without this manifest.
    output_keys:
        Sorted top-level keys present in the result payload before metadata.
    evidence_classification:
        Stable evidence lane label for analysis results.
    status:
        Terminal status for this analysis evidence object.
    """

    analysis_type: str
    source: AnalysisSource
    input_sha256: str
    result_sha256: str
    output_keys: tuple[str, ...]
    evidence_classification: StudioEvidenceClassification = "analysis"
    status: StudioEvidenceStatus = "completed"

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the public, path-free analysis manifest."""

        return {
            "analysis_type": self.analysis_type,
            "evidence_classification": validate_studio_evidence_classification(
                self.evidence_classification
            ),
            "input_sha256": self.input_sha256,
            "output_keys": list(self.output_keys),
            "result_sha256": self.result_sha256,
            "schema_version": STUDIO_ANALYSIS_RESULT_SCHEMA_VERSION,
            "source": self.source,
            "status": validate_studio_evidence_status(self.status),
        }


def build_analysis_result_manifest(
    *,
    analysis_type: str,
    source: AnalysisSource,
    request_payload: Mapping[str, Any],
    result_payload: Mapping[str, Any],
) -> StudioAnalysisResultManifest:
    """Build digest-backed reproducibility metadata for an analysis result.

    Parameters
    ----------
    analysis_type:
        Stable analysis endpoint identifier.
    source:
        Input surface used to produce the result.
    request_payload:
        Request body used to run the analysis.
    result_payload:
        Result payload before or after metadata insertion.

    Returns
    -------
    StudioAnalysisResultManifest
        Path-free metadata suitable for UI display, exports, and evidence
        bundles.

    Raises
    ------
    ValueError
        If request or result payloads cannot be encoded as portable JSON.
    """

    result_without_manifest = {
        key: value for key, value in result_payload.items() if key != "analysis_metadata"
    }
    return StudioAnalysisResultManifest(
        analysis_type=analysis_type,
        source=source,
        input_sha256=_sha256_json(request_payload),
        result_sha256=_sha256_json(result_without_manifest),
        output_keys=tuple(sorted(str(key) for key in result_without_manifest)),
    )


def attach_analysis_result_manifest(
    *,
    analysis_type: str,
    source: AnalysisSource,
    request_payload: Mapping[str, Any],
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return an analysis result with a path-free metadata manifest attached.

    Parameters
    ----------
    analysis_type:
        Stable analysis endpoint identifier.
    source:
        Input surface used to produce the result.
    request_payload:
        Request body used to run the analysis.
    result_payload:
        Mutable analysis result payload.

    Returns
    -------
    dict[str, Any]
        The same result object with ``analysis_metadata`` set.
    """

    result_payload["analysis_metadata"] = build_analysis_result_manifest(
        analysis_type=analysis_type,
        source=source,
        request_payload=request_payload,
        result_payload=result_payload,
    ).to_public_dict()
    return result_payload


def infer_analysis_source(request_payload: Mapping[str, Any]) -> AnalysisSource:
    """Infer the Studio input surface from an analysis request payload.

    Parameters
    ----------
    request_payload:
        Request body submitted to an analysis endpoint.

    Returns
    -------
    AnalysisSource
        ``"model"`` when a model name is present, ``"ode"`` when equations are
        present, ``"mixed"`` for comparison payloads, otherwise ``"unknown"``.
    """

    if "config_a" in request_payload or "config_b" in request_payload:
        return "mixed"
    model_name = request_payload.get("model_name")
    if isinstance(model_name, str) and model_name:
        return "model"
    equations = request_payload.get("equations")
    if isinstance(equations, list) and equations:
        return "ode"
    return "unknown"


def _sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest over portable canonical JSON."""

    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Analysis manifest payload must be portable JSON.") from exc
    return hashlib.sha256(encoded).hexdigest()
