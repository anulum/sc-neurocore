# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio adaptive-precision routes

"""Tune synapse precision and materialise formal evidence bundles."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
from fastapi import APIRouter

from sc_neurocore.compiler import (
    assign_synapse_precisions,
    auto_tune_synapse_precisions,
    write_precision_formal_evidence_bundle,
)
from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import (
    AdaptivePrecisionAutoTuneRequest,
    AdaptivePrecisionFormalBundleRequest,
)


def _parse_layer_weight_arrays(
    layer_weights: list[list[list[float]] | list[float]],
) -> list[np.ndarray[Any, Any]]:
    arrays: list[np.ndarray[Any, Any]] = []
    for idx, layer in enumerate(layer_weights):
        array = np.asarray(layer, dtype=float)
        if array.size == 0:
            raise ValueError(f"layer {idx} must not be empty")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"layer {idx} contains non-finite values")
        arrays.append(array)
    return arrays


def _run_adaptive_precision_auto_tune_payload(payload: dict[str, Any]) -> dict[str, Any]:
    layer_arrays = _parse_layer_weight_arrays(payload["layer_weights"])
    return auto_tune_synapse_precisions(
        layer_arrays,
        layer_names=payload.get("layer_names"),
        target_error_percent=float(payload.get("target_error_percent", 0.1)),
        min_bits=int(payload.get("min_bits", 4)),
        max_bits=int(payload.get("max_bits", 16)),
        min_length=int(payload.get("min_length", 32)),
        max_length=int(payload.get("max_length", 4096)),
        confidence=float(payload.get("confidence", 0.95)),
    )


def _run_adaptive_precision_formal_bundle_payload(payload: dict[str, Any]) -> dict[str, Any]:
    layer_arrays = _parse_layer_weight_arrays(payload["layer_weights"])
    assignments = assign_synapse_precisions(
        layer_arrays,
        layer_names=payload.get("layer_names"),
        target_error=float(payload.get("target_error_percent", 0.1)) / 100.0,
        min_bits=int(payload.get("min_bits", 4)),
        max_bits=int(payload.get("max_bits", 16)),
        min_length=int(payload.get("min_length", 32)),
        max_length=int(payload.get("max_length", 4096)),
        confidence=float(payload.get("confidence", 0.95)),
    )
    module_name = str(payload.get("module_name", "adaptive_precision_plan"))
    with tempfile.TemporaryDirectory(prefix="scnc_precision_bundle_") as tmp_dir:
        bundle_manifest = write_precision_formal_evidence_bundle(
            tmp_dir, assignments, module_name=module_name
        )
        root = Path(tmp_dir)
        artifact_texts: dict[str, str] = {}
        for key, rel_path in bundle_manifest["artifacts"].items():
            artifact_path = root / rel_path
            if artifact_path.exists():
                artifact_texts[key] = artifact_path.read_text(encoding="utf-8")
            else:
                artifact_texts[key] = ""
        formal_manifest_path = root / f"{module_name}_formal_manifest.json"
        return {
            "bundle_manifest": bundle_manifest,
            "formal_manifest_json": formal_manifest_path.read_text(encoding="utf-8"),
            "artifacts_text": artifact_texts,
        }


def _execute_resolved_preset_action(resolved: dict[str, Any]) -> dict[str, Any]:
    endpoint = resolved.get("endpoint")
    payload = cast(dict[str, Any], resolved.get("payload"))
    if endpoint == "/api/adaptive-precision/auto-tune":
        return _run_adaptive_precision_auto_tune_payload(payload)
    if endpoint == "/api/adaptive-precision/formal-bundle":
        return _run_adaptive_precision_formal_bundle_payload(payload)
    raise ValueError(f"preset action endpoint is not executable: {endpoint}")


def _is_executable_preset_action_endpoint(endpoint: Any) -> bool:
    return endpoint in {
        "/api/adaptive-precision/auto-tune",
        "/api/adaptive-precision/formal-bundle",
    }


def build_adaptive_precision_router(context: StudioApiContext) -> APIRouter:
    """Build the adaptive-precision router over shared Studio runtime state."""
    router = APIRouter()

    @router.post("/api/adaptive-precision/auto-tune")
    def api_adaptive_precision_auto_tune(req: AdaptivePrecisionAutoTuneRequest) -> Any:
        payload = req.model_dump()
        return _safe(lambda: _run_adaptive_precision_auto_tune_payload(payload))

    @router.post("/api/adaptive-precision/formal-bundle")
    def api_adaptive_precision_formal_bundle(req: AdaptivePrecisionFormalBundleRequest) -> Any:
        payload = req.model_dump()
        return _safe(lambda: _run_adaptive_precision_formal_bundle_payload(payload))

    return router
