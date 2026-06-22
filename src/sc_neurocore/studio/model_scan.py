# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Batch model scanning for behaviour classification

"""Evidence-producing model scan workflow for the Studio model browser."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from sc_neurocore.studio.codegen import classify_firing_pattern
from sc_neurocore.studio.evidence_classification import (
    StudioEvidenceClassification,
    StudioEvidenceStatus,
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)
from sc_neurocore.studio.models import list_models, simulate_model

STUDIO_MODEL_SCAN_SCHEMA_VERSION = "studio.model-scan.v1"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
ModelScanCacheKey: TypeAlias = tuple[float, float]


@dataclass(frozen=True, slots=True)
class ModelScanEntry:
    """Path-free firing-pattern classification for one Studio model.

    A model that could not be driven by the scan's constant current carries the
    ``error`` pattern and a non-empty ``error_type`` so the failure is visible in
    the result rather than aborting the whole scan.
    """

    name: str
    category: str
    pattern: str
    description: str
    rate_hz: float
    spike_count: int
    error_type: str = ""

    @property
    def is_error(self) -> bool:
        """True when this entry records a simulation failure."""

        return bool(self.error_type)

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible model scan entry."""

        entry: dict[str, JsonValue] = {
            "category": self.category,
            "description": self.description,
            "name": self.name,
            "pattern": self.pattern,
            "rate_hz": self.rate_hz,
            "spike_count": self.spike_count,
        }
        if self.error_type:
            entry["error_type"] = self.error_type
        return entry


@dataclass(frozen=True, slots=True)
class StudioModelScanManifest:
    """Path-free metadata for one complete Studio model scan.

    Parameters
    ----------
    current:
        Constant input current used for each model simulation.
    duration:
        Simulation duration used for each model scan run.
    model_count:
        Number of successfully classified models in the result.
    pattern_counts:
        Count of each detected firing-pattern label.
    input_sha256:
        SHA-256 digest of the scan configuration.
    result_sha256:
        SHA-256 digest of the returned model classifications.
    evidence_classification:
        Controlled evidence class for the scan workflow.
    status:
        Controlled terminal status for the scan workflow.
    """

    current: float
    duration: float
    model_count: int
    pattern_counts: Mapping[str, int]
    input_sha256: str
    result_sha256: str
    error_count: int = 0
    failed_models: tuple[Mapping[str, str], ...] = ()
    evidence_classification: StudioEvidenceClassification = "analysis"
    status: StudioEvidenceStatus = "completed"

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the JSON-compatible model scan metadata.

        ``error_count`` and ``failed_models`` make a partial scan explicit, so a
        consumer never mistakes a result with undriveable models for a complete
        classification of the whole catalogue.
        """

        return {
            "current": self.current,
            "duration": self.duration,
            "error_count": self.error_count,
            "evidence_classification": validate_studio_evidence_classification(
                self.evidence_classification
            ),
            "failed_models": [dict(sorted(model.items())) for model in self.failed_models],
            "input_sha256": self.input_sha256,
            "model_count": self.model_count,
            "pattern_counts": dict(sorted(self.pattern_counts.items())),
            "result_sha256": self.result_sha256,
            "schema_version": STUDIO_MODEL_SCAN_SCHEMA_VERSION,
            "status": validate_studio_evidence_status(self.status),
        }


_CACHE: dict[ModelScanCacheKey, tuple[ModelScanEntry, ...]] = {}


def scan_all_models(current: float = 10.0, duration: float = 100.0) -> dict[str, JsonValue]:
    """Simulate every model at a given current and classify its firing pattern.

    Results are cached per ``(current, duration)`` pair so a scan for one
    configuration cannot be served as evidence for another configuration.
    """
    cache_key = (float(current), float(duration))
    if cache_key not in _CACHE:
        _CACHE[cache_key] = _run_model_scan(current=cache_key[0], duration=cache_key[1])
    entries = _CACHE[cache_key]
    models = cast(list[JsonValue], [entry.to_public_dict() for entry in entries])
    manifest = _build_model_scan_manifest(
        current=cache_key[0], duration=cache_key[1], models=models
    )
    return {
        "models": models,
        "scan_metadata": manifest.to_public_dict(),
        "schema_version": STUDIO_MODEL_SCAN_SCHEMA_VERSION,
    }


def _run_model_scan(*, current: float, duration: float) -> tuple[ModelScanEntry, ...]:
    """Classify every model, recording per-model failures rather than aborting.

    A model that cannot be driven at the scan's constant current — one whose
    ``step`` needs an extra synaptic input, or one whose internal stability guard
    rejects the operating point — yields an ``error`` entry instead of failing the
    whole scan. The failures are surfaced in the manifest (``error_count`` /
    ``failed_models``) so the result stays honest about what was classified.
    """

    results: dict[str, ModelScanEntry] = {}
    models = list_models()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in models:
            name = str(m["name"])
            category = str(m.get("category", "Other"))
            try:
                r = simulate_model(name, duration=duration, current=current)
                pattern = classify_firing_pattern(r["spikes"], r["n_steps"], r["dt"])
                results[name] = ModelScanEntry(
                    name=name,
                    category=category,
                    pattern=str(pattern["pattern"]),
                    description=str(pattern["description"]),
                    rate_hz=float(pattern.get("rate_hz", 0.0)),
                    spike_count=int(r["spike_count"]),
                )
            except Exception as exc:
                results[name] = ModelScanEntry(
                    name=name,
                    category=category,
                    pattern="error",
                    description=f"{type(exc).__name__}: {exc}",
                    rate_hz=0.0,
                    spike_count=0,
                    error_type=type(exc).__name__,
                )

    return tuple(results.values())


def _build_model_scan_manifest(
    *,
    current: float,
    duration: float,
    models: list[JsonValue],
) -> StudioModelScanManifest:
    """Build digest-backed metadata for a complete model scan."""

    pattern_counts: dict[str, int] = {}
    failed_models: list[Mapping[str, str]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        pattern = model.get("pattern")
        if isinstance(pattern, str):
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        if pattern == "error":
            failed_models.append(
                {
                    "name": str(model.get("name", "")),
                    "category": str(model.get("category", "Other")),
                    "error_type": str(model.get("error_type", "")),
                    "error_message": str(model.get("description", "")),
                }
            )
    return StudioModelScanManifest(
        current=current,
        duration=duration,
        model_count=len(models),
        pattern_counts=pattern_counts,
        input_sha256=_sha256_json({"current": current, "duration": duration}),
        result_sha256=_sha256_json({"models": models}),
        error_count=len(failed_models),
        failed_models=tuple(failed_models),
    )


def _sha256_json(payload: Mapping[str, JsonValue]) -> str:
    """Return a SHA-256 digest over canonical JSON."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
