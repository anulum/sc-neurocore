# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MLPerf-SC result schema and validator

"""Fail-closed validation for MLPerf-SC benchmark result records."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence, cast

MLPERF_SC_RESULT_SCHEMA_VERSION = "sc-neurocore.mlperf-sc-result.v0.1"

MLPerfSCEvidenceClass = Literal[
    "simulation",
    "synthesis_estimate",
    "board_measurement",
    "asic_estimate",
    "analytical_estimate",
]
MLPerfSCMode = Literal[
    "unipolar",
    "bipolar",
    "low_discrepancy",
    "deterministic_replay",
    "mixed",
]

_EVIDENCE_CLASSES = frozenset(
    {
        "simulation",
        "synthesis_estimate",
        "board_measurement",
        "asic_estimate",
        "analytical_estimate",
    }
)
_SC_MODES = frozenset(
    {
        "unipolar",
        "bipolar",
        "low_discrepancy",
        "deterministic_replay",
        "mixed",
    }
)
_ARTIFACT_KINDS = frozenset(
    {
        "raw_results",
        "environment_manifest",
        "synthesis_report",
        "timing_report",
        "power_report",
        "board_log",
        "power_trace",
        "thermal_trace",
        "asic_report",
        "analysis_notebook",
        "summary_report",
    }
)
_RAW_BOARD_KINDS = frozenset({"board_log", "power_trace", "thermal_trace"})
_SYNTHESIS_KINDS = frozenset({"synthesis_report", "timing_report", "power_report"})
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MLPerfSCValidationError(ValueError):
    """Raised when an MLPerf-SC result violates the fail-closed schema."""


@dataclass(frozen=True, slots=True)
class MLPerfSCRun:
    """Benchmark run identity and dataset contract."""

    run_id: str
    task: str
    model: str
    dataset: str
    started_at: str
    producer: str


@dataclass(frozen=True, slots=True)
class MLPerfSCExecution:
    """Execution target and stochastic-computing mode metadata."""

    backend: str
    target: str
    sc_mode: MLPerfSCMode
    bitstream_length: int
    seed: int | None


@dataclass(frozen=True, slots=True)
class MLPerfSCArea:
    """Hardware utilisation metrics when a synthesis or board path exists."""

    luts: int | None
    ffs: int | None
    bram: int | None
    dsp: int | None


@dataclass(frozen=True, slots=True)
class MLPerfSCMetrics:
    """Accuracy, latency, energy, power, and area metrics."""

    accuracy: float
    latency_ms: float
    throughput_inferences_per_s: float | None
    energy_j_per_inference: float | None
    power_w: float | None
    area: MLPerfSCArea


@dataclass(frozen=True, slots=True)
class MLPerfSCArtifact:
    """One evidence artifact referenced by the result."""

    kind: str
    path: str
    sha256: str | None


@dataclass(frozen=True, slots=True)
class MLPerfSCEvidence:
    """Evidence class, environment manifest, and raw artifact references."""

    evidence_class: MLPerfSCEvidenceClass
    environment: Mapping[str, str]
    artifacts: Sequence[MLPerfSCArtifact]


@dataclass(frozen=True, slots=True)
class MLPerfSCResult:
    """Typed MLPerf-SC benchmark result."""

    run: MLPerfSCRun
    execution: MLPerfSCExecution
    metrics: MLPerfSCMetrics
    evidence: MLPerfSCEvidence
    schema_version: str = MLPERF_SC_RESULT_SCHEMA_VERSION


def validate_mlperf_sc_result(
    payload: Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
) -> MLPerfSCResult:
    """Validate a decoded MLPerf-SC result and return a typed result."""

    _expect_keys(payload, {"schema_version", "run", "execution", "metrics", "evidence"}, "result")
    if payload["schema_version"] != MLPERF_SC_RESULT_SCHEMA_VERSION:
        raise MLPerfSCValidationError(f"schema_version must be {MLPERF_SC_RESULT_SCHEMA_VERSION!r}")

    run = _run_from_mapping(_expect_mapping(payload["run"], "run"))
    execution = _execution_from_mapping(_expect_mapping(payload["execution"], "execution"))
    metrics = _metrics_from_mapping(_expect_mapping(payload["metrics"], "metrics"))
    evidence = _evidence_from_mapping(
        _expect_mapping(payload["evidence"], "evidence"),
        artifact_root=Path(artifact_root) if artifact_root is not None else None,
    )
    _validate_evidence_metrics(evidence, metrics)
    return MLPerfSCResult(
        schema_version=cast(str, payload["schema_version"]),
        run=run,
        execution=execution,
        metrics=metrics,
        evidence=evidence,
    )


def mlperf_sc_result_to_dict(result: MLPerfSCResult) -> dict[str, Any]:
    """Serialise a typed MLPerf-SC result to the canonical dictionary shape."""

    return {
        "schema_version": result.schema_version,
        "run": {
            "run_id": result.run.run_id,
            "task": result.run.task,
            "model": result.run.model,
            "dataset": result.run.dataset,
            "started_at": result.run.started_at,
            "producer": result.run.producer,
        },
        "execution": {
            "backend": result.execution.backend,
            "target": result.execution.target,
            "sc_mode": result.execution.sc_mode,
            "bitstream_length": result.execution.bitstream_length,
            "seed": result.execution.seed,
        },
        "metrics": {
            "accuracy": result.metrics.accuracy,
            "latency_ms": result.metrics.latency_ms,
            "throughput_inferences_per_s": result.metrics.throughput_inferences_per_s,
            "energy_j_per_inference": result.metrics.energy_j_per_inference,
            "power_w": result.metrics.power_w,
            "area": {
                "luts": result.metrics.area.luts,
                "ffs": result.metrics.area.ffs,
                "bram": result.metrics.area.bram,
                "dsp": result.metrics.area.dsp,
            },
        },
        "evidence": {
            "evidence_class": result.evidence.evidence_class,
            "environment": dict(result.evidence.environment),
            "artifacts": [
                {
                    "kind": artifact.kind,
                    "path": artifact.path,
                    "sha256": artifact.sha256,
                }
                for artifact in result.evidence.artifacts
            ],
        },
    }


def _run_from_mapping(payload: Mapping[str, Any]) -> MLPerfSCRun:
    _expect_keys(payload, {"run_id", "task", "model", "dataset", "started_at", "producer"}, "run")
    return MLPerfSCRun(
        run_id=_expect_non_empty_string(payload["run_id"], "run.run_id"),
        task=_expect_non_empty_string(payload["task"], "run.task"),
        model=_expect_non_empty_string(payload["model"], "run.model"),
        dataset=_expect_non_empty_string(payload["dataset"], "run.dataset"),
        started_at=_expect_non_empty_string(payload["started_at"], "run.started_at"),
        producer=_expect_non_empty_string(payload["producer"], "run.producer"),
    )


def _execution_from_mapping(payload: Mapping[str, Any]) -> MLPerfSCExecution:
    _expect_keys(payload, {"backend", "target", "sc_mode", "bitstream_length", "seed"}, "execution")
    sc_mode = _expect_non_empty_string(payload["sc_mode"], "execution.sc_mode")
    if sc_mode not in _SC_MODES:
        raise MLPerfSCValidationError("execution.sc_mode has unsupported value")
    bitstream_length = _expect_int(payload["bitstream_length"], "execution.bitstream_length")
    if bitstream_length <= 0:
        raise MLPerfSCValidationError("execution.bitstream_length must be positive")
    seed = payload["seed"]
    if seed is not None:
        seed = _expect_int(seed, "execution.seed")
        if seed < 0:
            raise MLPerfSCValidationError("execution.seed must be non-negative")
    return MLPerfSCExecution(
        backend=_expect_non_empty_string(payload["backend"], "execution.backend"),
        target=_expect_non_empty_string(payload["target"], "execution.target"),
        sc_mode=cast(MLPerfSCMode, sc_mode),
        bitstream_length=bitstream_length,
        seed=seed,
    )


def _metrics_from_mapping(payload: Mapping[str, Any]) -> MLPerfSCMetrics:
    _expect_keys(
        payload,
        {
            "accuracy",
            "latency_ms",
            "throughput_inferences_per_s",
            "energy_j_per_inference",
            "power_w",
            "area",
        },
        "metrics",
    )
    accuracy = _expect_float(payload["accuracy"], "metrics.accuracy")
    if not 0.0 <= accuracy <= 1.0:
        raise MLPerfSCValidationError("metrics.accuracy must be between 0 and 1")
    latency_ms = _expect_float(payload["latency_ms"], "metrics.latency_ms")
    if latency_ms < 0.0:
        raise MLPerfSCValidationError("metrics.latency_ms must be non-negative")
    throughput = _expect_optional_positive_float(
        payload["throughput_inferences_per_s"],
        "metrics.throughput_inferences_per_s",
    )
    energy = _expect_optional_positive_float(
        payload["energy_j_per_inference"],
        "metrics.energy_j_per_inference",
    )
    power = _expect_optional_positive_float(payload["power_w"], "metrics.power_w")
    return MLPerfSCMetrics(
        accuracy=accuracy,
        latency_ms=latency_ms,
        throughput_inferences_per_s=throughput,
        energy_j_per_inference=energy,
        power_w=power,
        area=_area_from_mapping(_expect_mapping(payload["area"], "metrics.area")),
    )


def _area_from_mapping(payload: Mapping[str, Any]) -> MLPerfSCArea:
    _expect_keys(payload, {"luts", "ffs", "bram", "dsp"}, "metrics.area")
    return MLPerfSCArea(
        luts=_expect_optional_non_negative_int(payload["luts"], "metrics.area.luts"),
        ffs=_expect_optional_non_negative_int(payload["ffs"], "metrics.area.ffs"),
        bram=_expect_optional_non_negative_int(payload["bram"], "metrics.area.bram"),
        dsp=_expect_optional_non_negative_int(payload["dsp"], "metrics.area.dsp"),
    )


def _evidence_from_mapping(
    payload: Mapping[str, Any],
    *,
    artifact_root: Path | None,
) -> MLPerfSCEvidence:
    _expect_keys(payload, {"evidence_class", "environment", "artifacts"}, "evidence")
    evidence_class = _expect_non_empty_string(payload["evidence_class"], "evidence.evidence_class")
    if evidence_class not in _EVIDENCE_CLASSES:
        raise MLPerfSCValidationError("evidence.evidence_class has unsupported value")
    environment_payload = _expect_mapping(payload["environment"], "evidence.environment")
    if not environment_payload:
        raise MLPerfSCValidationError("evidence.environment must not be empty")
    environment = {
        _expect_non_empty_string(key, "evidence.environment key"): _expect_non_empty_string(
            value, f"evidence.environment.{key}"
        )
        for key, value in environment_payload.items()
    }
    artifacts_payload = _expect_sequence(payload["artifacts"], "evidence.artifacts")
    artifacts = tuple(
        _artifact_from_mapping(
            _expect_mapping(item, f"evidence.artifacts[{index}]"),
            artifact_root=artifact_root,
        )
        for index, item in enumerate(artifacts_payload)
    )
    if not artifacts:
        raise MLPerfSCValidationError("evidence.artifacts must not be empty")
    return MLPerfSCEvidence(
        evidence_class=cast(MLPerfSCEvidenceClass, evidence_class),
        environment=environment,
        artifacts=artifacts,
    )


def _artifact_from_mapping(
    payload: Mapping[str, Any],
    *,
    artifact_root: Path | None,
) -> MLPerfSCArtifact:
    _expect_keys(payload, {"kind", "path", "sha256"}, "evidence.artifact")
    kind = _expect_non_empty_string(payload["kind"], "evidence.artifact.kind")
    if kind not in _ARTIFACT_KINDS:
        raise MLPerfSCValidationError("evidence.artifact.kind has unsupported value")
    path = _expect_non_empty_string(payload["path"], "evidence.artifact.path")
    _validate_relative_artifact_path(path, artifact_root=artifact_root)
    sha256 = payload["sha256"]
    if sha256 is not None:
        sha256 = _expect_non_empty_string(sha256, "evidence.artifact.sha256")
        if _HEX_SHA256_RE.fullmatch(sha256) is None:
            raise MLPerfSCValidationError("evidence.artifact.sha256 must be 64 lowercase hex chars")
    return MLPerfSCArtifact(kind=kind, path=path, sha256=sha256)


def _validate_relative_artifact_path(path: str, *, artifact_root: Path | None) -> None:
    artifact_path = Path(path)
    if artifact_path.is_absolute() or ".." in artifact_path.parts:
        raise MLPerfSCValidationError("evidence artifact path must be relative and contained")
    if artifact_root is not None:
        root = artifact_root.resolve()
        resolved = (root / artifact_path).resolve()
        if root != resolved and root not in resolved.parents:
            raise MLPerfSCValidationError("evidence artifact path escapes artifact_root")
        if not resolved.is_file():
            raise MLPerfSCValidationError(f"evidence artifact path does not exist: {path}")


def _validate_evidence_metrics(
    evidence: MLPerfSCEvidence,
    metrics: MLPerfSCMetrics,
) -> None:
    kinds = {artifact.kind for artifact in evidence.artifacts}
    if evidence.evidence_class == "board_measurement":
        if not kinds & _RAW_BOARD_KINDS:
            raise MLPerfSCValidationError(
                "board_measurement evidence requires at least one raw board artifact"
            )
        if metrics.energy_j_per_inference is None and metrics.power_w is None:
            raise MLPerfSCValidationError(
                "board_measurement evidence requires energy or power metrics"
            )
    if evidence.evidence_class in {"synthesis_estimate", "asic_estimate"}:
        if not kinds & _SYNTHESIS_KINDS and "asic_report" not in kinds:
            raise MLPerfSCValidationError(
                f"{evidence.evidence_class} evidence requires synthesis or ASIC report artifacts"
            )
    if evidence.evidence_class in {"simulation", "analytical_estimate"}:
        if metrics.energy_j_per_inference is not None or metrics.power_w is not None:
            raise MLPerfSCValidationError(
                f"{evidence.evidence_class} evidence must not claim measured energy or power"
            )


def _expect_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise MLPerfSCValidationError(f"{label} missing required keys: {sorted(missing)}")
    if extra:
        raise MLPerfSCValidationError(f"{label} has unsupported keys: {sorted(extra)}")


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MLPerfSCValidationError(f"{label} must be a mapping")
    return value


def _expect_sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise MLPerfSCValidationError(f"{label} must be a sequence")
    return cast(Sequence[Any], value)


def _expect_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise MLPerfSCValidationError(f"{label} must be a non-empty string")
    return value


def _expect_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MLPerfSCValidationError(f"{label} must be an integer")
    return cast(int, value)


def _expect_optional_non_negative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    integer = _expect_int(value, label)
    if integer < 0:
        raise MLPerfSCValidationError(f"{label} must be non-negative")
    return integer


def _expect_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MLPerfSCValidationError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise MLPerfSCValidationError(f"{label} must be finite")
    return numeric


def _expect_optional_positive_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    numeric = _expect_float(value, label)
    if numeric <= 0.0:
        raise MLPerfSCValidationError(f"{label} must be positive when present")
    return numeric
