# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Analytic benchmark reports for SC side-channel encoding studies."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .side_channel_metrics import ClassActivityProxy, compute_class_activity_proxy
from .thermal_sc_encoding import (
    ThermalSCEncodingConfig,
    ThermalSCEncodingError,
    encode_activity_balanced_probabilities,
)

SIDE_CHANNEL_BENCHMARK_SCHEMA_VERSION = "sc-neurocore.side-channel-benchmark.v0.1"
SIDE_CHANNEL_DEPLOY_MANIFEST_SCHEMA_VERSION = "sc-neurocore.side-channel-deploy-manifest.v0.1"
_EVIDENCE_BOUNDARY = "analytic_simulation_only"
_EVIDENCE_CLASS = "analytic_simulation"
_THREAT_MODEL = "class_activity_correlation_proxy"
_BOUNDARY_NOTES = (
    "no physical power measurement",
    "no physical thermal measurement",
    "no DPA-resistance claim",
    "no silicon-security claim",
)


class SideChannelBenchmarkError(ValueError):
    """Raised when side-channel benchmark inputs or outputs are invalid."""


@dataclass(frozen=True, slots=True)
class SideChannelBenchmarkArm:
    """One benchmark arm with class-activity leakage proxy evidence."""

    name: str
    class_activity_proxy: ClassActivityProxy
    dummy_stream_overhead_ratio: float
    bitstream_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SideChannelBenchmarkError("arm name must be a non-empty string")
        if not isinstance(self.class_activity_proxy, ClassActivityProxy):
            raise SideChannelBenchmarkError("class_activity_proxy must be a ClassActivityProxy")
        if isinstance(self.dummy_stream_overhead_ratio, bool) or not isinstance(
            self.dummy_stream_overhead_ratio, int | float
        ):
            raise SideChannelBenchmarkError(
                "dummy_stream_overhead_ratio must be a finite non-negative value"
            )
        if (
            not math.isfinite(float(self.dummy_stream_overhead_ratio))
            or float(self.dummy_stream_overhead_ratio) < 0.0
        ):
            raise SideChannelBenchmarkError(
                "dummy_stream_overhead_ratio must be a finite non-negative value"
            )
        if (
            isinstance(self.bitstream_count, bool)
            or not isinstance(self.bitstream_count, int)
            or self.bitstream_count < 0
        ):
            raise SideChannelBenchmarkError("bitstream_count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SideChannelBenchmarkRecord:
    """Per-sample benchmark record with realised protected probability."""

    label: int | float
    probability: float
    protected_realised_probability: float
    protected_dummy_streams_inserted: int

    def __post_init__(self) -> None:
        if isinstance(self.label, bool) or not isinstance(self.label, int | float):
            raise SideChannelBenchmarkError("label must be a finite numeric value")
        if not math.isfinite(float(self.label)):
            raise SideChannelBenchmarkError("label must be a finite numeric value")
        for value, field_name in (
            (self.probability, "probability"),
            (self.protected_realised_probability, "protected_realised_probability"),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise SideChannelBenchmarkError(f"{field_name} must be a finite value in [0, 1]")
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or numeric_value < 0.0 or numeric_value > 1.0:
                raise SideChannelBenchmarkError(f"{field_name} must be a finite value in [0, 1]")
        if (
            isinstance(self.protected_dummy_streams_inserted, bool)
            or not isinstance(self.protected_dummy_streams_inserted, int)
            or self.protected_dummy_streams_inserted < 0
        ):
            raise SideChannelBenchmarkError(
                "protected_dummy_streams_inserted must be a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class SideChannelDeployManifest:
    """Deploy/evidence manifest for an analytic side-channel benchmark."""

    schema_version: str
    evidence_class: str
    benchmark_artifact: dict[str, str]
    security_parameters: dict[str, int | float]
    overhead_measurements: dict[str, int | float]
    boundary_notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise SideChannelBenchmarkError("schema_version must be a non-empty string")
        if not isinstance(self.evidence_class, str) or not self.evidence_class.strip():
            raise SideChannelBenchmarkError("evidence_class must be a non-empty string")
        if not isinstance(self.benchmark_artifact, dict):
            raise SideChannelBenchmarkError("benchmark_artifact must be a dictionary")
        if not isinstance(self.security_parameters, dict):
            raise SideChannelBenchmarkError("security_parameters must be a dictionary")
        if not isinstance(self.overhead_measurements, dict):
            raise SideChannelBenchmarkError("overhead_measurements must be a dictionary")
        if not isinstance(self.boundary_notes, tuple) or not self.boundary_notes:
            raise SideChannelBenchmarkError("boundary_notes must be a non-empty tuple of strings")
        for note in self.boundary_notes:
            if not isinstance(note, str) or not note.strip():
                raise SideChannelBenchmarkError("boundary_notes must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class SideChannelBenchmarkReport:
    """Analytic baseline-versus-protected side-channel benchmark report."""

    schema_version: str
    evidence_boundary: str
    threat_model: str
    baseline: SideChannelBenchmarkArm
    protected: SideChannelBenchmarkArm
    max_class_mean_gap_reduction: float
    deploy_manifest: SideChannelDeployManifest
    boundary_notes: tuple[str, ...]
    records: tuple[SideChannelBenchmarkRecord, ...]


def run_side_channel_leakage_benchmark(
    *,
    probabilities: Sequence[float],
    labels: Sequence[int | float],
    protected_config: ThermalSCEncodingConfig,
) -> SideChannelBenchmarkReport:
    """Compare correlated baseline streams against activity-balanced streams."""

    probability_values = _normalise_probabilities(probabilities)
    label_values = _normalise_labels(labels)
    if len(probability_values) != len(label_values):
        raise SideChannelBenchmarkError("probabilities and labels must have equal length")
    if len(probability_values) < 2:
        raise SideChannelBenchmarkError("at least two samples are required")

    try:
        protected_batch = encode_activity_balanced_probabilities(
            probability_values,
            protected_config,
            labels=label_values,
        )
    except ThermalSCEncodingError as exc:
        raise SideChannelBenchmarkError(str(exc)) from exc

    baseline_samples = tuple(
        (_correlated_activity_fixture_stream(value, protected_config.bitstream_length),)
        for value in probability_values
    )
    baseline_proxy = compute_class_activity_proxy(baseline_samples, label_values)
    protected_proxy = protected_batch.summary.class_activity_proxy
    reduction = baseline_proxy.max_class_mean_gap - protected_proxy.max_class_mean_gap

    total_dummy_streams = sum(record.dummy_streams_inserted for record in protected_batch.records)

    return SideChannelBenchmarkReport(
        schema_version=SIDE_CHANNEL_BENCHMARK_SCHEMA_VERSION,
        evidence_boundary=_EVIDENCE_BOUNDARY,
        threat_model=_THREAT_MODEL,
        baseline=SideChannelBenchmarkArm(
            name="correlated_activity_fixture",
            class_activity_proxy=baseline_proxy,
            dummy_stream_overhead_ratio=0.0,
            bitstream_count=len(baseline_samples),
        ),
        protected=SideChannelBenchmarkArm(
            name="activity_balanced",
            class_activity_proxy=protected_proxy,
            dummy_stream_overhead_ratio=(protected_batch.summary.dummy_stream_overhead_ratio),
            bitstream_count=len(protected_batch.records),
        ),
        max_class_mean_gap_reduction=reduction,
        deploy_manifest=SideChannelDeployManifest(
            schema_version=SIDE_CHANNEL_DEPLOY_MANIFEST_SCHEMA_VERSION,
            evidence_class=_EVIDENCE_CLASS,
            benchmark_artifact={"path": ""},
            security_parameters={
                "bitstream_length": protected_config.bitstream_length,
                "dummy_streams_per_record": protected_config.dummy_streams_per_record,
                "max_dummy_overhead_ratio": (protected_config.max_dummy_overhead_ratio),
                "rotation_stride": protected_config.rotation_stride,
                "seed": protected_config.seed,
            },
            overhead_measurements={
                "dummy_stream_overhead_ratio": (
                    protected_batch.summary.dummy_stream_overhead_ratio
                ),
                "protected_bitstream_count": len(protected_batch.records),
                "total_dummy_streams_inserted": total_dummy_streams,
            },
            boundary_notes=_BOUNDARY_NOTES,
        ),
        boundary_notes=_BOUNDARY_NOTES,
        records=tuple(
            SideChannelBenchmarkRecord(
                label=label,
                probability=probability,
                protected_realised_probability=record.realised_probability,
                protected_dummy_streams_inserted=record.dummy_streams_inserted,
            )
            for label, probability, record in zip(
                label_values,
                probability_values,
                protected_batch.records,
                strict=True,
            )
        ),
    )


def write_side_channel_benchmark_report(
    output_path: str | Path,
    *,
    probabilities: Sequence[float],
    labels: Sequence[int | float],
    protected_config: ThermalSCEncodingConfig,
) -> SideChannelBenchmarkReport:
    """Run the analytic benchmark and write a canonical JSON artifact."""

    report = run_side_channel_leakage_benchmark(
        probabilities=probabilities,
        labels=labels,
        protected_config=protected_config,
    )
    path = Path(output_path)
    report = _with_artifact_path(report, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_report_payload(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _normalise_probabilities(probabilities: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(probabilities, Sequence) or not probabilities:
        raise SideChannelBenchmarkError("probabilities must be a non-empty sequence")
    values: list[float] = []
    for probability in probabilities:
        if isinstance(probability, bool) or not isinstance(probability, int | float):
            raise SideChannelBenchmarkError("probabilities must be finite values in [0, 1]")
        value = float(probability)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise SideChannelBenchmarkError("probabilities must be finite values in [0, 1]")
        values.append(value)
    return tuple(values)


def _normalise_labels(labels: Sequence[int | float]) -> tuple[int | float, ...]:
    if not isinstance(labels, Sequence) or not labels:
        raise SideChannelBenchmarkError("labels must be a non-empty sequence")
    normalised: list[int | float] = []
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, int | float):
            raise SideChannelBenchmarkError("labels must be finite numeric values")
        if not math.isfinite(float(label)):
            raise SideChannelBenchmarkError("labels must be finite numeric values")
        normalised.append(label)
    return tuple(normalised)


def _correlated_activity_fixture_stream(
    probability: float,
    bitstream_length: int,
) -> tuple[int, ...]:
    ones = round(probability * bitstream_length)
    if probability >= 0.5:
        return tuple(index % 2 for index in range(bitstream_length))
    return tuple(1 if index < ones else 0 for index in range(bitstream_length))


def _report_payload(report: SideChannelBenchmarkReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "evidence_boundary": report.evidence_boundary,
        "deploy_manifest": _deploy_manifest_payload(report.deploy_manifest),
        "report": {
            "threat_model": report.threat_model,
            "baseline": _arm_payload(report.baseline),
            "protected": _arm_payload(report.protected),
            "max_class_mean_gap_reduction": report.max_class_mean_gap_reduction,
            "boundary_notes": list(report.boundary_notes),
            "records": [
                {
                    "label": record.label,
                    "probability": record.probability,
                    "protected_realised_probability": (record.protected_realised_probability),
                    "protected_dummy_streams_inserted": (record.protected_dummy_streams_inserted),
                }
                for record in report.records
            ],
        },
    }


def _with_artifact_path(
    report: SideChannelBenchmarkReport,
    path: Path,
) -> SideChannelBenchmarkReport:
    manifest = SideChannelDeployManifest(
        schema_version=report.deploy_manifest.schema_version,
        evidence_class=report.deploy_manifest.evidence_class,
        benchmark_artifact={"path": str(path)},
        security_parameters=report.deploy_manifest.security_parameters,
        overhead_measurements=report.deploy_manifest.overhead_measurements,
        boundary_notes=report.deploy_manifest.boundary_notes,
    )
    return SideChannelBenchmarkReport(
        schema_version=report.schema_version,
        evidence_boundary=report.evidence_boundary,
        threat_model=report.threat_model,
        baseline=report.baseline,
        protected=report.protected,
        max_class_mean_gap_reduction=report.max_class_mean_gap_reduction,
        deploy_manifest=manifest,
        boundary_notes=report.boundary_notes,
        records=report.records,
    )


def _deploy_manifest_payload(manifest: SideChannelDeployManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "evidence_class": manifest.evidence_class,
        "benchmark_artifact": dict(manifest.benchmark_artifact),
        "security_parameters": dict(manifest.security_parameters),
        "overhead_measurements": dict(manifest.overhead_measurements),
        "boundary_notes": list(manifest.boundary_notes),
    }


def _arm_payload(arm: SideChannelBenchmarkArm) -> dict[str, Any]:
    return {
        "name": arm.name,
        "class_activity_proxy": _class_proxy_payload(arm.class_activity_proxy),
        "dummy_stream_overhead_ratio": arm.dummy_stream_overhead_ratio,
        "bitstream_count": arm.bitstream_count,
    }


def _class_proxy_payload(proxy: ClassActivityProxy) -> dict[str, Any]:
    return {
        "class_means": {str(key): value for key, value in proxy.class_means.items()},
        "max_class_mean_gap": proxy.max_class_mean_gap,
        "label_activity_correlation": proxy.label_activity_correlation,
        "sample_count": proxy.sample_count,
    }
