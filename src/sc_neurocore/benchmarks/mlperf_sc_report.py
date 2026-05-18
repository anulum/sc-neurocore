# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MLPerf-SC result aggregation

"""Deterministic aggregation for validated MLPerf-SC result records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .mlperf_sc_schema import MLPerfSCResult, MLPerfSCValidationError, validate_mlperf_sc_result

MLPERF_SC_REPORT_SCHEMA_VERSION = "sc-neurocore.mlperf-sc-report.v0.1"


def aggregate_mlperf_sc_results(
    result_paths: Sequence[str | Path],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Aggregate validated MLPerf-SC result files into a deterministic report."""

    if not result_paths:
        raise MLPerfSCValidationError("MLPerf-SC aggregation requires at least one result")
    output = Path(output_path)
    rows: list[dict[str, Any]] = []
    typed_results: list[MLPerfSCResult] = []
    for path_value in result_paths:
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = validate_mlperf_sc_result(payload, artifact_root=path.parent)
        typed_results.append(result)
        rows.append(
            {
                "run_id": result.run.run_id,
                "task": result.run.task,
                "model": result.run.model,
                "dataset": result.run.dataset,
                "producer": result.run.producer,
                "backend": result.execution.backend,
                "target": result.execution.target,
                "sc_mode": result.execution.sc_mode,
                "bitstream_length": result.execution.bitstream_length,
                "evidence_class": result.evidence.evidence_class,
                "accuracy": result.metrics.accuracy,
                "latency_ms": result.metrics.latency_ms,
                "throughput_inferences_per_s": result.metrics.throughput_inferences_per_s,
                "energy_j_per_inference": result.metrics.energy_j_per_inference,
                "power_w": result.metrics.power_w,
            }
        )
    rows.sort(key=lambda row: row["run_id"])
    accuracies = [result.metrics.accuracy for result in typed_results]
    latencies = [result.metrics.latency_ms for result in typed_results]
    throughputs = [
        result.metrics.throughput_inferences_per_s
        for result in typed_results
        if result.metrics.throughput_inferences_per_s is not None
    ]
    report = {
        "schema_version": MLPERF_SC_REPORT_SCHEMA_VERSION,
        "summary": {
            "result_count": len(typed_results),
            "tasks": sorted({result.run.task for result in typed_results}),
            "models": sorted({result.run.model for result in typed_results}),
            "evidence_classes": sorted(
                {result.evidence.evidence_class for result in typed_results}
            ),
            "best_accuracy": max(accuracies),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "min_latency_ms": min(latencies),
            "max_throughput_inferences_per_s": max(throughputs) if throughputs else None,
        },
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
