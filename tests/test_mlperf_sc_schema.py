# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MLPerf-SC result schema validation

"""Contract tests for the MLPerf-SC result schema foundation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sc_neurocore.benchmarks import (
    MLPERF_SC_RESULT_SCHEMA_VERSION,
    MLPerfSCValidationError,
    mlperf_sc_result_to_dict,
    validate_mlperf_sc_result,
)


def _simulation_payload() -> dict[str, object]:
    return {
        "schema_version": MLPERF_SC_RESULT_SCHEMA_VERSION,
        "run": {
            "run_id": "shd-simulation-seed17",
            "task": "shd",
            "model": "dense-lif-frontier",
            "dataset": "shd",
            "started_at": "2026-05-18T03:20:00+02:00",
            "producer": "sc-neurocore",
        },
        "execution": {
            "backend": "python",
            "target": "cpu",
            "sc_mode": "bipolar",
            "bitstream_length": 1024,
            "seed": 17,
        },
        "metrics": {
            "accuracy": 0.869274,
            "latency_ms": 1.25,
            "throughput_inferences_per_s": 800.0,
            "energy_j_per_inference": None,
            "power_w": None,
            "area": {
                "luts": None,
                "ffs": None,
                "bram": None,
                "dsp": None,
            },
        },
        "evidence": {
            "evidence_class": "simulation",
            "environment": {
                "python": "3.12.0",
                "platform": "linux-x86_64",
            },
            "artifacts": [
                {
                    "kind": "raw_results",
                    "path": "runs/shd-simulation-seed17.json",
                    "sha256": "a" * 64,
                }
            ],
        },
    }


def test_mlperf_sc_accepts_valid_simulation_payload(tmp_path: Path) -> None:
    artifact = tmp_path / "runs" / "shd-simulation-seed17.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")

    result = validate_mlperf_sc_result(_simulation_payload(), artifact_root=tmp_path)

    assert result.run.task == "shd"
    assert result.execution.sc_mode == "bipolar"
    assert result.metrics.accuracy == pytest.approx(0.869274)
    assert mlperf_sc_result_to_dict(result) == _simulation_payload()


def test_mlperf_sc_schema_resource_matches_runtime_version() -> None:
    schema_path = Path("schemas/mlperf_sc/mlperf_sc_result.schema.json")

    payload = json.loads(schema_path.read_text(encoding="utf-8"))

    assert payload["$id"] == MLPERF_SC_RESULT_SCHEMA_VERSION
    assert payload["required"] == [
        "schema_version",
        "run",
        "execution",
        "metrics",
        "evidence",
    ]


def test_mlperf_sc_accepts_board_measurement_only_with_raw_artifacts(
    tmp_path: Path,
) -> None:
    payload = _simulation_payload()
    payload["evidence"] = {
        "evidence_class": "board_measurement",
        "environment": {"board": "PYNQ-Z2", "toolchain": "Vivado 2025.2"},
        "artifacts": [
            {
                "kind": "board_log",
                "path": "board/run.log",
                "sha256": "b" * 64,
            },
            {
                "kind": "power_trace",
                "path": "board/power.csv",
                "sha256": "c" * 64,
            },
        ],
    }
    payload["metrics"] = {
        **payload["metrics"],  # type: ignore[arg-type]
        "energy_j_per_inference": 1.8e-6,
        "power_w": 0.42,
    }
    for relative in ("board/run.log", "board/power.csv"):
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("measured\n", encoding="utf-8")

    result = validate_mlperf_sc_result(payload, artifact_root=tmp_path)

    assert result.evidence.evidence_class == "board_measurement"
    assert {artifact.kind for artifact in result.evidence.artifacts} == {
        "board_log",
        "power_trace",
    }


def test_mlperf_sc_rejects_physical_claim_without_raw_artifact(tmp_path: Path) -> None:
    payload = _simulation_payload()
    payload["evidence"] = {
        "evidence_class": "board_measurement",
        "environment": {"board": "PYNQ-Z2"},
        "artifacts": [
            {
                "kind": "summary_report",
                "path": "board/summary.json",
                "sha256": "d" * 64,
            }
        ],
    }
    payload["metrics"] = {
        **payload["metrics"],  # type: ignore[arg-type]
        "energy_j_per_inference": 1.8e-6,
    }
    (tmp_path / "board").mkdir()
    (tmp_path / "board" / "summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(MLPerfSCValidationError, match="raw board"):
        validate_mlperf_sc_result(payload, artifact_root=tmp_path)


def test_mlperf_sc_rejects_bad_metrics_evidence_class_and_path_escape(
    tmp_path: Path,
) -> None:
    invalid_metric = _simulation_payload()
    invalid_metric["metrics"] = {
        **invalid_metric["metrics"],  # type: ignore[arg-type]
        "accuracy": 1.5,
    }
    with pytest.raises(MLPerfSCValidationError, match="accuracy"):
        validate_mlperf_sc_result(invalid_metric, artifact_root=tmp_path)

    fabricated = _simulation_payload()
    fabricated["evidence"] = {
        "evidence_class": "fabricated",
        "environment": {"python": "3.12.0"},
        "artifacts": [],
    }
    with pytest.raises(MLPerfSCValidationError, match="evidence_class"):
        validate_mlperf_sc_result(fabricated, artifact_root=tmp_path)

    escaped = _simulation_payload()
    escaped["evidence"] = {
        "evidence_class": "simulation",
        "environment": {"python": "3.12.0"},
        "artifacts": [
            {
                "kind": "raw_results",
                "path": "../outside.json",
                "sha256": "e" * 64,
            }
        ],
    }
    with pytest.raises(MLPerfSCValidationError, match="artifact path"):
        validate_mlperf_sc_result(escaped, artifact_root=tmp_path)
