# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MLPerf-SC report aggregation

"""Contract tests for MLPerf-SC aggregation over validated result records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sc_neurocore.benchmarks import (
    MLPerfSCValidationError,
    aggregate_mlperf_sc_results,
    run_mlperf_sc_fixture,
)


def test_mlperf_sc_aggregation_writes_deterministic_report(tmp_path: Path) -> None:
    first = run_mlperf_sc_fixture(
        output_dir=tmp_path / "seed1",
        seed=1,
        bitstream_length=64,
    )
    second = run_mlperf_sc_fixture(
        output_dir=tmp_path / "seed2",
        seed=2,
        bitstream_length=128,
    )
    report_path = tmp_path / "mlperf_sc_report.json"

    report = aggregate_mlperf_sc_results(
        [first, second],
        output_path=report_path,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload == report
    assert payload["schema_version"] == "sc-neurocore.mlperf-sc-report.v0.1"
    assert payload["summary"] == {
        "result_count": 2,
        "tasks": ["synthetic_sc_xor"],
        "models": ["fixture_sc_linear"],
        "evidence_classes": ["simulation"],
        "best_accuracy": 1.0,
        "mean_accuracy": 1.0,
        "min_latency_ms": 0.000256,
        "max_throughput_inferences_per_s": 3906250.0,
    }
    assert [row["run_id"] for row in payload["results"]] == [
        "synthetic_sc_xor-fixture_sc_linear-seed1-l64",
        "synthetic_sc_xor-fixture_sc_linear-seed2-l128",
    ]


def test_mlperf_sc_aggregation_compares_sc_neurocore_and_external_baseline(
    tmp_path: Path,
) -> None:
    sc_result = run_mlperf_sc_fixture(
        output_dir=tmp_path / "sc",
        model="fixture_sc_linear",
        seed=1,
        bitstream_length=64,
    )
    external_result = run_mlperf_sc_fixture(
        output_dir=tmp_path / "external",
        model="fixture_external_majority",
        seed=1,
        bitstream_length=64,
    )

    report = aggregate_mlperf_sc_results(
        [external_result, sc_result],
        output_path=tmp_path / "report.json",
    )

    assert report["summary"]["models"] == [
        "fixture_external_majority",
        "fixture_sc_linear",
    ]
    assert report["summary"]["best_accuracy"] == pytest.approx(1.0)
    assert report["summary"]["mean_accuracy"] == pytest.approx(0.75)
    assert [row["producer"] for row in report["results"]] == [
        "external-reference-fixture",
        "sc-neurocore",
    ]
    assert {row["evidence_class"] for row in report["results"]} == {"simulation"}


def test_mlperf_sc_aggregation_rejects_empty_or_invalid_result_set(
    tmp_path: Path,
) -> None:
    with pytest.raises(MLPerfSCValidationError, match="at least one"):
        aggregate_mlperf_sc_results([], output_path=tmp_path / "empty.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            {
                "schema_version": "sc-neurocore.mlperf-sc-result.v0.1",
                "run": {},
                "execution": {},
                "metrics": {},
                "evidence": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MLPerfSCValidationError, match="run missing"):
        aggregate_mlperf_sc_results([invalid], output_path=tmp_path / "invalid-report.json")
    assert not (tmp_path / "invalid-report.json").exists()
