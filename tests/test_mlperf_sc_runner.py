# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MLPerf-SC fixture runner

"""Contract tests for the low-load MLPerf-SC local fixture runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sc_neurocore.benchmarks import (
    MLPERF_SC_RESULT_SCHEMA_VERSION,
    MLPerfSCValidationError,
    run_mlperf_sc_fixture,
    validate_mlperf_sc_result,
)


def test_mlperf_sc_fixture_runner_writes_valid_result_and_raw_artifact(
    tmp_path: Path,
) -> None:
    result_path = run_mlperf_sc_fixture(
        output_dir=tmp_path,
        task="synthetic_sc_xor",
        model="fixture_sc_linear",
        seed=7,
        bitstream_length=64,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = validate_mlperf_sc_result(payload, artifact_root=tmp_path)

    assert payload["schema_version"] == MLPERF_SC_RESULT_SCHEMA_VERSION
    assert result.run.run_id == "synthetic_sc_xor-fixture_sc_linear-seed7-l64"
    assert result.execution.sc_mode == "bipolar"
    assert result.execution.bitstream_length == 64
    assert result.metrics.accuracy == pytest.approx(1.0)

    artifacts = {artifact.kind: artifact for artifact in result.evidence.artifacts}
    raw_path = tmp_path / artifacts["raw_results"].path
    environment_path = tmp_path / artifacts["environment_manifest"].path
    assert raw_path.is_file()
    assert environment_path.is_file()
    assert artifacts["raw_results"].sha256 == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert (
        artifacts["environment_manifest"].sha256
        == hashlib.sha256(environment_path.read_bytes()).hexdigest()
    )


def test_mlperf_sc_fixture_runner_is_deterministic(tmp_path: Path) -> None:
    first = run_mlperf_sc_fixture(
        output_dir=tmp_path / "a",
        task="synthetic_sc_xor",
        model="fixture_sc_linear",
        seed=11,
        bitstream_length=128,
    )
    second = run_mlperf_sc_fixture(
        output_dir=tmp_path / "b",
        task="synthetic_sc_xor",
        model="fixture_sc_linear",
        seed=11,
        bitstream_length=128,
    )

    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))

    assert first_payload["metrics"] == second_payload["metrics"]
    assert (
        first_payload["evidence"]["artifacts"][0]["sha256"]
        == second_payload["evidence"]["artifacts"][0]["sha256"]
    )


def test_mlperf_sc_fixture_runner_writes_external_reference_boundary(
    tmp_path: Path,
) -> None:
    result_path = run_mlperf_sc_fixture(
        output_dir=tmp_path,
        task="synthetic_sc_xor",
        model="fixture_external_majority",
        seed=13,
        bitstream_length=64,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = validate_mlperf_sc_result(payload, artifact_root=tmp_path)
    raw_artifact = next(
        artifact for artifact in result.evidence.artifacts if artifact.kind == "raw_results"
    )
    raw_payload = json.loads((tmp_path / raw_artifact.path).read_text(encoding="utf-8"))

    assert result.run.producer == "external-reference-fixture"
    assert result.run.model == "fixture_external_majority"
    assert result.execution.sc_mode == "deterministic_replay"
    assert result.metrics.accuracy == pytest.approx(0.5)
    assert raw_payload["baseline_family"] == "external_reference"
    assert raw_payload["evidence_boundary"] == "deterministic fixture, not measured hardware"


def test_mlperf_sc_fixture_runner_rejects_invalid_contract(tmp_path: Path) -> None:
    with pytest.raises(MLPerfSCValidationError, match="bitstream_length"):
        run_mlperf_sc_fixture(
            output_dir=tmp_path,
            task="synthetic_sc_xor",
            model="fixture_sc_linear",
            seed=1,
            bitstream_length=0,
        )


def test_mlperf_sc_fixture_runner_rejects_unknown_fixture_model(tmp_path: Path) -> None:
    with pytest.raises(MLPerfSCValidationError, match="fixture runner"):
        run_mlperf_sc_fixture(
            output_dir=tmp_path,
            task="synthetic_sc_xor",
            model="unknown_baseline",
            seed=1,
            bitstream_length=64,
        )


def test_mlperf_sc_fixture_runner_rejects_negative_seed(tmp_path: Path) -> None:
    with pytest.raises(MLPerfSCValidationError, match="seed must be non-negative"):
        run_mlperf_sc_fixture(output_dir=tmp_path, seed=-1)


def test_mlperf_sc_fixture_runner_rejects_unsupported_task(tmp_path: Path) -> None:
    with pytest.raises(MLPerfSCValidationError, match="supports synthetic_sc_xor"):
        run_mlperf_sc_fixture(output_dir=tmp_path, task="image_classification")


def test_fixture_baseline_rejects_unknown_model() -> None:
    from sc_neurocore.benchmarks.mlperf_sc_runner import _fixture_baseline

    with pytest.raises(MLPerfSCValidationError, match="fixture_sc_linear"):
        _fixture_baseline("not_a_fixture_model")
