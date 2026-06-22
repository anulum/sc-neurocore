# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for stochastic backpropagation benchmark evidence

"""Tests for reproducible stochastic backpropagation benchmark evidence."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch

from sc_neurocore.benchmarks.stochastic_backprop import (
    STOCHASTIC_BACKPROP_BENCHMARK_SCHEMA_VERSION,
    STOCHASTIC_BACKPROP_ESTIMATOR_REGRESSION_SCHEMA_VERSION,
    STOCHASTIC_BACKPROP_EVIDENCE_BOUNDARY,
    _all_estimator_variances_are_finite_nonnegative,
    _design_length_options,
    _estimator_variance_evidence,
    _validate_bitstream_length_grid,
    build_stochastic_backprop_benchmark,
    build_stochastic_backprop_estimator_regression_manifest,
    write_stochastic_backprop_estimator_regression_manifest,
    write_stochastic_backprop_benchmark,
)


def test_stochastic_backprop_benchmark_reports_loss_and_stream_evidence() -> None:
    report = build_stochastic_backprop_benchmark(bitstream_length=256, steps=32, learning_rate=0.4)

    assert report["schema_version"] == STOCHASTIC_BACKPROP_BENCHMARK_SCHEMA_VERSION
    assert report["evidence_class"] == "deterministic_training_simulation"
    assert report["evidence_boundary"] == STOCHASTIC_BACKPROP_EVIDENCE_BOUNDARY
    assert report["hardware_measurement_claimed"] is False
    assert report["sc_config"]["generator"] == "sobol"
    assert report["sc_config"]["bitstream_length"] == 256
    assert report["joint_design"]["enabled"] is True
    assert report["joint_design"]["final"]["selected_bitstream_length"] == 256
    assert report["joint_design"]["final"]["selected_encoding"] == report["sc_config"]["encoding"]
    assert (
        report["joint_design"]["final"]["expected_bitstream_length"]
        > report["joint_design"]["initial"]["expected_bitstream_length"]
    )
    assert report["loss"]["final"] < report["loss"]["initial"] * 0.5
    assert report["loss"]["best"] <= report["loss"]["final"]
    assert report["stream_evidence"]["sampled_product_mae"] < 0.08
    assert 0.0 <= report["stream_evidence"]["input_max_abs_correlation"] <= 1.0
    assert 0.0 <= report["stream_evidence"]["weight_max_abs_correlation"] <= 1.0
    estimator_variance = report["estimator_variance"]
    assert estimator_variance["sample_count"] >= 16
    assert set(estimator_variance["estimators"]) == {
        "pathwise_relaxation",
        "straight_through",
        "score_function",
    }
    assert estimator_variance["reference"]["estimator"] == "pathwise_relaxation"
    assert estimator_variance["estimators"]["pathwise_relaxation"]["variance"] == 0.0
    assert estimator_variance["estimators"]["straight_through"]["variance"] >= 0.0
    assert estimator_variance["estimators"]["score_function"]["variance"] > 0.0
    assert (
        estimator_variance["estimators"]["score_function"]["variance"]
        > estimator_variance["estimators"]["straight_through"]["variance"]
    )
    assert report["objective_terms"]["length_cost"] > 0.0
    assert report["trained_parameters"]["weight"] != report["initial_parameters"]["weight"]


def test_stochastic_backprop_benchmark_rejects_invalid_contract_values() -> None:
    for kwargs in (
        {"bitstream_length": 0},
        {"steps": 0},
        {"learning_rate": 0.0},
    ):
        try:
            build_stochastic_backprop_benchmark(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_write_stochastic_backprop_benchmark_writes_canonical_json(tmp_path) -> None:
    output = tmp_path / "stochastic_backprop.json"

    path = write_stochastic_backprop_benchmark(
        output,
        bitstream_length=128,
        steps=16,
        learning_rate=0.4,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path == output
    assert payload == build_stochastic_backprop_benchmark(
        bitstream_length=128,
        steps=16,
        learning_rate=0.4,
    )
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_estimator_regression_manifest_covers_multiple_lengths_and_acceptance_gates() -> None:
    manifest = build_stochastic_backprop_estimator_regression_manifest(
        bitstream_lengths=(64, 128, 256),
        sample_count=32,
    )

    assert manifest["schema_version"] == STOCHASTIC_BACKPROP_ESTIMATOR_REGRESSION_SCHEMA_VERSION
    assert manifest["SPDX-License-Identifier"] == "AGPL-3.0-or-later"
    assert manifest["evidence_class"] == "deterministic_estimator_regression"
    assert manifest["evidence_boundary"] == STOCHASTIC_BACKPROP_EVIDENCE_BOUNDARY
    assert manifest["hardware_measurement_claimed"] is False
    assert manifest["status"] == "pass"
    assert manifest["bitstream_lengths"] == [64, 128, 256]
    assert set(manifest["estimators"]) == {
        "pathwise_relaxation",
        "straight_through",
        "score_function",
    }
    score_variances = [
        row["estimators"]["score_function"]["variance"] for row in manifest["results"]
    ]
    assert score_variances[0] > score_variances[-1]
    assert manifest["acceptance"]["score_function_longest_variance_below_shortest"] is True
    assert manifest["acceptance"]["pathwise_variance_zero"] is True


def test_write_estimator_regression_manifest_writes_canonical_json(tmp_path) -> None:
    output = tmp_path / "stochastic_backprop_estimator_regression.json"

    path = write_stochastic_backprop_estimator_regression_manifest(
        output,
        bitstream_lengths=(64, 128),
        sample_count=16,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path == output
    assert payload == build_stochastic_backprop_estimator_regression_manifest(
        bitstream_lengths=(64, 128),
        sample_count=16,
    )
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_estimator_regression_manifest_requires_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="sample_count must be at least two"):
        build_stochastic_backprop_estimator_regression_manifest(sample_count=1)


def test_estimator_variance_evidence_requires_at_least_two_samples() -> None:
    with pytest.raises(ValueError, match="sample_count must be at least two"):
        _estimator_variance_evidence(bitstream_length=64, sample_count=1)


def test_estimator_variance_evidence_raises_when_reference_grad_missing() -> None:
    # Suppressing autograd leaves the reference weight gradient unpopulated,
    # which the defensive guard must surface as a RuntimeError.
    with (
        patch.object(torch.Tensor, "backward", lambda self, *args, **kwargs: None),
        pytest.raises(RuntimeError, match="reference gradient was not populated"),
    ):
        _estimator_variance_evidence(bitstream_length=64, sample_count=2)


def test_design_length_options_collapses_for_minimal_length() -> None:
    # A length-2 grid collapses the distinct-options set below three, falling
    # back to the consecutive (n, n+1, n+2) ladder.
    assert _design_length_options(2) == (2, 3, 4)


def test_validate_bitstream_length_grid_rejects_malformed_grids() -> None:
    with pytest.raises(ValueError, match="at least two entries"):
        _validate_bitstream_length_grid((64,))
    with pytest.raises(ValueError, match="positive integers"):
        _validate_bitstream_length_grid((64, -1))
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_bitstream_length_grid((64, 64))


def test_estimator_variance_finiteness_guard_rejects_bad_variances() -> None:
    non_numeric = {"estimators": {"e": {"variance": "nan"}}}
    assert _all_estimator_variances_are_finite_nonnegative(non_numeric) is False
    non_finite = {"estimators": {"e": {"variance": float("inf")}}}
    assert _all_estimator_variances_are_finite_nonnegative(non_finite) is False
    negative = {"estimators": {"e": {"variance": -1.0}}}
    assert _all_estimator_variances_are_finite_nonnegative(negative) is False
