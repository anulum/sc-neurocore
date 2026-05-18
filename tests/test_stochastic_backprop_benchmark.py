# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Tests for stochastic backpropagation benchmark evidence

"""Tests for reproducible stochastic backpropagation benchmark evidence."""

from __future__ import annotations

import json

from sc_neurocore.benchmarks.stochastic_backprop import (
    STOCHASTIC_BACKPROP_BENCHMARK_SCHEMA_VERSION,
    build_stochastic_backprop_benchmark,
    write_stochastic_backprop_benchmark,
)


def test_stochastic_backprop_benchmark_reports_loss_and_stream_evidence() -> None:
    report = build_stochastic_backprop_benchmark(bitstream_length=256, steps=32, learning_rate=0.4)

    assert report["schema_version"] == STOCHASTIC_BACKPROP_BENCHMARK_SCHEMA_VERSION
    assert report["evidence_class"] == "deterministic_training_simulation"
    assert report["hardware_measurement_claimed"] is False
    assert report["sc_config"]["generator"] == "sobol"
    assert report["sc_config"]["bitstream_length"] == 256
    assert report["loss"]["final"] < report["loss"]["initial"] * 0.5
    assert report["loss"]["best"] <= report["loss"]["final"]
    assert report["stream_evidence"]["sampled_product_mae"] < 0.08
    assert 0.0 <= report["stream_evidence"]["input_max_abs_correlation"] <= 1.0
    assert 0.0 <= report["stream_evidence"]["weight_max_abs_correlation"] <= 1.0
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
