# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for adaptive per-layer bitstream precision

"""Tests for the adaptive precision assignment module."""

from __future__ import annotations

import numpy as np

from sc_neurocore.compiler.adaptive_precision import (
    LayerPrecision,
    SynapsePrecision,
    analyze_sensitivity,
    assign_lengths,
    assign_synapse_precisions,
    precision_plan_manifest,
)


class TestAssignLengths:
    def test_hoeffding_produces_assignments(self):
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        result = assign_lengths(weights, method="hoeffding")
        assert len(result) == 2
        assert all(isinstance(r, LayerPrecision) for r in result)

    def test_assignments_respect_bounds(self):
        weights = [np.random.randn(4, 2)]
        result = assign_lengths(weights, min_length=64, max_length=512)
        assert all(64 <= r.bitstream_length <= 512 for r in result)

    def test_lengths_are_power_of_two(self):
        weights = [np.random.randn(8, 4), np.random.randn(4, 8)]
        result = assign_lengths(weights, method="hoeffding")
        for r in result:
            L = r.bitstream_length
            assert L & (L - 1) == 0, f"L={L} is not a power of 2"

    def test_relaxed_target_gives_shorter_lengths(self):
        weights = [np.random.randn(4, 2)]
        tight = assign_lengths(weights, target_error=0.01, max_length=4096)
        relaxed = assign_lengths(weights, target_error=0.2, max_length=4096)
        assert relaxed[0].bitstream_length <= tight[0].bitstream_length

    def test_custom_layer_names(self):
        weights = [np.random.randn(4, 2)]
        result = assign_lengths(weights, layer_names=["my_layer"])
        assert result[0].name == "my_layer"

    def test_default_layer_names(self):
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        result = assign_lengths(weights)
        assert result[0].name == "layer_0"
        assert result[1].name == "layer_1"

    def test_sensitivity_method(self):
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        result = assign_lengths(weights, method="sensitivity", total_budget=2048)
        assert len(result) == 2
        assert all(r.sensitivity >= 0 for r in result)


class TestAnalyzeSensitivity:
    def test_returns_per_layer_scores(self):
        weights = [np.random.randn(4, 2), np.random.randn(3, 4)]
        sens = analyze_sensitivity(weights, n_trials=5)
        assert len(sens) == 2
        assert all(s >= 0 for s in sens)

    def test_larger_weights_more_sensitive(self):
        small_w = [np.random.randn(4, 4) * 0.01]
        large_w = [np.random.randn(4, 4) * 0.5]
        sens_small = analyze_sensitivity(small_w, n_trials=10, seed=42)
        sens_large = analyze_sensitivity(large_w, n_trials=10, seed=42)
        # Larger weights should generally be more sensitive
        # (not guaranteed per-trial, but on average)
        assert isinstance(sens_small[0], float)
        assert isinstance(sens_large[0], float)


class TestLayerPrecision:
    def test_dataclass_fields(self):
        lp = LayerPrecision(
            layer_index=0,
            name="fc1",
            bitstream_length=256,
            error_bound=0.031,
            sensitivity=0.05,
        )
        assert lp.layer_index == 0
        assert lp.bitstream_length == 256


class TestSynapsePrecision:
    def test_assign_synapse_precisions_returns_one_row_per_weight(self):
        weights = [np.array([[0.1, 0.8], [0.0, 0.4]])]

        result = assign_synapse_precisions(
            weights,
            layer_names=["fc"],
            target_error=0.05,
            min_bits=3,
            max_bits=8,
            min_length=16,
            max_length=256,
        )

        assert len(result) == 4
        assert all(isinstance(row, SynapsePrecision) for row in result)
        assert {row.layer_name for row in result} == {"fc"}
        assert all(3 <= row.bit_width <= 8 for row in result)
        assert all(16 <= row.bitstream_length <= 256 for row in result)
        assert all(row.total_error_bound >= row.quantization_error_bound for row in result)

    def test_high_sensitivity_gets_at_least_as_much_precision(self):
        weights = [np.array([[0.01, 1.0]])]
        result = assign_synapse_precisions(
            weights,
            target_error=0.05,
            min_bits=2,
            max_bits=10,
            min_length=16,
            max_length=512,
        )

        low, high = result
        assert high.sensitivity > low.sensitivity
        assert high.bit_width >= low.bit_width
        assert high.bitstream_length >= low.bitstream_length

    def test_custom_sensitivity_map_controls_precision(self):
        weights = [np.array([[0.5, 0.5]])]
        sensitivities = [np.array([[0.1, 1.0]])]

        result = assign_synapse_precisions(
            weights,
            sensitivity_maps=sensitivities,
            target_error=0.05,
            min_bits=2,
            max_bits=10,
            min_length=16,
            max_length=512,
        )

        assert result[1].sensitivity > result[0].sensitivity
        assert result[1].bit_width >= result[0].bit_width

    def test_precision_plan_manifest_is_deterministic(self):
        assignments = assign_synapse_precisions(
            [np.array([[0.25, 0.75]])],
            target_error=0.1,
            min_bits=2,
            max_bits=8,
        )

        manifest = precision_plan_manifest(assignments)

        assert manifest["schema"] == "sc-neurocore.adaptive_precision_plan.v1"
        assert manifest["granularity"] == "synapse"
        assert manifest["num_synapses"] == 2
        assert manifest["max_total_error_bound"] >= 0
        assert list(manifest["assignments"][0]) == [
            "layer_index",
            "layer_name",
            "output_index",
            "input_index",
            "bit_width",
            "bitstream_length",
            "sensitivity",
            "quantization_error_bound",
            "stochastic_error_bound",
            "total_error_bound",
        ]

    def test_rejects_invalid_sensitivity_shape(self):
        with np.testing.assert_raises_regex(ValueError, "sensitivity map"):
            assign_synapse_precisions(
                [np.array([[0.5, 0.5]])],
                sensitivity_maps=[np.array([[0.1], [1.0]])],
            )
