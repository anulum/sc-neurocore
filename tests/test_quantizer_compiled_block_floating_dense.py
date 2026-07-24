# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompiledBlockFloatingDense from former test_quantizer.py

"""Focused suite: TestCompiledBlockFloatingDense from former test_quantizer.py."""

from __future__ import annotations

from tests.quantizer_support import *  # noqa: F403


class TestCompiledBlockFloatingDense:
    """Validate dense block-floating weights with fixed-point inputs."""

    def test_block_floating_dense_matches_reconstructed_matrix_dot(self):
        weights = np.array([[0.5, -0.25], [1.25, 0.125]], dtype=np.float64)
        inputs = np.array([0.5, -0.25], dtype=np.float64)
        compiled = compile_dense_block_floating(weights, fmt="BFP16E3X2")

        assert compiled.manifest()["operation"] == "dense_block_floating"
        assert compiled.manifest()["weight_shape"] == [2, 2]
        assert compiled.manifest()["parameter_count"] == 4
        assert compiled.manifest()["block_exponent_count"] == 2
        assert compiled.manifest()["block_exponent_layout"]["last_block_size"] == 2

        q_inputs = quantize_weights(inputs, fmt=Q16_16).astype(np.float64) / Q16_16.scale
        expected = compiled.reconstructed_weights @ q_inputs
        np.testing.assert_allclose(compiled.forward_float(inputs), expected, rtol=0.0, atol=1e-12)

        envelope = compiled.precision_envelope_report(inputs)
        assert envelope.observed_overflow_free is True
        assert envelope.conservative_overflow_free is True
        assert envelope.max_abs_bound_code == 43008

    def test_block_floating_dense_seeded_exponent_edges_match_reference(self):
        weights = np.array(
            [
                [0.125, -0.25, 1_000_000.0, -1_000_000.0],
                [-0.375, 0.5, -1_000_000.0, 1_000_000.0],
            ],
            dtype=np.float64,
        )
        inputs = np.array([0.5, -0.25, 1 / Q16_16.scale, -1 / Q16_16.scale])
        compiled = compile_dense_block_floating(weights, fmt="BFP16E3X2")

        assert compiled.exponents.tolist() == [0, 7, 0, 7]
        assert compiled.manifest()["exponent_code_range"] == [0, 7]

        q_inputs = quantize_weights(inputs, fmt=Q16_16).astype(np.float64) / Q16_16.scale
        reference = compiled.reconstructed_weights @ q_inputs
        np.testing.assert_allclose(compiled.forward_float(inputs), reference, rtol=0.0, atol=0.0)

        codes, overflow = compiled.forward_with_overflow(inputs)
        np.testing.assert_array_equal(codes, np.array([1_056_736, -1_069_024], dtype=np.int64))
        assert overflow.tolist() == [False, False]

        envelope = compiled.precision_envelope_report(inputs)
        assert envelope.observed_overflow_free is True
        assert envelope.observed_underflow_free is True
        assert envelope.conservative_overflow_free is True
        assert envelope.max_abs_bound_code == 1_069_024
        assert envelope.min_headroom_code == 2_146_414_623

    def test_block_floating_dense_max_exponent_edge_saturates_and_reports_trap(self):
        weights = np.array([[1_000_000.0, 1_000_000.0]], dtype=np.float64)
        inputs = np.array([32767.0, 32767.0], dtype=np.float64)
        compiled = compile_dense_block_floating(weights, fmt="BFP16E3X2")

        assert compiled.exponents.tolist() == [7]

        codes, overflow = compiled.forward_with_overflow(inputs)
        assert int(codes[0]) == (1 << (Q16_16.total_bits - 1)) - 1
        assert overflow.tolist() == [True]

        report = compiled.precision_trap_report(inputs)
        assert report.overflow_count == 1
        assert report.underflow_count == 0
        assert report.saturated_max_count == 1

        envelope = compiled.precision_envelope_report(inputs)
        assert envelope.observed_overflow_free is False
        assert envelope.observed_underflow_free is True
        assert envelope.conservative_overflow_free is False
        assert envelope.max_abs_bound_code > envelope.conservative_safe_bound_code

    def test_block_floating_dense_saturates_outputs(self):
        weights = np.array([[8192.0, 8192.0]], dtype=np.float64)
        inputs = np.array([32767.0, 32767.0], dtype=np.float64)
        compiled = compile_dense_block_floating(weights, fmt="BFP16E3X2")

        codes, overflow = compiled.forward_with_overflow(inputs)

        assert overflow.tolist() == [True]
        assert int(codes[0]) == (1 << (Q16_16.total_bits - 1)) - 1

        report = compiled.precision_trap_report(inputs)
        assert report.manifest()["operation"] == "dense_block_floating"
        assert report.overflow_count == 1
        assert report.saturated_max_count == 1
        assert report.saturated_min_count == 0

        envelope = compiled.precision_envelope_report(inputs)
        assert envelope.observed_overflow_free is False
        assert envelope.conservative_overflow_free is False
        assert envelope.max_abs_bound_code > envelope.conservative_safe_bound_code

    def test_block_floating_dense_reports_sub_lsb_underflow(self):
        weights = np.array([[0.125]], dtype=np.float64)
        inputs = np.array([1 / Q16_16.scale], dtype=np.float64)
        compiled = compile_dense_block_floating(weights, fmt="BFP16E3X1")

        codes, overflow = compiled.forward_with_overflow(inputs)

        assert int(codes[0]) == 0
        assert overflow.tolist() == [False]

        report = compiled.precision_trap_report(inputs)
        assert report.overflow_count == 0
        assert report.underflow_count == 1

        envelope = compiled.precision_envelope_report(inputs)
        assert envelope.observed_overflow_free is True
        assert envelope.observed_underflow_free is False

    def test_block_floating_dense_rejects_invalid_shapes_and_inputs(self):
        with pytest.raises(ValueError, match="2-D dense"):
            compile_dense_block_floating(np.array([0.5, -0.25]))
        compiled = compile_dense_block_floating(np.array([[0.5, -0.25]], dtype=np.float64))
        with pytest.raises(ValueError, match="input length mismatch"):
            compiled.forward_float(np.array([0.5]))
        with pytest.raises(ValueError, match="finite values"):
            compiled.forward_float(np.array([0.5, np.inf]))
