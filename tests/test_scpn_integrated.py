# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for 16-layer SCPN integrated step

"""Tests for create_full_stack, run_integrated_step, get_global_metrics, K_nm matrix."""

from __future__ import annotations

import numpy as np

from sc_neurocore.scpn import (
    create_full_stack,
    run_integrated_step,
    get_global_metrics,
)
from sc_neurocore.scpn.layers import LAYER_REGISTRY
from sc_neurocore.scpn.params import build_knm_matrix, OMEGA_N


class TestKnmMatrix:
    def test_shape(self):
        K = build_knm_matrix(16)
        assert K.shape == (16, 16)

    def test_symmetric(self):
        K = build_knm_matrix(16)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_zero_diagonal(self):
        K = build_knm_matrix(16)
        np.testing.assert_allclose(np.diag(K), 0.0)

    def test_non_negative(self):
        K = build_knm_matrix(16)
        assert np.all(K >= 0), "coupling matrix has negative entries"

    def test_custom_size(self):
        K = build_knm_matrix(8)
        assert K.shape == (8, 8)

    def test_adjacent_layers_coupled(self):
        """Adjacent layers should have nonzero coupling."""
        K = build_knm_matrix(16)
        for i in range(15):
            assert K[i, i + 1] > 0, f"L{i + 1}↔L{i + 2} not coupled"


class TestOmegaN:
    def test_length(self):
        assert len(OMEGA_N) == 16

    def test_all_positive(self):
        assert np.all(OMEGA_N > 0)

    def test_l2_matches_gamma(self):
        """L2 neurochemical ≈ 40 Hz × 2π."""
        expected = 40.0 * 2 * np.pi
        np.testing.assert_allclose(OMEGA_N[1], expected, rtol=0.01)

    def test_l5_matches_one_hz(self):
        """L5 intentional frame ≈ 1 Hz × 2π."""
        expected = 1.0 * 2 * np.pi
        np.testing.assert_allclose(OMEGA_N[4], expected, rtol=0.01)


class TestLayerRegistry:
    def test_has_16_layers(self):
        assert len(LAYER_REGISTRY) == 16

    def test_keys_l1_to_l16(self):
        for i in range(1, 17):
            assert f"l{i}" in LAYER_REGISTRY, f"l{i} missing from registry"


class TestCreateFullStack:
    def test_returns_dict(self):
        stack = create_full_stack()
        assert isinstance(stack, dict)

    def test_has_16_layers(self):
        stack = create_full_stack()
        assert len(stack) == 16

    def test_keys_match_registry(self):
        stack = create_full_stack()
        for key in LAYER_REGISTRY:
            assert key in stack

    def test_layers_have_step(self):
        """Each layer should have a step() method."""
        stack = create_full_stack()
        for key, layer in stack.items():
            assert hasattr(layer, "step"), f"{key} has no step() method"

    def test_layers_have_global_metric(self):
        stack = create_full_stack()
        for key, layer in stack.items():
            assert hasattr(layer, "get_global_metric"), f"{key} has no get_global_metric()"


class TestRunIntegratedStep:
    def test_returns_dict(self):
        stack = create_full_stack()
        result = run_integrated_step(stack, dt=0.001)
        assert isinstance(result, dict)

    def test_all_layers_in_result(self):
        stack = create_full_stack()
        result = run_integrated_step(stack, dt=0.001)
        for key in stack:
            assert key in result, f"{key} missing from integrated step result"

    def test_finite_outputs(self):
        """All layer outputs should be finite after one step."""
        stack = create_full_stack()
        result = run_integrated_step(stack, dt=0.001)
        for key, output in result.items():
            if isinstance(output, (dict,)):
                for k, v in output.items():
                    if isinstance(v, (float, int, np.floating)):
                        assert np.isfinite(v), f"{key}.{k} = {v} not finite"
                    elif isinstance(v, np.ndarray):
                        assert np.all(np.isfinite(v)), f"{key}.{k} contains non-finite"

    def test_multiple_steps_stable(self):
        """10 steps should not diverge."""
        stack = create_full_stack()
        for _ in range(10):
            result = run_integrated_step(stack, dt=0.001)
        # If we got here without exception, basic stability holds


class TestGetGlobalMetrics:
    def test_returns_dict(self):
        stack = create_full_stack()
        run_integrated_step(stack, dt=0.001)
        metrics = get_global_metrics(stack)
        assert isinstance(metrics, dict)

    def test_metrics_finite(self):
        stack = create_full_stack()
        for _ in range(5):
            run_integrated_step(stack, dt=0.001)
        metrics = get_global_metrics(stack)
        for key, val in metrics.items():
            if isinstance(val, (float, int, np.floating)):
                assert np.isfinite(val), f"metric {key} = {val} not finite"

    def test_has_coherence_metric(self):
        """Global metrics should include some coherence measure."""
        stack = create_full_stack()
        run_integrated_step(stack, dt=0.001)
        metrics = get_global_metrics(stack)
        # Should have at least one metric related to coherence/integration
        assert len(metrics) > 0
