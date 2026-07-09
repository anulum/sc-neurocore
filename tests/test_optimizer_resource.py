# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.optimizer (resource optimizer)

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.optimizer import fit_to_target, OptimizationResult
from sc_neurocore.optimizer.resource_optimizer import OptimizationStep


class TestOptimizationResult:
    def test_summary(self):
        r = OptimizationResult(
            fits=True,
            target="ice40",
            final_luts=1000,
            target_luts=5000,
            utilization_pct=20.0,
            final_bitstream_length=64,
            final_sparsity=0.3,
            steps=[
                OptimizationStep(
                    action="Reduce L to 128",
                    luts_before=6000,
                    luts_after=3000,
                    sparsity=0.0,
                    bitstream_length=128,
                ),
            ],
        )
        s = r.summary()
        assert "ice40" in s
        assert "YES" in s
        assert "1,000" in s

    def test_summary_does_not_fit(self):
        r = OptimizationResult(
            fits=False,
            target="ice40",
            final_luts=9000,
            target_luts=5000,
            utilization_pct=180.0,
            final_bitstream_length=256,
            final_sparsity=0.0,
        )
        s = r.summary()
        assert "NO" in s


class TestFitToTarget:
    def test_small_network_fits(self):
        layers = [(4, 2)]
        weights = [np.random.randn(2, 4) * 0.1]
        result = fit_to_target(layers, weights, target="artix7")
        assert isinstance(result, OptimizationResult)
        assert result.fits is True

    def test_ice40_may_need_optimization(self):
        layers = [(64, 32), (32, 16)]
        weights = [np.random.randn(32, 64), np.random.randn(16, 32)]
        result = fit_to_target(layers, weights, target="ice40", initial_bitstream_length=256)
        assert isinstance(result, OptimizationResult)
        assert result.target == "ice40"

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            fit_to_target([(4, 2)], [np.random.randn(2, 4)], target="nonexistent_fpga")

    def test_steps_recorded(self):
        layers = [(32, 16)]
        weights = [np.random.randn(16, 32)]
        result = fit_to_target(layers, weights, target="ice40", initial_bitstream_length=512)
        assert isinstance(result.steps, list)
        for step in result.steps:
            assert isinstance(step, OptimizationStep)

    def test_sparsity_range(self):
        layers = [(16, 8)]
        weights = [np.random.randn(8, 16)]
        result = fit_to_target(layers, weights, target="ice40")
        assert 0.0 <= result.final_sparsity <= 1.0

    def test_optimized_weights_returned(self):
        layers = [(8, 4)]
        weights = [np.random.randn(4, 8)]
        result = fit_to_target(layers, weights, target="ecp5")
        assert len(result.optimized_weights) == 1
        assert result.optimized_weights[0].shape == (4, 8)

    def test_prune_and_quantize_path_when_l_reduction_disabled(self):
        layers = [(64, 32), (32, 16)]
        weights = [np.random.randn(32, 64), np.random.randn(16, 32)]
        result = fit_to_target(
            layers,
            weights,
            target="ice40",
            max_iterations=3,
            min_bitstream_length=256,
            initial_bitstream_length=256,
        )
        actions = [step.action for step in result.steps]
        assert any(action.startswith("Prune threshold=") for action in actions)
        assert any(action.startswith("Quantize to ") for action in actions)
