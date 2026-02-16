"""
Phase 3 tests: chaos RNG JIT, FSM JIT, L4 calcium vectorized, L6 ring coupling vectorized.
"""

import numpy as np
import pytest

from sc_neurocore.chaos.rng import ChaoticRNG, _logistic_map
from sc_neurocore.utils.fsm_activations import TanhFSM, ReLKFSM
from sc_neurocore.scpn.layers.l4_cellular import L4_CellularLayer, L4_StochasticParameters
from sc_neurocore.scpn.layers.l6_ecological import L6_EcologicalLayer, L6_StochasticParameters


class TestChaoticRNGJIT:
    """Verify JIT logistic map matches Python loop."""

    def test_logistic_map_deterministic(self):
        """JIT kernel reproduces the logistic map sequence."""
        r, x0 = 4.0, 0.5
        out_jit, final_jit = _logistic_map(r, x0, 200)

        # Reference
        curr = x0
        ref = np.zeros(200)
        for i in range(200):
            curr = r * curr * (1.0 - curr)
            ref[i] = curr

        np.testing.assert_allclose(out_jit, ref, atol=1e-12)
        assert abs(final_jit - curr) < 1e-12

    def test_chaotic_rng_generates_bitstream(self):
        rng = ChaoticRNG(r=4.0, x=0.3)
        bs = rng.generate_bitstream(0.5, 1000)
        assert bs.shape == (1000,)
        assert set(np.unique(bs)).issubset({0, 1})


class TestFSMJIT:
    """Verify JIT FSM process matches per-step Python loop."""

    def test_tanh_fsm_process_matches_step(self):
        np.random.seed(42)
        bits = (np.random.random(500) > 0.5).astype(np.uint8)

        fsm_ref = TanhFSM(states=16)
        ref_out = np.zeros_like(bits)
        for i, b in enumerate(bits):
            ref_out[i] = fsm_ref.step(b)

        fsm_jit = TanhFSM(states=16)
        jit_out = fsm_jit.process(bits)

        np.testing.assert_array_equal(jit_out, ref_out)

    def test_relk_fsm_process_matches_step(self):
        np.random.seed(42)
        bits = (np.random.random(500) > 0.5).astype(np.uint8)

        fsm_ref = ReLKFSM(states=16)
        ref_out = np.zeros_like(bits)
        for i, b in enumerate(bits):
            ref_out[i] = fsm_ref.step(b)

        fsm_jit = ReLKFSM(states=16)
        jit_out = fsm_jit.process(bits)

        np.testing.assert_array_equal(jit_out, ref_out)


class TestL4CalciumVectorized:
    """Verify vectorized calcium diffusion produces valid output."""

    def test_step_returns_valid_calcium(self):
        np.random.seed(42)
        params = L4_StochasticParameters(grid_size=(5, 5))
        layer = L4_CellularLayer(params)
        result = layer.step(0.01)
        assert result["calcium"].shape == (25,)
        assert np.all(result["calcium"] >= 0) and np.all(result["calcium"] <= 1)


class TestL6RingCouplingVectorized:
    """Verify vectorized ring coupling produces valid output."""

    def test_step_returns_valid_biospheric(self):
        np.random.seed(42)
        params = L6_StochasticParameters(n_field_nodes=32)
        layer = L6_EcologicalLayer(params)
        result = layer.step(0.1)
        assert result["biospheric_field"].shape == (32,)
        assert np.all(result["biospheric_field"] >= 0)
        assert np.all(result["biospheric_field"] <= 1)
