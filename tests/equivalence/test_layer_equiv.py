"""Equivalence: v2 VectorizedSCLayer.forward vs v3 DenseLayer."""

import numpy as np
import pytest

from sc_neurocore.layers import VectorizedSCLayer as V2Layer
from sc_neurocore_engine.layers import VectorizedSCLayer as V3Layer


class TestDenseLayerEquivalence:
    """
    Both v2 and v3 use Bernoulli sampling to encode probabilities as bitstreams.
    They use different RNG implementations (NumPy vs ChaCha8), so outputs are
    compared for statistical equivalence, not bit-identical streams.
    """

    @pytest.mark.parametrize(
        "n_inputs,n_neurons",
        [
            (4, 2),
            (16, 8),
            (32, 16),
            (64, 32),
        ],
    )
    def test_forward_statistical_equivalence(self, n_inputs, n_neurons):
        # Long bitstreams reduce stochastic variance enough for a tight tolerance.
        length = 32768

        # v2 uses NumPy global RNG internally; seed for deterministic tests.
        np.random.seed(7)
        v2 = V2Layer(
            n_inputs=n_inputs,
            n_neurons=n_neurons,
            length=length,
            use_gpu=False,
        )
        v3 = V3Layer(
            n_inputs=n_inputs,
            n_neurons=n_neurons,
            length=length,
        )

        # Use v2's weights inside v3 for apples-to-apples comparison.
        v3._engine.set_weights(v2.weights.tolist())
        v3._engine.refresh_packed_weights()

        rng = np.random.RandomState(42)
        inputs = rng.uniform(0.1, 0.9, n_inputs)

        # v2 forward also uses global RNG.
        np.random.seed(123)
        v2_out = v2.forward(inputs)
        v3_out = v3.forward(inputs)

        np.testing.assert_allclose(
            v2_out,
            v3_out,
            atol=0.05,
            err_msg="Dense layer outputs diverge beyond tolerance",
        )

    def test_output_shape(self):
        v3 = V3Layer(n_inputs=8, n_neurons=4, length=1024)
        inputs = np.full(8, 0.5)
        out = v3.forward(inputs)
        assert out.shape == (4,)
        # Current layer semantics are sum_i (w_ji * p_i), so range is [0, n_inputs].
        assert np.all(out >= 0.0) and np.all(out <= v3.n_inputs)

    def test_deterministic_with_same_seed(self):
        """Two v3 layers with the same default seed should match exactly."""
        v3a = V3Layer(n_inputs=8, n_neurons=4, length=1024)
        v3b = V3Layer(n_inputs=8, n_neurons=4, length=1024)
        inputs = np.full(8, 0.5)
        np.testing.assert_array_equal(v3a.forward(inputs), v3b.forward(inputs))
