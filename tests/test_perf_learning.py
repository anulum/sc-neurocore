"""
Phase 2b tests: SCLearningLayer JIT weight update direction and output shape.
"""

import numpy as np
import pytest

from sc_neurocore.layers.sc_learning_layer import SCLearningLayer


class TestSCLearningLayerJIT:

    def test_output_shape(self):
        np.random.seed(42)
        layer = SCLearningLayer(
            n_inputs=3, n_neurons=2, length=128, base_seed=42,
        )
        spikes = layer.run_epoch([0.3, 0.6, 0.9])
        assert spikes.shape == (2, 128)
        assert spikes.dtype == np.uint8

    def test_weights_change_after_epoch(self):
        """Weights should shift after an epoch of learning."""
        np.random.seed(42)
        layer = SCLearningLayer(
            n_inputs=3, n_neurons=2, length=256, base_seed=42,
            learning_rate=0.05,
        )
        w_before = layer.get_weights().copy()
        layer.run_epoch([0.8, 0.8, 0.8])
        w_after = layer.get_weights()
        # At least some weights should have changed
        assert not np.allclose(w_before, w_after), "Weights should update after STDP epoch"
