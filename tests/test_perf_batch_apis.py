"""
Phase 1 tests: JIT/Rust equivalence for stochastic_lif, dense_layer,
vectorized_layer, orchestrator batch_step.
"""

import numpy as np
import pytest

from sc_neurocore.neurons.stochastic_lif import (
    StochasticLIFNeuron,
    _lif_bitstream_kernel,
    _lif_step_array_kernel,
)
from sc_neurocore.neurons.base import BaseNeuron
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer
from sc_neurocore.layers.sc_dense_layer import SCDenseLayer


class TestStochasticLIFJIT:
    """Verify JIT kernel produces identical output to the Python loop."""

    def test_process_bitstream_deterministic(self):
        """No-noise neuron: JIT path matches manual per-step loop."""
        neuron_jit = StochasticLIFNeuron(noise_std=0.0, seed=42)
        neuron_ref = StochasticLIFNeuron(noise_std=0.0, seed=42)

        bits = (np.random.RandomState(0).random(200) > 0.4).astype(np.uint8)

        # JIT path (via process_bitstream fast path)
        spikes_jit = neuron_jit.process_bitstream(bits, input_scale=0.08)

        # Reference: per-step Python
        spikes_ref = np.zeros_like(bits, dtype=np.uint8)
        for i, bit in enumerate(bits):
            spikes_ref[i] = neuron_ref.step(bit * 0.08)

        np.testing.assert_array_equal(spikes_jit, spikes_ref)

    def test_step_array_deterministic(self):
        """step_array() matches per-step loop for float currents."""
        neuron_jit = StochasticLIFNeuron(noise_std=0.0, seed=42)
        neuron_ref = StochasticLIFNeuron(noise_std=0.0, seed=42)

        currents = np.random.RandomState(1).random(300) * 0.1

        spikes_jit = neuron_jit.step_array(currents)

        spikes_ref = np.zeros(len(currents), dtype=np.uint8)
        for i in range(len(currents)):
            spikes_ref[i] = neuron_ref.step(currents[i])

        np.testing.assert_array_equal(spikes_jit, spikes_ref)

    def test_step_array_noisy_falls_back(self):
        """With noise enabled, step_array falls back to per-step loop."""
        neuron = StochasticLIFNeuron(noise_std=0.1, seed=42)
        currents = np.ones(50) * 0.06
        spikes = neuron.step_array(currents)
        assert spikes.dtype == np.uint8
        assert len(spikes) == 50

    def test_process_bitstream_noisy_falls_back(self):
        """With noise enabled, process_bitstream falls back."""
        neuron = StochasticLIFNeuron(noise_std=0.1, seed=42)
        bits = np.ones(50, dtype=np.uint8)
        spikes = neuron.process_bitstream(bits)
        assert spikes.dtype == np.uint8


class TestVectorizedLayer:
    """Verify vectorized popcount path produces valid output."""

    def test_forward_shape(self):
        np.random.seed(42)
        layer = VectorizedSCLayer(n_inputs=4, n_neurons=3, length=256, use_gpu=False)
        out = layer.forward([0.3, 0.5, 0.7, 0.2])
        assert out.shape == (3,)
        assert np.all(np.isfinite(out)) and np.all(out >= 0)

    def test_forward_raises_on_bad_input(self):
        layer = VectorizedSCLayer(n_inputs=4, n_neurons=3, length=256, use_gpu=False)
        with pytest.raises(ValueError):
            layer.forward([0.1, 0.2])  # wrong length


class TestSCDenseLayerJIT:
    """Verify JIT dense run kernel produces valid spikes."""

    def test_run_produces_spikes(self):
        np.random.seed(42)
        layer = SCDenseLayer(
            n_neurons=3,
            x_inputs=[0.05, 0.06],
            weight_values=[0.5, 0.5],
            x_min=0.0,
            x_max=0.1,
            w_min=0.0,
            w_max=1.0,
            length=512,
            neuron_params={"noise_std": 0.0},
            base_seed=42,
        )
        layer.run(100)
        trains = layer.get_spike_trains()
        assert trains.shape == (3, 100)
        assert trains.dtype == np.uint8


class TestBaseNeuronBatchStep:
    """Verify batch_step default on BaseNeuron subclass."""

    def test_batch_step_default(self):
        neuron = StochasticLIFNeuron(noise_std=0.0, seed=42)
        inputs = np.array([0.05, 0.06, 0.07])
        out = neuron.batch_step(inputs)
        assert len(out) == 3
