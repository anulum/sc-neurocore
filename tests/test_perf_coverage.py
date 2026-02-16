"""
Additional tests to exercise fallback code paths and increase coverage.
"""

import numpy as np
import pytest

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer
from sc_neurocore.layers.sc_dense_layer import SCDenseLayer
from sc_neurocore.layers.sc_learning_layer import SCLearningLayer
from sc_neurocore.learning.neuroevolution import SNNGeneticEvolver


class TestVectorizedLayerNumpyPath:
    """Force the NumPy vectorized path (bypass Rust)."""

    def test_forward_numpy_path(self):
        np.random.seed(42)
        layer = VectorizedSCLayer(n_inputs=4, n_neurons=3, length=256, use_gpu=False)
        # Force NumPy path by disabling Rust layer
        layer._rust_layer = None
        out = layer.forward([0.3, 0.5, 0.7, 0.2])
        assert out.shape == (3,)
        assert np.all(np.isfinite(out))


class TestSCDenseLayerFallback:
    """Force the Python fallback path (noisy neurons)."""

    def test_run_fallback_path(self):
        np.random.seed(42)
        layer = SCDenseLayer(
            n_neurons=2,
            x_inputs=[0.05, 0.06],
            weight_values=[0.5, 0.5],
            x_min=0.0, x_max=0.1,
            w_min=0.0, w_max=1.0,
            length=128,
            neuron_params={"noise_std": 0.1},  # noise -> fallback
            base_seed=42,
        )
        assert not layer._can_use_fast_path()
        layer.run(50)
        trains = layer.get_spike_trains()
        assert trains.shape == (2, 50)


class TestSCLearningLayerFallback:
    """Force the Python fallback path for learning layer."""

    def test_run_epoch_fallback(self):
        np.random.seed(42)
        layer = SCLearningLayer(
            n_inputs=2, n_neurons=2, length=64,
            base_seed=42, learning_rate=0.01,
        )
        # Force noisy neurons -> fallback
        for n in layer.neurons:
            n.noise_std = 0.1
        assert not layer._can_use_fast_path()
        spikes = layer.run_epoch([0.5, 0.5])
        assert spikes.shape == (2, 64)
        assert spikes.dtype == np.uint8


class TestNeuroevolution:
    """Exercise genetic algorithm evolve path."""

    def test_evolve_basic(self):
        np.random.seed(42)

        class SimpleIndividual:
            def __init__(self):
                self.weights = np.random.random(5)

        def fitness(ind):
            return float(np.sum(ind.weights))

        evolver = SNNGeneticEvolver.__new__(SNNGeneticEvolver)
        evolver.population_size = 10
        evolver.mutation_rate = 0.1
        evolver.elite_fraction = 0.3
        evolver.layer_factory = SimpleIndividual
        evolver.fitness_func = fitness
        evolver.population = [SimpleIndividual() for _ in range(10)]

        best = evolver.evolve(3)
        assert hasattr(best, "weights")
        assert best.weights.shape == (5,)
