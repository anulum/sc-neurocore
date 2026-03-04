import pytest
import numpy as np
from sc_neurocore.accel.jax_backend import HAS_JAX, to_jax, to_host
from sc_neurocore.layers.jax_dense_layer import JaxSCDenseLayer


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
def test_jax_dense_layer_init():
    layer = JaxSCDenseLayer(n_neurons=10, n_inputs=5, seed=42)
    assert layer.v.shape == (10,)
    assert np.all(to_host(layer.v) == 0.0)


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
def test_jax_dense_layer_step():
    layer = JaxSCDenseLayer(n_neurons=4, n_inputs=2, seed=123)

    # Apply high current to force spikes
    I_t = to_jax(np.array([2.0, 0.0, 2.0, 0.0], dtype=np.float32))

    spikes = layer.step(I_t)
    spikes_np = to_host(spikes)

    assert spikes_np.shape == (4,)
    # Neurons 0 and 2 should have spiked or be close to it
    # Given threshold 1.0 and current 2.0, they should spike
    assert spikes_np[0] == 1
    assert spikes_np[2] == 1
    assert spikes_np[1] == 0
    assert spikes_np[3] == 0


@pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed")
def test_jax_dense_layer_run():
    layer = JaxSCDenseLayer(n_neurons=2, n_inputs=2, seed=456)

    # 5 time steps
    currents = to_jax(
        np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    )

    all_spikes = layer.run(currents)
    all_spikes_np = to_host(all_spikes)

    assert all_spikes_np.shape == (5, 2)
    # Neuron 0 should have many spikes
    assert np.sum(all_spikes_np[:, 0]) >= 3
    # Neuron 1 should have zero spikes
    assert np.sum(all_spikes_np[:, 1]) == 0
