# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for JAX Layer

import pytest
import numpy as np

pytest.importorskip("jax")

from sc_neurocore.accel.jax_backend import to_jax, to_host
from sc_neurocore.layers.jax_dense_layer import JaxSCDenseLayer


def test_jax_dense_layer_init():
    layer = JaxSCDenseLayer(n_neurons=10, n_inputs=5, seed=42)
    assert layer.v.shape == (10,)
    assert np.all(to_host(layer.v) == 0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_neurons": 0, "n_inputs": 5}, "n_neurons"),
        ({"n_neurons": 4, "n_inputs": 0}, "n_inputs"),
        ({"n_neurons": 4, "n_inputs": 2, "bitstream_length": 0}, "bitstream_length"),
        ({"n_neurons": 4, "n_inputs": 2, "dt_ms": 0.0}, "dt_ms"),
        ({"n_neurons": 4, "n_inputs": 2, "neuron_params": {"tau_mem": 0.0}}, "tau_mem"),
        ({"n_neurons": 4, "n_inputs": 2, "neuron_params": {"noise_std": -0.1}}, "noise_std"),
        ({"n_neurons": 4, "n_inputs": 2, "neuron_params": {"v_threshold": np.nan}}, "v_threshold"),
    ],
)
def test_jax_dense_layer_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JaxSCDenseLayer(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_neurons": 4, "n_inputs": 2, "neuron_params": {"threshold": 1.0}}, "neuron_params"),
        ({"n_neurons": 4, "n_inputs": 2, "seed": -1}, "seed"),
        ({"n_neurons": 4, "n_inputs": 2, "seed": True}, "seed"),
        ({"n_neurons": 4, "n_inputs": 2, "seed": 2**32}, "seed"),
    ],
)
def test_jax_dense_layer_rejects_unknown_params_and_invalid_seed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JaxSCDenseLayer(**kwargs)


def test_jax_dense_layer_preserves_zero_seed_reproducibly():
    layer_a = JaxSCDenseLayer(n_neurons=4, n_inputs=2, seed=0)
    layer_b = JaxSCDenseLayer(n_neurons=4, n_inputs=2, seed=0)
    current = np.array([1.2, 0.1, 1.2, 0.1], dtype=np.float32)

    assert np.array_equal(to_host(layer_a.step(current)), to_host(layer_b.step(current)))


def test_jax_dense_layer_projects_input_vector_through_dense_weights():
    weights = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.1],
            [1.0, 0.0],
            [0.0, 0.1],
        ],
        dtype=np.float32,
    )
    layer = JaxSCDenseLayer(n_neurons=4, n_inputs=2, weights=weights, seed=123)

    spikes = to_host(layer.step(np.array([2.0, 0.0], dtype=np.float32)))

    assert np.array_equal(spikes, np.array([1, 0, 1, 0], dtype=np.uint8))


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (np.ones((3, 2), dtype=np.float32), "weights"),
        (np.ones((4, 3), dtype=np.float32), "weights"),
        (np.ones((4, 2), dtype=np.int64), "floating-point"),
        (np.array([[np.nan, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32), "finite"),
    ],
)
def test_jax_dense_layer_rejects_invalid_dense_weights(weights, match):
    with pytest.raises(ValueError, match=match):
        JaxSCDenseLayer(n_neurons=4, n_inputs=2, weights=weights)


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


@pytest.mark.parametrize(
    ("current", "match"),
    [
        (np.array([1.0, 0.0, 1.0], dtype=np.float32), "shape"),
        (np.array([1, 0, 1, 0], dtype=np.int64), "floating-point"),
        (np.array([1.0, np.nan, 1.0, 0.0], dtype=np.float32), "finite"),
    ],
)
def test_jax_dense_layer_step_rejects_invalid_currents(current, match):
    layer = JaxSCDenseLayer(n_neurons=4, n_inputs=2, seed=123)

    with pytest.raises(ValueError, match=match):
        layer.step(current)


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


def test_jax_dense_layer_run_projects_input_sequence_through_dense_weights():
    weights = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.1],
            [1.0, 0.0],
            [0.0, 0.1],
        ],
        dtype=np.float32,
    )
    layer = JaxSCDenseLayer(n_neurons=4, n_inputs=2, weights=weights, seed=123)

    all_spikes = to_host(
        layer.run(
            np.array(
                [
                    [2.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
    )

    assert all_spikes.shape == (3, 4)
    assert np.array_equal(all_spikes[0], np.array([1, 0, 1, 0], dtype=np.uint8))


@pytest.mark.parametrize(
    ("currents", "match"),
    [
        (np.array([2.0, 0.0], dtype=np.float32), "2-D"),
        (np.zeros((0, 2), dtype=np.float32), "non-empty"),
        (np.zeros((3, 3), dtype=np.float32), "shape"),
        (np.array([[2.0, np.inf]], dtype=np.float32), "finite"),
    ],
)
def test_jax_dense_layer_run_rejects_invalid_currents(currents, match):
    layer = JaxSCDenseLayer(n_neurons=2, n_inputs=2, seed=456)

    with pytest.raises(ValueError, match=match):
        layer.run(currents)
