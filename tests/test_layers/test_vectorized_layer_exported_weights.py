# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — VectorizedSCLayer exported-weight contracts

"""Verify trained-weight import, scaling, shape, and deterministic stream contracts."""

import numpy as np
import pytest

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer


def test_vectorized_from_exported_bipolar_weights_preserves_classifier_accuracy() -> None:
    """Exported signed trained weights preserve classifier predictions."""
    torch = pytest.importorskip("torch")
    from sc_neurocore.training.snn_modules import SpikingNet

    net = SpikingNet(n_input=2, n_hidden=2, n_output=2, n_layers=0)
    with torch.no_grad():
        net.linears[0].weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
        net.linears[0].bias.copy_(torch.tensor([0.1, -0.1]))
    exported = net.to_sc_weights(encoding="bipolar")
    layer = VectorizedSCLayer.from_exported_weights(
        exported[0],
        length=65_536,
        use_gpu=False,
        seed=17,
    )
    samples = np.array(
        [
            [0.95, -0.95],
            [0.75, -0.60],
            [-0.90, 0.85],
            [-0.70, 0.95],
            [0.60, -0.90],
            [-0.80, 0.65],
        ],
        dtype=np.float64,
    )
    float_logits = samples @ exported[0]["weight"].numpy().T + exported[0]["bias"].numpy()
    sc_logits = np.vstack([layer.forward(sample) for sample in samples])

    assert np.mean(sc_logits.argmax(axis=1) == float_logits.argmax(axis=1)) == 1.0
    assert np.max(np.abs(sc_logits - float_logits)) < 0.08


def test_vectorized_from_exported_bipolar_weights_uses_scaled_bias() -> None:
    """Packed inference consumes bias in the same normalized domain as weights."""
    torch = pytest.importorskip("torch")
    from sc_neurocore.training.snn_modules import SpikingNet

    net = SpikingNet(n_input=2, n_hidden=2, n_output=2, n_layers=0)
    with torch.no_grad():
        net.linears[0].weight.copy_(torch.tensor([[4.0, -2.0], [-4.0, 2.0]]))
        net.linears[0].bias.copy_(torch.tensor([2.0, -2.0]))
    exported = net.to_sc_weights(encoding="bipolar")
    layer = VectorizedSCLayer.from_exported_weights(
        exported[0], length=65_536, use_gpu=False, seed=19
    )
    sample = np.array([0.8, -0.6], dtype=np.float64)
    expected = sample @ exported[0]["weight"].numpy().T + exported[0]["bias"].numpy()

    np.testing.assert_allclose(layer.forward(sample), expected, atol=0.08)


def test_vectorized_from_exported_weights_rejects_encoding_mismatch() -> None:
    exported = {"weight": np.array([[0.25, 0.5]], dtype=np.float64), "encoding": "bipolar"}
    with pytest.raises(ValueError, match="encoding"):
        VectorizedSCLayer.from_exported_weights(exported, sc_mode="unipolar")


def test_vectorized_from_exported_weights_rejects_bad_bias_shape() -> None:
    exported = {
        "weight": np.array([[0.25, 0.5], [0.5, 0.25]], dtype=np.float64),
        "bias": np.array([0.1], dtype=np.float64),
        "encoding": "unipolar",
    }
    with pytest.raises(ValueError, match="bias"):
        VectorizedSCLayer.from_exported_weights(exported)


def test_vectorized_seed_reproducibly_controls_forward_streams() -> None:
    exported = {
        "weight": np.array([[0.5, -0.25]], dtype=np.float64),
        "bias": np.array([0.125], dtype=np.float64),
        "encoding": "bipolar",
    }
    first = VectorizedSCLayer.from_exported_weights(exported, length=4096, use_gpu=False, seed=23)
    second = VectorizedSCLayer.from_exported_weights(exported, length=4096, use_gpu=False, seed=23)
    np.testing.assert_array_equal(first.forward([0.8, -0.4]), second.forward([0.8, -0.4]))


def test_vectorized_exported_dense_seed_does_not_depend_on_constructor_prefill() -> None:
    exported = {
        "weight": np.array([[0.5, -0.25]], dtype=np.float64),
        "bias": np.array([0.125], dtype=np.float64),
        "encoding": "bipolar",
    }
    from_export = VectorizedSCLayer.from_exported_weights(
        exported, length=4096, use_gpu=False, seed=41
    )
    direct = VectorizedSCLayer(
        n_inputs=2, n_neurons=1, length=4096, use_gpu=False, sc_mode="bipolar", seed=41
    )
    direct.weights = exported["weight"].copy()
    direct.bias_values = exported["bias"].copy()
    direct._rng = np.random.default_rng(41)
    direct._refresh_packed_weights()

    np.testing.assert_array_equal(from_export.packed_weights, direct.packed_weights)
    np.testing.assert_array_equal(from_export.forward([0.8, -0.4]), direct.forward([0.8, -0.4]))
