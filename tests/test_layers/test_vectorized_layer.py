# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for VectorizedSCLayer packed operations and outputs

"""Tests for VectorizedSCLayer packed operations and outputs."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def _expected_words(length: int) -> int:
    return (length + 63) // 64


def test_vectorized_packed_shape():
    """Packed weights should have expected word dimension."""
    np.random.seed(0)
    layer = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=130)
    assert layer.packed_weights.shape == (2, 3, _expected_words(130))


def test_vectorized_packed_dtype():
    """Packed weights should be uint64 for bitwise ops."""
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=64)
    assert layer.packed_weights.dtype == np.uint64


def test_vectorized_forward_shape():
    """Forward returns (n_neurons,) output."""
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=4, length=32)
    out = layer.forward([0.3, 0.7])
    assert out.shape == (4,)


def test_vectorized_forward_zero_input_returns_zero():
    """Zero inputs yield near-zero outputs."""
    layer = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=32)
    out = layer.forward([0.0, 0.0, 0.0])
    assert np.allclose(out, 0.0)


def test_vectorized_output_range():
    """Outputs should be within [0, n_inputs]."""
    layer = VectorizedSCLayer(n_inputs=4, n_neurons=3, length=64)
    out = layer.forward([0.2, 0.4, 0.6, 0.8])
    assert np.all(out >= 0.0)
    assert np.all(out <= 4.0)


def test_vectorized_refresh_changes_packed_weights():
    """Refreshing after weight changes updates packed representation."""
    np.random.seed(1)
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=32)
    before = layer.packed_weights.copy()
    layer.weights[:] = 0.0
    layer._refresh_packed_weights()
    assert not np.array_equal(before, layer.packed_weights)


def test_vectorized_deterministic_with_seed():
    """Setting numpy seed yields repeatable weights."""
    np.random.seed(99)
    layer_a = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=32)
    np.random.seed(99)
    layer_b = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=32)
    assert np.allclose(layer_a.weights, layer_b.weights)


def test_vectorized_input_length_mismatch_raises():
    """Input length mismatch should raise a broadcasting error."""
    layer = VectorizedSCLayer(n_inputs=3, n_neurons=2, length=16)
    with pytest.raises(ValueError):
        _ = layer.forward([0.1, 0.2])


def test_vectorized_length_not_multiple_of_64():
    """Lengths not divisible by 64 should still work."""
    layer = VectorizedSCLayer(n_inputs=1, n_neurons=1, length=70)
    out = layer.forward([0.5])
    assert out.shape == (1,)


def test_vectorized_reject_zero_inputs():
    with pytest.raises(ValueError, match="n_inputs must be >= 1"):
        VectorizedSCLayer(n_inputs=0, n_neurons=2, length=16)


def test_vectorized_reject_zero_neurons():
    with pytest.raises(ValueError, match="n_neurons must be >= 1"):
        VectorizedSCLayer(n_inputs=2, n_neurons=0, length=16)


def test_vectorized_reject_zero_length():
    with pytest.raises(ValueError, match="length must be >= 1"):
        VectorizedSCLayer(n_inputs=2, n_neurons=2, length=0)


def test_vectorized_reject_nan_input():
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=16)
    with pytest.raises(ValueError, match="NaN or Inf"):
        layer.forward([float("nan"), 0.5])


def test_vectorized_reject_out_of_range_input():
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=16)
    with pytest.raises(ValueError, match="probabilities must be in"):
        layer.forward([1.5, 0.5])


def test_vectorized_bipolar_dense_preserves_signed_dot_product():
    """Bipolar mode should use XNOR so negative weights contribute negative terms."""
    np.random.seed(0)
    layer = VectorizedSCLayer(
        n_inputs=2, n_neurons=1, length=4096, use_gpu=False, sc_mode="bipolar"
    )
    layer.weights[:] = np.array([[1.0, -1.0]])
    layer._refresh_packed_weights()

    out = layer.forward([1.0, 1.0])

    assert abs(out[0]) < 0.02


def test_vectorized_bipolar_dense_handles_fractional_signed_weights():
    """Packed bipolar XNOR output should approximate the signed dot product."""
    np.random.seed(1)
    layer = VectorizedSCLayer(
        n_inputs=2, n_neurons=1, length=65536, use_gpu=False, sc_mode="bipolar"
    )
    layer.weights[:] = np.array([[0.5, -0.25]])
    layer._refresh_packed_weights()

    out = layer.forward([1.0, -1.0])

    assert np.allclose(out, [0.75], atol=0.03)


def test_vectorized_bipolar_rejects_out_of_range_input():
    """Bipolar mode accepts only signed values in [-1, 1]."""
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=1, length=64, use_gpu=False, sc_mode="bipolar")
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        layer.forward([0.0, 1.5])


def test_vectorized_rejects_unknown_sc_mode():
    """Unknown SC mode should fail closed instead of silently using AND semantics."""
    with pytest.raises(ValueError, match="sc_mode"):
        VectorizedSCLayer(n_inputs=2, n_neurons=1, sc_mode="ternary")


def test_vectorized_sparse_mode_requires_scipy_sparse_support():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "sc_neurocore.layers.vectorized_layer._has_scipy_sparse",
            lambda: False,
        )
        with pytest.raises(ImportError, match="scipy"):
            VectorizedSCLayer(n_inputs=4, n_neurons=8, sparse=True)


def test_vectorized_from_exported_bipolar_weights_preserves_classifier_accuracy():
    """Exported signed trained weights should run through packed bipolar SC inference."""
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


def test_vectorized_from_exported_bipolar_weights_uses_scaled_bias():
    """Packed inference should consume bias in the same normalised domain as weights."""
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

    out = layer.forward(sample)

    np.testing.assert_allclose(out, expected, atol=0.08)


def test_vectorized_from_exported_weights_rejects_encoding_mismatch():
    exported = {"weight": np.array([[0.25, 0.5]], dtype=np.float64), "encoding": "bipolar"}

    with pytest.raises(ValueError, match="encoding"):
        VectorizedSCLayer.from_exported_weights(exported, sc_mode="unipolar")


def test_vectorized_from_exported_weights_rejects_bad_bias_shape():
    exported = {
        "weight": np.array([[0.25, 0.5], [0.5, 0.25]], dtype=np.float64),
        "bias": np.array([0.1], dtype=np.float64),
        "encoding": "unipolar",
    }

    with pytest.raises(ValueError, match="bias"):
        VectorizedSCLayer.from_exported_weights(exported)


def test_vectorized_seed_reproducibly_controls_forward_streams():
    exported = {
        "weight": np.array([[0.5, -0.25]], dtype=np.float64),
        "bias": np.array([0.125], dtype=np.float64),
        "encoding": "bipolar",
    }
    first = VectorizedSCLayer.from_exported_weights(exported, length=4096, use_gpu=False, seed=23)
    second = VectorizedSCLayer.from_exported_weights(exported, length=4096, use_gpu=False, seed=23)

    np.testing.assert_array_equal(first.forward([0.8, -0.4]), second.forward([0.8, -0.4]))


def test_vectorized_exported_dense_seed_does_not_depend_on_constructor_prefill():
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


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_vectorized_layer_perf_small():
    """Benchmark a small vectorized forward pass."""
    layer = VectorizedSCLayer(n_inputs=8, n_neurons=32, length=128)
    start = time.perf_counter()
    _ = layer.forward([0.5] * 8)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0


def test_mask_unused_tail_bits_masks_partial_final_word():
    """A non-64-aligned length zeroes the unused high bits of the final word."""
    from sc_neurocore.layers.vectorized_layer import _mask_unused_tail_bits

    packed = np.array([0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    masked = _mask_unused_tail_bits(packed, length=64 + 4)  # valid_tail = 4
    assert masked[-1] == np.uint64((1 << 4) - 1)
    assert masked[0] == np.uint64(0xFFFFFFFFFFFFFFFF)  # earlier words untouched
    # The helper must not mutate its input in place.
    assert packed[-1] == np.uint64(0xFFFFFFFFFFFFFFFF)


def test_as_float_array_rejects_non_finite_values():
    from sc_neurocore.layers.vectorized_layer import _as_float_array

    with pytest.raises(ValueError, match="NaN or Inf"):
        _as_float_array([1.0, np.nan], "weight")


def test_from_exported_weights_validates_payload():
    with pytest.raises(ValueError, match="must contain a 'weight'"):
        VectorizedSCLayer.from_exported_weights({})
    with pytest.raises(ValueError, match="2-D matrix"):
        VectorizedSCLayer.from_exported_weights({"weight": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="must be 'unipolar' or 'bipolar'"):
        VectorizedSCLayer.from_exported_weights({"weight": [[0.5, 0.5]], "encoding": "ternary"})
    with pytest.raises(ValueError, match=r"bipolar exported weights must be in \[-1, 1\]"):
        VectorizedSCLayer.from_exported_weights({"weight": [[2.0, 0.0]], "encoding": "bipolar"})
    with pytest.raises(ValueError, match=r"unipolar exported weights must be in \[0, 1\]"):
        VectorizedSCLayer.from_exported_weights({"weight": [[2.0, 0.0]], "encoding": "unipolar"})
