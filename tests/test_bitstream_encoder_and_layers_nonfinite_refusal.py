# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Non-finite refusal tests for BitstreamEncoder and SC layers

"""Verify non-finite and shape refusals across BitstreamEncoder and SC layer primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.exceptions import SCEncodingError
from sc_neurocore.layers.fusion import SCFusionLayer
from sc_neurocore.layers.predictive_coding import PredictiveCodingSCLayer
from sc_neurocore.layers.recurrent import SCRecurrentLayer
from sc_neurocore.layers.sc_dense_layer import SCDenseLayer
from sc_neurocore.layers.sc_learning_layer import SCLearningLayer
from sc_neurocore.utils.bitstreams import (
    BitstreamEncoder,
    adaptive_length,
    bipolar_to_value,
    bitstream_to_probability,
    generate_bernoulli_bitstream,
    generate_bipolar_bitstream,
    generate_sobol_bitstream,
    unipolar_prob_to_value,
    value_to_bipolar_prob,
    value_to_unipolar_prob,
)


@pytest.mark.parametrize("mode", ["bernoulli", "bipolar", "sobol", "chaotic"])
@pytest.mark.parametrize(
    "bad_x",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        np.bool_(True),
        0.5 + 0.0j,
        np.complex128(0.25),
    ],
)
def test_bitstream_encoder_all_modes_refuse_nonfinite_bool_or_complex_without_advancing_rng(
    mode: str, bad_x: Any
) -> None:
    """BitstreamEncoder.encode rejects non-finite, boolean, and complex inputs cleanly."""
    enc = BitstreamEncoder(x_min=-1.0, x_max=1.0, length=64, seed=17, mode=mode)
    enc_ref = BitstreamEncoder(x_min=-1.0, x_max=1.0, length=64, seed=17, mode=mode)

    with pytest.raises(SCEncodingError):
        enc.encode(bad_x)

    # Subsequent valid encode matches a fresh identically-seeded encoder bit for bit.
    np.testing.assert_array_equal(enc.encode(float(np.float32(0.25))), enc_ref.encode(0.25))
    np_scalar_val: Any = np.float32(0.25)
    assert enc.encode(np_scalar_val).shape == (64,)


def test_bitstream_encoder_preserves_finite_out_of_range_clipping() -> None:
    """Finite out-of-range values clamp when clip=True and raise when clip=False."""
    for mode in ("bernoulli", "bipolar", "sobol", "chaotic"):
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=64, seed=5, mode=mode)
        low_bits = enc.encode(-2.5)
        high_bits = enc.encode(3.5)
        assert int(low_bits.sum()) == 0
        assert int(high_bits.sum()) == 64
        assert enc.decode(low_bits) == pytest.approx(0.0)
        assert enc.decode(high_bits) == pytest.approx(1.0)

    assert value_to_unipolar_prob(0.25, 0.0, 1.0, clip=False) == pytest.approx(0.25)
    assert value_to_bipolar_prob(0.0) == pytest.approx(0.5)
    assert generate_bernoulli_bitstream(0.5, 16).shape == (16,)
    assert adaptive_length(0.5, epsilon=0.05, method="variance") >= 64
    assert adaptive_length(0.5, epsilon=0.05, confidence=0.9, method="chebyshev") >= 64
    assert adaptive_length(0.5, epsilon=0.05, confidence=0.9, method="hoeffding") >= 64
    with pytest.raises(SCEncodingError):
        value_to_bipolar_prob(1.5)
    with pytest.raises(SCEncodingError):
        generate_bipolar_bitstream(1.5, 16)
    with pytest.raises(ValueError):
        adaptive_length(0.5, confidence=1.0, method="chebyshev")
    with pytest.raises(ValueError):
        adaptive_length(0.5, confidence=1.0, method="hoeffding")
    with pytest.raises(ValueError):
        adaptive_length(0.5, method="unknown")
    with pytest.raises(SCEncodingError, match="clip=False"):
        value_to_unipolar_prob(1.5, 0.0, 1.0, clip=False)
    with pytest.raises(SCEncodingError, match="x_min"):
        unipolar_prob_to_value(0.5, 1.0, 1.0)


@pytest.mark.parametrize("bad_bound", [float("nan"), float("inf"), float("-inf"), True])
def test_bitstream_encoder_and_unipolar_helpers_reject_nonfinite_bounds(bad_bound: Any) -> None:
    """Encoder construction and unipolar range helpers reject non-finite or boolean bounds."""
    with pytest.raises(SCEncodingError):
        BitstreamEncoder(x_min=bad_bound, x_max=1.0, length=32)
    with pytest.raises(SCEncodingError):
        BitstreamEncoder(x_min=0.0, x_max=bad_bound, length=32)
    with pytest.raises(SCEncodingError):
        value_to_unipolar_prob(0.5, bad_bound, 1.0)
    with pytest.raises(SCEncodingError):
        value_to_unipolar_prob(0.5, 0.0, bad_bound)
    with pytest.raises(SCEncodingError):
        unipolar_prob_to_value(0.5, bad_bound, 1.0)
    with pytest.raises(SCEncodingError):
        unipolar_prob_to_value(0.5, 0.0, bad_bound)


@pytest.mark.parametrize("bad_len", [0, -8, True])
def test_bitstream_encoder_and_generators_reject_invalid_lengths(bad_len: Any) -> None:
    """Encoder construction requires length >= 1 and generators reject negative/bool lengths."""
    with pytest.raises(SCEncodingError):
        BitstreamEncoder(x_min=0.0, x_max=1.0, length=bad_len)
    if bad_len is True or bad_len < 0:
        with pytest.raises(SCEncodingError):
            generate_bernoulli_bitstream(0.5, bad_len)
        with pytest.raises(SCEncodingError):
            generate_sobol_bitstream(0.5, bad_len, seed=3)


@pytest.mark.parametrize(
    "bad_val",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        np.bool_(False),
        0.5 + 0.0j,
        np.complex128(0.5),
    ],
)
def test_bitstream_scalar_and_decode_helpers_reject_nonfinite_inputs(bad_val: Any) -> None:
    """Scalar probability/value helpers, decoders, and adaptive_length reject non-finite values."""
    with pytest.raises(SCEncodingError):
        value_to_unipolar_prob(bad_val, 0.0, 1.0, clip=True)
    with pytest.raises(SCEncodingError):
        generate_bernoulli_bitstream(bad_val, 32)
    with pytest.raises(SCEncodingError):
        generate_sobol_bitstream(bad_val, 32, seed=7)
    with pytest.raises(SCEncodingError):
        generate_bipolar_bitstream(bad_val, 32)
    with pytest.raises(SCEncodingError):
        value_to_bipolar_prob(bad_val)
    with pytest.raises(SCEncodingError):
        unipolar_prob_to_value(bad_val, 0.0, 1.0)
    with pytest.raises(ValueError):
        adaptive_length(p=bad_val)
    with pytest.raises(ValueError):
        adaptive_length(p=0.5, epsilon=bad_val)
    with pytest.raises(ValueError):
        adaptive_length(p=0.5, confidence=bad_val)

    if not isinstance(bad_val, (bool, np.bool_, complex, np.complexfloating)):
        arr = np.array([1.0, bad_val, 0.0], dtype=np.float64)
        with pytest.raises(SCEncodingError):
            bitstream_to_probability(arr)
        with pytest.raises(SCEncodingError):
            bipolar_to_value(arr)


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_sc_dense_layer_rejects_nonfinite_inputs_and_bounds(bad_val: float) -> None:
    """SCDenseLayer rejects non-finite x_inputs, range bounds, and dt_ms at construction."""
    with pytest.raises((ValueError, SCEncodingError)):
        SCDenseLayer(
            n_neurons=2,
            x_inputs=[0.5, bad_val],
            weight_values=[0.4, 0.6],
            x_min=0.0,
            x_max=1.0,
            w_min=0.0,
            w_max=1.0,
            length=32,
        )
    with pytest.raises((ValueError, SCEncodingError)):
        SCDenseLayer(
            n_neurons=2,
            x_inputs=[0.5, 0.2],
            weight_values=[0.4, 0.6],
            x_min=0.0,
            x_max=1.0,
            w_min=0.0,
            w_max=1.0,
            y_max=bad_val,
            length=32,
        )
    with pytest.raises(ValueError):
        SCDenseLayer(
            n_neurons=2,
            x_inputs=[0.5, 0.2],
            weight_values=[0.4, 0.6],
            x_min=0.0,
            x_max=1.0,
            w_min=0.0,
            w_max=1.0,
            dt_ms=bad_val,
            length=32,
        )


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_sc_recurrent_layer_refuses_nonfinite_or_mismatched_input_without_mutating_state(
    bad_val: float,
) -> None:
    """SCRecurrentLayer.step rejects non-finite or wrong-shape inputs before mutating state."""
    layer = SCRecurrentLayer(n_inputs=3, n_neurons=4, length=32, seed=9)
    initial = layer.step(np.array([0.4, 0.2, 0.7], dtype=np.float64)).copy()

    with pytest.raises(ValueError, match="finite"):
        layer.step(np.array([0.3, bad_val, 0.1], dtype=np.float64))
    np.testing.assert_array_equal(layer.state, initial)

    with pytest.raises(ValueError, match="shape"):
        layer.step(np.array([0.3, 0.1], dtype=np.float64))
    np.testing.assert_array_equal(layer.state, initial)
    layer.reset()
    np.testing.assert_array_equal(layer.state, np.zeros(4))

    with pytest.raises(ValueError, match="positive"):
        SCRecurrentLayer(n_inputs=0, n_neurons=2)
    with pytest.raises(ValueError, match="length"):
        SCRecurrentLayer(n_inputs=2, n_neurons=2, length=0)
    with pytest.raises(ValueError, match="finite"):
        SCRecurrentLayer(n_inputs=2, n_neurons=2, input_strength=bad_val)


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_sc_learning_layer_refuses_nonfinite_or_mismatched_epoch_inputs_atomically(
    bad_val: float,
) -> None:
    """SCLearningLayer.run_epoch rejects non-finite or mismatched inputs before mutating weights."""
    layer = SCLearningLayer(n_inputs=3, n_neurons=2, length=32, base_seed=11)
    weights_before = layer.get_weights().copy()
    voltages_before = [float(n.v) for n in layer.neurons]

    with pytest.raises((ValueError, SCEncodingError)):
        layer.run_epoch([0.5, bad_val, 0.2])
    with pytest.raises(ValueError, match="n_inputs"):
        layer.run_epoch([0.5, 0.2])

    np.testing.assert_array_equal(layer.get_weights(), weights_before)
    assert [float(n.v) for n in layer.neurons] == voltages_before


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_predictive_coding_and_fusion_layers_refuse_nonfinite_inputs_atomically(
    bad_val: float,
) -> None:
    """PredictiveCodingSCLayer.forward and SCFusionLayer.forward reject non-finite inputs."""
    pc = PredictiveCodingSCLayer(n_inputs=3, n_neurons=2, length=32, seed=5)
    weights_before = pc.weights.copy()

    with pytest.raises(ValueError, match="finite"):
        pc.forward([0.2, bad_val, 0.8])
    with pytest.raises(ValueError, match="shape"):
        pc.forward([0.2, 0.8])
    np.testing.assert_array_equal(pc.weights, weights_before)
    valid_pc = pc.forward([0.2, 0.5, 0.8])
    assert 0.0 <= valid_pc["prediction_error"] <= 1.0
    pc.reset()
    np.testing.assert_array_equal(pc.weights, weights_before)

    fusion = SCFusionLayer(
        input_dims={"audio": 3, "visual": 3},
        fusion_weights={"audio": 0.6, "visual": 0.4},
    )
    with pytest.raises(ValueError, match="finite"):
        fusion.forward(
            {
                "audio": np.array([0.1, bad_val, 0.3], dtype=np.float64),
                "visual": np.array([0.2, 0.4, 0.6], dtype=np.float64),
            }
        )
    with pytest.raises(ValueError, match="finite"):
        SCFusionLayer(
            input_dims={"audio": 3},
            fusion_weights={"audio": bad_val},
        )
