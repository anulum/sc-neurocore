# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive-coding engine-binding contracts

"""Public contracts for predictive-coding PyO3 functions."""

from __future__ import annotations

from collections.abc import Callable
import importlib

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.engine_requirement import require_engine

require_engine()
import sc_neurocore_engine as engine
from sc_neurocore.spike_codec import PredictiveSpikeCodec
from sc_neurocore.spike_codec import predictive_codec

Predict = Callable[..., tuple[NDArray[np.int8], int]]
Recover = Callable[..., NDArray[np.int8]]
inner = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")


def test_exported_function_names_and_signatures_are_stable() -> None:
    """Require the package facade to publish the installed native objects."""
    expected = {
        "py_predict_xor_ema": "(spikes, n_channels, alpha, threshold)",
        "py_recover_xor_ema": "(errors, n_channels, alpha, threshold)",
        "py_predict_xor_lfsr": "(spikes, n_channels, alpha_q8, seed)",
        "py_recover_xor_lfsr": "(errors, n_channels, alpha_q8, seed)",
        "py_predict_xor_context": "(spikes, n_channels, history_bits)",
        "py_recover_xor_context": "(errors, n_channels, history_bits)",
        "py_update_spike_world_model": (
            "(weights, bias, features, probabilities, samples, learning_rate)"
        ),
    }
    for name, signature in expected.items():
        function = getattr(engine, name)
        assert name in engine.__all__
        assert function is getattr(inner, name)
        assert function.__name__ == name
        assert function.__text_signature__ == signature
    assert inner.py_prediction_error.__name__ == "py_prediction_error"
    assert inner.py_prediction_error.__text_signature__ == "(predicted, actual, length)"
    assert not hasattr(engine, "py_prediction_error")


def test_packed_prediction_error_matches_xor_popcount() -> None:
    """Compare packed disagreement against the known changed-bit fraction."""
    predicted = np.array([0xFFFF, 0], dtype=np.uint64)
    actual = np.zeros(2, dtype=np.uint64)

    assert inner.py_prediction_error(predicted, actual, 128) == 0.125
    assert inner.py_prediction_error(predicted, predicted, 128) == 0.0
    assert inner.py_prediction_error(predicted, actual, 0) == 0.0


@pytest.mark.parametrize(
    ("predict", "recover", "parameters"),
    (
        (engine.py_predict_xor_ema, engine.py_recover_xor_ema, (0.1, 0.5)),
        (engine.py_predict_xor_lfsr, engine.py_recover_xor_lfsr, (1, 0xACE1)),
    ),
)
def test_seeded_predictors_roundtrip_exactly(
    predict: Predict,
    recover: Recover,
    parameters: tuple[float, float] | tuple[int, int],
) -> None:
    """Recover the original seeded raster and check the native prediction count."""
    spikes = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
    errors, correct = predict(spikes, 2, *parameters)
    restored = recover(errors, 2, *parameters)

    np.testing.assert_array_equal(restored, spikes)
    assert correct == 4


def test_noncontiguous_arrays_preserve_value_error_contracts() -> None:
    """Refuse reversed views with the established public error messages."""
    packed = np.array([0, 1, 2, 3], dtype=np.uint64)
    spikes = np.array([0, 1, 0, 1], dtype=np.int8)

    with pytest.raises(ValueError, match=r"^predicted array must be contiguous:"):
        inner.py_prediction_error(packed[::-1], packed, 256)
    with pytest.raises(ValueError, match=r"^actual array must be contiguous:"):
        inner.py_prediction_error(packed, packed[::-1], 256)
    with pytest.raises(ValueError, match=r"^spikes must be contiguous:"):
        engine.py_predict_xor_ema(spikes[::-1], 2, 0.1, 0.5)
    with pytest.raises(ValueError, match=r"^errors must be contiguous:"):
        engine.py_recover_xor_lfsr(spikes[::-1], 2, 1, 0xACE1)


@pytest.mark.parametrize("predictor", ("ema", "lfsr"))
def test_production_predictive_codec_uses_native_lossless_path(predictor: str) -> None:
    """Verify public dispatch through the registered engine and decode a sparse raster."""
    assert predictive_codec._HAS_RUST is True
    if predictor == "ema":
        assert predictive_codec._rust_predict_ema is engine.py_predict_xor_ema
        assert predictive_codec._rust_recover_ema is engine.py_recover_xor_ema
    else:
        assert predictive_codec._rust_predict_lfsr is engine.py_predict_xor_lfsr
        assert predictive_codec._rust_recover_lfsr is engine.py_recover_xor_lfsr

    spikes = np.zeros((128, 8), dtype=np.int8)
    spikes[3, 2] = 1
    spikes[64, 7] = 1
    codec = PredictiveSpikeCodec(predictor=predictor, seed=0xACE1)
    payload, result = codec.compress(spikes)

    np.testing.assert_array_equal(codec.decompress(payload, 128, 8), spikes)
    assert result.lossless is True


@pytest.mark.parametrize("predictor", ("ema", "lfsr", "context", "world_model"))
@pytest.mark.parametrize("precision", (1, 2))
def test_explicit_backends_interoperate(predictor: str, precision: int) -> None:
    """Compare complete packets, statistics and decoded values in both directions."""
    spikes = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.int8)
    expected = (
        spikes
        if precision == 1
        else np.array([[1, 1], [1, 1], [0, 0], [0, 0], [0, 0]], dtype=np.int8)
    )
    mode = "lossless" if precision == 1 else "lossy"
    reference = PredictiveSpikeCodec(
        predictor=predictor, base_mode=mode, timing_precision=precision, backend="python"
    )
    native = PredictiveSpikeCodec(
        predictor=predictor, base_mode=mode, timing_precision=precision, backend="rust"
    )
    python_packet, python_result = reference.compress(spikes)
    rust_packet, rust_result = native.compress(spikes)

    assert python_packet == rust_packet
    assert python_result == rust_result
    np.testing.assert_array_equal(reference.decompress(rust_packet, *spikes.shape), expected)
    np.testing.assert_array_equal(native.decompress(python_packet, *spikes.shape), expected)


def test_native_context_refuses_invalid_input() -> None:
    """Reject incomplete rows, nonbinary samples and unrepresentable history."""
    spikes = np.array([1, 0, 1, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="complete channel rows"):
        engine.py_predict_xor_context(spikes, 3, 8)
    with pytest.raises(ValueError, match="samples must be 0 or 1"):
        engine.py_recover_xor_context(np.array([0, 2], dtype=np.int8), 2, 8)
    with pytest.raises(ValueError, match="history length"):
        engine.py_predict_xor_context(spikes, 2, 64)
    with pytest.raises(ValueError, match="spikes must be contiguous"):
        engine.py_predict_xor_context(spikes[::-1], 2, 8)
    with pytest.raises(ValueError, match="errors must be contiguous"):
        engine.py_recover_xor_context(spikes[::-1], 2, 8)


def test_native_world_model_update_keeps_state_after_shape_refusal() -> None:
    """Check rejection, retry and exact per-channel NumPy weight and bias updates."""
    weights = np.array([[0.125, 0.25], [0.5, -1.0]])
    bias = np.array([0.0, 0.125])
    original_weights, original_bias = weights.copy(), bias.copy()
    history = np.array([1.0, 0.0])
    probabilities = np.array([0.25, 0.75])
    actual = np.array([1.0, 0.0])

    for features, observed in [(history[:1], actual), (history, actual[:1])]:
        with pytest.raises(ValueError, match="world-model update dimensions must match"):
            engine.py_update_spike_world_model(
                weights, bias, features, probabilities, observed, 0.5
            )
        np.testing.assert_array_equal(weights, original_weights)
        np.testing.assert_array_equal(bias, original_bias)

    engine.py_update_spike_world_model(weights, bias, history, probabilities, actual, 0.5)
    error = actual - probabilities
    np.testing.assert_array_equal(weights, original_weights + 0.5 * np.outer(error, history))
    np.testing.assert_array_equal(bias, original_bias + 0.5 * error)
