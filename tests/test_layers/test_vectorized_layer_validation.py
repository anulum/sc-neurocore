# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — VectorizedSCLayer validation contracts

"""Verify constructor, input, packed-tail, and exported-payload validation."""

import numpy as np
import pytest

from sc_neurocore.layers.vectorized_layer import (
    VectorizedSCLayer,
    _as_float_array,
    _mask_unused_tail_bits,
)


def test_vectorized_reject_zero_inputs() -> None:
    with pytest.raises(ValueError, match="n_inputs must be >= 1"):
        VectorizedSCLayer(n_inputs=0, n_neurons=2, length=16)


def test_vectorized_reject_zero_neurons() -> None:
    with pytest.raises(ValueError, match="n_neurons must be >= 1"):
        VectorizedSCLayer(n_inputs=2, n_neurons=0, length=16)


def test_vectorized_reject_zero_length() -> None:
    with pytest.raises(ValueError, match="length must be >= 1"):
        VectorizedSCLayer(n_inputs=2, n_neurons=2, length=0)


def test_vectorized_reject_nan_input() -> None:
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=16)
    with pytest.raises(ValueError, match="NaN or Inf"):
        layer.forward([float("nan"), 0.5])


def test_vectorized_reject_out_of_range_input() -> None:
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=16)
    with pytest.raises(ValueError, match="probabilities must be in"):
        layer.forward([1.5, 0.5])


def test_vectorized_rejects_unknown_sc_mode() -> None:
    """Unknown modes fail closed rather than silently using AND semantics."""
    with pytest.raises(ValueError, match="sc_mode"):
        VectorizedSCLayer(n_inputs=2, n_neurons=1, sc_mode="ternary")


def test_vectorized_sparse_mode_requires_scipy_sparse_support() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "sc_neurocore.layers.vectorized_layer._has_scipy_sparse",
            lambda: False,
        )
        with pytest.raises(ImportError, match="scipy"):
            VectorizedSCLayer(n_inputs=4, n_neurons=8, sparse=True)


def test_mask_unused_tail_bits_masks_partial_final_word() -> None:
    """A non-word-aligned length zeroes unused high bits without mutating input."""
    packed = np.array([0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    masked = _mask_unused_tail_bits(packed, length=64 + 4)
    assert masked[-1] == np.uint64((1 << 4) - 1)
    assert masked[0] == np.uint64(0xFFFFFFFFFFFFFFFF)
    assert packed[-1] == np.uint64(0xFFFFFFFFFFFFFFFF)


def test_as_float_array_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        _as_float_array([1.0, np.nan], "weight")


def test_from_exported_weights_validates_payload() -> None:
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
