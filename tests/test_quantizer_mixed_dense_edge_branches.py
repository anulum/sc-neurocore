# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMixedDenseEdgeBranches from former test_quantizer.py

"""Focused suite: TestMixedDenseEdgeBranches from former test_quantizer.py."""

from __future__ import annotations

from typing import Any, cast
import numpy as np
import pytest
from sc_neurocore.compiler.quantizer import (
    compile_dense_mixed_precision,
    CompiledMixedDense,
    QFormatMixed,
)


class TestMixedDenseEdgeBranches:
    """Validation guards on direct construction and compilation, the 2-D input rejection, and the per-tensor-scaled envelope rounding path."""

    def test_post_init_rejects_non_mixed_format(self) -> None:
        """Construction refuses a format that is not QFormatMixed."""
        with pytest.raises(TypeError, match="fmt must be a QFormatMixed"):
            CompiledMixedDense(
                quantized_weights=np.array([[1, -2], [3, -4]], dtype=np.int64),
                tensor_scale=1.0,
                fmt=cast(Any, "not-a-format"),
            )

    def test_post_init_rejects_non_positive_or_non_finite_scale(self) -> None:
        """Construction refuses a tensor scale that is not finite and positive."""
        with pytest.raises(ValueError, match="tensor_scale must be finite and positive"):
            CompiledMixedDense(
                quantized_weights=np.array([[1, -2], [3, -4]], dtype=np.int64),
                tensor_scale=0.0,
                fmt=QFormatMixed(),
            )

    def test_post_init_rejects_non_2d_weight_matrix(self) -> None:
        """Construction refuses a weight matrix that is not 2-D."""
        with pytest.raises(ValueError, match="2-D dense weight matrix"):
            CompiledMixedDense(
                quantized_weights=np.array([1, 2, 3], dtype=np.int64),
                tensor_scale=1.0,
                fmt=QFormatMixed(),
            )

    def test_post_init_rejects_codes_outside_weight_format(self) -> None:
        """Construction refuses codes outside the weight format."""
        with pytest.raises(ValueError, match="exceed the configured weight format"):
            CompiledMixedDense(
                quantized_weights=np.array([[10**9, 0]], dtype=np.int64),
                tensor_scale=1.0,
                fmt=QFormatMixed(),
            )

    def test_input_codes_reject_non_vector_inputs(self) -> None:
        """Input encoding refuses inputs that are not a 1-D vector."""
        compiled = compile_dense_mixed_precision(
            np.array([[0.5, -0.25]], dtype=np.float64),
            fmt=QFormatMixed(scale_per_tensor=False),
        )
        with pytest.raises(ValueError, match="inputs must be a 1-D vector"):
            compiled.forward_float(np.array([[0.5, -0.25]], dtype=np.float64))

    def test_envelope_report_uses_scaled_rounding_when_tensor_scaled(self) -> None:
        """A tensor-scaled layer uses scaled rounding in its envelope report."""
        weights = np.array([[0.375, -0.125, 0.0625], [-0.5, 0.25, 0.125]], dtype=np.float64)
        inputs = np.array([0.5, -0.25, 0.125], dtype=np.float64)
        # Default (scale_per_tensor=True) yields tensor_scale != 1.0, so the
        # accumulator divisor differs from the weight scale and the envelope
        # takes the float-divide ceil path rather than the integer fast path.
        compiled = compile_dense_mixed_precision(weights)
        assert compiled.tensor_scale != 1.0

        envelope = compiled.precision_envelope_report(inputs)
        assert envelope.max_abs_bound_code >= 0
        assert envelope.observed_overflow_free is True

    def test_compile_rejects_non_mixed_format(self) -> None:
        """Compilation refuses a format that is not QFormatMixed."""
        with pytest.raises(TypeError, match="fmt must be a QFormatMixed"):
            compile_dense_mixed_precision(
                np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64),
                fmt=cast(Any, "not-a-format"),
            )

    def test_compile_rejects_non_tuple_quantization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Compilation refuses a quantisation that returns no tuple."""
        # A QFormatMixed must yield a (codes, tensor_scale) tuple; a scalar-format
        # result is a contract breach the compiler must reject rather than unpack.
        monkeypatch.setattr(
            "sc_neurocore.compiler.mixed_dense_quantization.quantize_weights",
            lambda *args, **kwargs: np.array([[1, -2], [3, -4]], dtype=np.int64),
        )
        with pytest.raises(TypeError, match="requires QFormatMixed quantization"):
            compile_dense_mixed_precision(
                np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64),
                fmt=QFormatMixed(),
            )
