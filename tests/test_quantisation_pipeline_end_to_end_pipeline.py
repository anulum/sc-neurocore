# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEndToEndPipeline from former test_quantisation_pipeline.py

"""Focused suite: TestEndToEndPipeline from former test_quantisation_pipeline.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.compiler.quantizer import (
    dequantize_weights,
    quantize_weights,
)


class TestEndToEndPipeline:
    """The quantise, dequantise and compute pipeline."""

    def test_dot_product_fidelity(self) -> None:
        """Q8.8 dot product should be close to float dot product."""
        rng = np.random.default_rng(42)
        W = rng.uniform(-2.0, 2.0, (4, 8))
        x = rng.uniform(0.0, 1.0, 8)

        # Float reference
        y_float = W @ x

        # Q8.8 pipeline
        W_q = quantize_weights(W, fmt="Q8.8")
        assert isinstance(W_q, np.ndarray)
        W_deq = dequantize_weights(W_q, fmt="Q8.8")
        y_q88 = W_deq @ x

        # Error should be small (dominated by weight quantisation)
        mae = np.mean(np.abs(y_float - y_q88))
        assert mae < 0.1, f"dot product MAE {mae:.4f} too large"

    def test_multiple_formats(self) -> None:
        """Quantisation should work for at least Q8.8."""
        w = np.array([0.5, -0.5])
        q = quantize_weights(w, fmt="Q8.8")
        assert isinstance(q, np.ndarray)
        d = dequantize_weights(q, fmt="Q8.8")
        assert np.allclose(w, d, atol=0.005)
