# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBlockFloatingQuantize from former test_quantizer.py

"""Focused suite: TestBlockFloatingQuantize from former test_quantizer.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.compiler.quantizer import (
    BlockFloatingMode,
    dequantize_block_floating,
    quantize_block_floating,
)


class TestBlockFloatingQuantize:
    """Validate block-floating quantization contracts."""

    def test_quantize_block_floating_roundtrip(self) -> None:
        """Block-floating quantisation round-trips within tolerance."""
        w = np.array([0.0, 0.25, -0.5, 0.75, -1.0, 1.0], dtype=np.float64)
        q, exponents = quantize_block_floating(w, fmt="BFP12E4X4", block_size=4, clip=True)
        recovered = dequantize_block_floating(q, exponents, fmt="BFP12E4X4")
        np.testing.assert_allclose(recovered, w, rtol=0.0, atol=0.02)

    def test_block_floating_exponent_range_matches_encoded_codes(self) -> None:
        """The exponent range matches the codes the mode can encode."""
        mode = BlockFloatingMode.from_aliases("BFP8E2X2")
        assert mode.min_exponent == -1
        assert mode.max_exponent == 2
        assert mode.metadata["exponent_max"] == 2

    def test_quantize_block_floating_block_size_conflict(self) -> None:
        """Two different block sizes are reported as a conflict."""
        w = np.array([1.0, 0.5, -0.25])
        with pytest.raises(ValueError, match="Block size conflict"):
            quantize_block_floating(w, fmt="BFP12E4X8", block_size=4)

    def test_quantize_block_floating_overflow_boundary_is_finite(self) -> None:
        """Values at the overflow boundary stay finite and inside the mantissa range."""
        fmt = "BFP8E2X2"
        mode = BlockFloatingMode.from_aliases(fmt)
        w = np.array([0.0, 7.0, -7.0, 112.0, -112.0, 120.0, -120.0], dtype=np.float64)
        q, exponents = quantize_block_floating(w, fmt=fmt, block_size=2, clip=True)

        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(exponents))
        assert np.all(np.abs(q) <= mode.mantissa_range)
        assert np.all(exponents >= 0)
        assert np.all(exponents <= (1 << mode.exponent_bits) - 1)

        restored = dequantize_block_floating(q, exponents, fmt=fmt)
        assert np.all(np.isfinite(restored))
        assert restored.shape == w.shape
