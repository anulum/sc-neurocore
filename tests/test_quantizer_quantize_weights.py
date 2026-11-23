# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantizeWeights from former test_quantizer.py

"""Focused suite: TestQuantizeWeights from former test_quantizer.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.compiler.quantizer import (
    dequantize_weights,
    quantize_weights,
)


class TestQuantizeWeights:
    """Quantising float weights to fixed point."""

    def test_roundtrip_identity(self) -> None:
        """Weights round-trip through Q8.8 within one step."""
        w = np.array([0.0, 0.5, 1.0, -1.0, -0.5])
        q = quantize_weights(w, fmt="Q8.8")
        assert isinstance(q, np.ndarray)
        r = dequantize_weights(q, fmt="Q8.8")
        np.testing.assert_allclose(r, w, atol=1 / 256)

    def test_nearest_rounding(self) -> None:
        """Nearest rounding sends 128.5 to the even 128."""
        # np.rint uses banker's rounding: 128.5 → 128 (round half to even)
        w = np.array([0.501953125])  # 128.5 / 256
        q = quantize_weights(w, fmt="Q8.8", rounding="nearest")
        assert isinstance(q, np.ndarray)
        assert q[0] == 128  # banker's rounding: 128.5 → 128 (even)

    def test_nearest_rounding_odd(self) -> None:
        """Nearest rounding sends 129.5 to the even 130."""
        # 129.5 → 130 (round half to even, 130 is even)
        w = np.array([129.5 / 256])
        q = quantize_weights(w, fmt="Q8.8", rounding="nearest")
        assert isinstance(q, np.ndarray)
        assert q[0] == 130

    def test_floor_rounding(self) -> None:
        """Floor rounding sends 128.5 down to 128."""
        w = np.array([0.501953125])
        q = quantize_weights(w, fmt="Q8.8", rounding="floor")
        assert isinstance(q, np.ndarray)
        assert q[0] == 128

    def test_stochastic_rounding_average(self) -> None:
        """Stochastic rounding averages between the two neighbours."""
        np.random.seed(42)
        w = np.array([0.501953125] * 10000)
        q = quantize_weights(w, fmt="Q8.8", rounding="stochastic")
        assert isinstance(q, np.ndarray)
        # Stochastic: half round up, half round down on average
        assert 128 < q.mean() < 129

    def test_clipping(self) -> None:
        """Out-of-range weights clip to the Q8.8 limits."""
        w = np.array([200.0, -200.0])
        q = quantize_weights(w, fmt="Q8.8", clip=True)
        assert isinstance(q, np.ndarray)
        assert q[0] == 32767  # max Q8.8
        assert q[1] == -32768  # min Q8.8

    def test_invalid_rounding_raises(self) -> None:
        """An unknown rounding mode is rejected."""
        with pytest.raises(ValueError, match="Unknown rounding"):
            quantize_weights(np.array([1.0]), rounding="random")
