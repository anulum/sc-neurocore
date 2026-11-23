# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDequantizeWeights from former test_quantisation_pipeline.py

"""Focused suite: TestDequantizeWeights from former test_quantisation_pipeline.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.compiler.quantizer import (
    dequantize_weights,
)


class TestDequantizeWeights:
    """Converting fixed-point weights back to floats."""

    def test_known_value(self) -> None:
        """The Q8.8 value 256 dequantises to 1.0."""
        # 256 in Q8.8 = 1.0
        q = np.array([256])
        d = dequantize_weights(q, fmt="Q8.8")
        np.testing.assert_allclose(d, 1.0, atol=1e-6)

    def test_negative_known(self) -> None:
        """The Q8.8 value -256 dequantises to -1.0."""
        # -256 in Q8.8 = -1.0
        q = np.array([-256])
        d = dequantize_weights(q, fmt="Q8.8")
        np.testing.assert_allclose(d, -1.0, atol=1e-6)
