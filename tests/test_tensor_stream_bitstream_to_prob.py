# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamToProb from former test_tensor_stream.py

"""Focused suite: TestBitstreamToProb from former test_tensor_stream.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.core.tensor_stream import (
    TensorStream,
)


class TestBitstreamToProb:
    """Converting a bitstream tensor to probabilities."""

    def test_all_ones(self) -> None:
        """All ones give probability one."""
        bits = np.ones((1, 100), dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        np.testing.assert_allclose(ts.to_prob(), 1.0)

    def test_all_zeros(self) -> None:
        """All zeros give probability zero."""
        bits = np.zeros((1, 100), dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        np.testing.assert_allclose(ts.to_prob(), 0.0)

    def test_half(self) -> None:
        """Half ones give probability one half."""
        bits = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        np.testing.assert_allclose(ts.to_prob(), 0.5)
