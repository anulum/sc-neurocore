# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProbToBitstream from former test_tensor_stream.py

"""Focused suite: TestProbToBitstream from former test_tensor_stream.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.core.tensor_stream import (
    TensorStream,
)


class TestProbToBitstream:
    """Converting probabilities to bitstreams."""

    def test_output_shape(self) -> None:
        """There is one bitstream row per probability."""
        ts = TensorStream.from_prob(np.array([0.5, 0.3]))
        bits = ts.to_bitstream(length=1024)
        assert bits.shape == (2, 1024)

    def test_output_binary(self) -> None:
        """The bitstreams hold only zeros and ones."""
        ts = TensorStream.from_prob(np.array([0.7]))
        bits = ts.to_bitstream(length=512)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_probability_preserved(self) -> None:
        """The bitstream mean matches the probability."""
        np.random.seed(42)
        p = 0.65
        ts = TensorStream.from_prob(np.array([p]))
        bits = ts.to_bitstream(length=10000)
        recovered = np.mean(bits)
        np.testing.assert_allclose(recovered, p, atol=0.02)

    @pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_roundtrip_accuracy(self, p: float) -> None:
        """Each probability round-trips within tolerance."""
        np.random.seed(42)
        ts = TensorStream.from_prob(np.array([p]))
        bits = ts.to_bitstream(length=8192)
        ts_back = TensorStream(data=bits, domain="bitstream")
        recovered = ts_back.to_prob()[0]
        np.testing.assert_allclose(recovered, p, atol=0.03)
