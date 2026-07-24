# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamToProb from former test_tensor_stream.py

"""Focused suite: TestBitstreamToProb from former test_tensor_stream.py."""

from __future__ import annotations

from tests.tensor_stream_support import *  # noqa: F403


class TestBitstreamToProb:
    def test_all_ones(self):
        bits = np.ones((1, 100), dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        np.testing.assert_allclose(ts.to_prob(), 1.0)

    def test_all_zeros(self):
        bits = np.zeros((1, 100), dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        np.testing.assert_allclose(ts.to_prob(), 0.0)

    def test_half(self):
        bits = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        np.testing.assert_allclose(ts.to_prob(), 0.5)
