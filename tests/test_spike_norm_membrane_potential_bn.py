# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMembranePotentialBN from former test_spike_norm.py

"""Focused suite: TestMembranePotentialBN from former test_spike_norm.py."""

from __future__ import annotations

from tests.spike_norm_support import *  # noqa: F403


class TestMembranePotentialBN:
    def test_forward(self):
        bn = MembranePotentialBN(n_features=8)
        out = bn.forward(np.random.randn(4, 8), training=True)
        assert out.shape == (4, 8)

    def test_fused_threshold(self):
        bn = MembranePotentialBN(n_features=4, threshold=1.0)
        bn.forward(np.random.randn(10, 4), training=True)
        fused = bn.fused_threshold()
        assert fused.shape == (4,)

    def test_inference_passthrough(self):
        bn = MembranePotentialBN(n_features=4)
        x = np.random.randn(3, 4)
        out = bn.forward(x, training=False)
        np.testing.assert_array_equal(out, x)
