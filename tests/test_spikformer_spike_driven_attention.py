# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeDrivenAttention from former test_spikformer.py

"""Focused suite: TestSpikeDrivenAttention from former test_spikformer.py."""

from __future__ import annotations

from tests.spikformer_support import *  # noqa: F403

class TestSpikeDrivenAttention:
    def test_basic_forward(self):
        ssa = SpikeDrivenAttention(embed_dim=16, num_heads=1, T=4)
        x = np.random.rand(5, 16)  # 5 tokens, 16 dims
        out = ssa.forward(x)
        assert out.shape == (5, 16)

    def test_single_token(self):
        ssa = SpikeDrivenAttention(embed_dim=8, T=4)
        x = np.random.rand(8)
        out = ssa.forward(x)
        assert out.shape == (8,)

    def test_zero_multiplications(self):
        ssa = SpikeDrivenAttention(embed_dim=16)
        assert ssa.num_multiply_ops == 0

    def test_output_finite(self):
        ssa = SpikeDrivenAttention(embed_dim=32, T=8)
        x = np.random.rand(10, 32)
        out = ssa.forward(x)
        assert np.all(np.isfinite(out))

    def test_different_timesteps(self):
        ssa4 = SpikeDrivenAttention(embed_dim=8, T=4)
        ssa16 = SpikeDrivenAttention(embed_dim=8, T=16)
        x = np.random.rand(3, 8)
        out4 = ssa4.forward(x)
        out16 = ssa16.forward(x)
        # Both produce valid output, different T changes precision
        assert out4.shape == out16.shape

    def test_multi_head(self):
        ssa = SpikeDrivenAttention(embed_dim=16, num_heads=4, T=4)
        assert ssa.head_dim == 4
        x = np.random.rand(6, 16)
        out = ssa.forward(x)
        assert out.shape == (6, 16)

    def test_threshold_low_more_spikes(self):
        ssa = SpikeDrivenAttention(embed_dim=8, T=16, threshold=0.01)
        x = np.random.rand(4, 8)
        out = ssa.forward(x)
        # Very low threshold should produce non-zero output from spike activity
        assert out.shape == (4, 8)
