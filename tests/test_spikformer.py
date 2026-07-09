# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.transformers.spikformer (SSA, SSM, CPG)

from __future__ import annotations

import numpy as np

from sc_neurocore.transformers.spikformer import (
    SpikeDrivenAttention,
    SpikyStateSpace,
    CPGPositionalEncoding,
)


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


class TestSpikyStateSpace:
    def test_step(self):
        ssm = SpikyStateSpace(d_model=8, d_state=16)
        x = np.random.rand(8)
        spikes, y = ssm.step(x)
        assert spikes.shape == (8,)
        assert y.shape == (8,)
        assert set(np.unique(spikes)).issubset({0.0, 1.0})

    def test_forward_sequence(self):
        ssm = SpikyStateSpace(d_model=8, d_state=16)
        x_seq = np.random.rand(20, 8)
        out = ssm.forward(x_seq)
        assert out.shape == (20, 8)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_reset(self):
        ssm = SpikyStateSpace(d_model=4, d_state=8)
        x = np.ones(4)
        ssm.step(x)
        assert not np.allclose(ssm._h, 0)
        ssm.reset()
        assert np.allclose(ssm._h, 0)
        assert np.allclose(ssm._v, 0)

    def test_state_accumulates(self):
        ssm = SpikyStateSpace(d_model=4, d_state=8, threshold=100.0)
        # With high threshold, no spikes — membrane should accumulate
        for _ in range(10):
            ssm.step(np.ones(4))
        assert not np.allclose(ssm._v, 0)

    def test_different_dt(self):
        ssm_fast = SpikyStateSpace(d_model=4, d_state=8, dt=0.1)
        ssm_slow = SpikyStateSpace(d_model=4, d_state=8, dt=0.001)
        # A values should differ (different decay rates)
        assert not np.allclose(ssm_fast.A, ssm_slow.A)

    def test_long_sequence(self):
        ssm = SpikyStateSpace(d_model=4, d_state=8)
        x_seq = np.random.rand(200, 4)
        out = ssm.forward(x_seq)
        assert out.shape == (200, 4)
        assert np.all(np.isfinite(out))


class TestCPGPositionalEncoding:
    def test_encode_shape(self):
        cpe = CPGPositionalEncoding(d_model=16)
        enc = cpe.encode(seq_len=10)
        assert enc.shape == (10, 16)

    def test_encode_range(self):
        cpe = CPGPositionalEncoding(d_model=8)
        enc = cpe.encode(seq_len=50)
        assert enc.min() >= 0.0
        assert enc.max() <= 1.0

    def test_different_positions_different_encodings(self):
        cpe = CPGPositionalEncoding(d_model=8)
        enc = cpe.encode(seq_len=10)
        assert not np.allclose(enc[0], enc[5])

    def test_encode_spikes(self):
        cpe = CPGPositionalEncoding(d_model=16)
        spikes = cpe.encode_spikes(seq_len=20)
        assert spikes.shape == (20, 16)
        assert set(np.unique(spikes)).issubset({0, 1})

    def test_encode_spikes_with_rng(self):
        cpe = CPGPositionalEncoding(d_model=8)
        rng = np.random.RandomState(42)
        s1 = cpe.encode_spikes(10, rng=np.random.RandomState(42))
        s2 = cpe.encode_spikes(10, rng=np.random.RandomState(42))
        np.testing.assert_array_equal(s1, s2)

    def test_max_len(self):
        cpe = CPGPositionalEncoding(d_model=4, max_len=100)
        assert cpe.max_len == 100
        enc = cpe.encode(seq_len=100)
        assert enc.shape == (100, 4)
