# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
from __future__ import annotations
import numpy as np
from sc_neurocore.residual import MembraneShortcutBlock, SEWBlock, DeepSNNStack


class TestMembraneShortcutBlock:
    def test_forward(self) -> None:
        b = MembraneShortcutBlock(n_features=8)
        assert b.forward(np.random.rand(8)).shape == (8,)

    def test_reset(self) -> None:
        b = MembraneShortcutBlock(n_features=4)
        b.forward(np.ones(4))
        b.reset()
        assert np.allclose(b._v, 0)


class TestSEWBlock:
    def test_forward(self) -> None:
        b = SEWBlock(n_features=8)
        out = b.forward(np.random.rand(8))
        assert out.shape == (8,) and out.max() <= 1.0

    def test_reset(self) -> None:
        """Reset should clear the SEW membrane state."""
        b = SEWBlock(n_features=4)
        b._v[:] = 0.5
        b.reset()
        assert np.allclose(b._v, 0)


class TestDeepSNNStack:
    def test_ms_stack(self) -> None:
        s = DeepSNNStack(n_features=8, n_blocks=5, block_type="ms")
        assert s.depth == 10
        assert s.forward(np.random.rand(8)).shape == (8,)

    def test_sew_stack(self) -> None:
        s = DeepSNNStack(n_features=8, n_blocks=3, block_type="sew")
        assert s.forward(np.random.rand(8)).shape == (8,)
        assert s.n_blocks == 3

    def test_deep_50(self) -> None:
        s = DeepSNNStack(n_features=16, n_blocks=50)
        out = s.forward(np.random.rand(16))
        assert out.shape == (16,) and np.all(np.isfinite(out))

    def test_reset_clears_all_blocks(self) -> None:
        """Stack reset should delegate to every residual block."""
        s = DeepSNNStack(n_features=4, n_blocks=3, block_type="sew")
        for block in s.blocks:
            block._v[:] = 0.5

        s.reset()

        assert s.n_blocks == 3
        for block in s.blocks:
            assert np.allclose(block._v, 0)
