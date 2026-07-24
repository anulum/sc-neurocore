# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeepSNNStack from former test_residual.py

"""Focused suite: TestDeepSNNStack from former test_residual.py."""

from __future__ import annotations

from tests.residual_support import *  # noqa: F403


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
