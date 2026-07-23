# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCMixedOp from former test_darts_sc_nas.py

"""Focused suite: TestSCMixedOp from former test_darts_sc_nas.py."""

from __future__ import annotations

from darts_sc_nas_support import *  # noqa: F403

class TestSCMixedOp(unittest.TestCase):
    def test_forward_shape(self) -> None:
        op = SCMixedOp(1, 16, 3, 1, 1)
        op.eval()
        x = torch.rand(2, 1, 8, 8)
        out = op(x)
        self.assertEqual(out.shape, (2, 16, 8, 8))

    def test_alphas_are_learnable(self) -> None:
        op = SCMixedOp(1, 16, 3, 1, 1)
        self.assertEqual(op.alphas.shape[0], 7)
        self.assertTrue(op.alphas.requires_grad)

    def test_expected_resource_cost_positive(self) -> None:
        op = SCMixedOp(1, 16, 3, 1, 1)
        luts, power = op.expected_resource_cost()
        self.assertGreater(luts.item(), 0)
        self.assertGreater(power.item(), 0)

    def test_extract_optimal_returns_valid_length(self) -> None:
        op = SCMixedOp(1, 16, 3, 1, 1)
        config = op.extract_optimal_config()
        self.assertIn(config, [64, 128, 256, 512, 1024, 2048, 4096])

    def test_seven_candidate_ops(self) -> None:
        op = SCMixedOp(1, 16, 3, 1, 1)
        self.assertEqual(len(op.ops), 7)
        self.assertEqual(op.num_ops, 7)

    def test_strided_conv_halves_spatial(self) -> None:
        op = SCMixedOp(16, 32, 3, 2, 1)
        op.eval()
        x = torch.rand(1, 16, 16, 16)
        out = op(x)
        self.assertEqual(out.shape, (1, 32, 8, 8))
