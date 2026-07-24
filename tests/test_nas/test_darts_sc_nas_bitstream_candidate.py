# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamCandidate from former test_darts_sc_nas.py

"""Focused suite: TestBitstreamCandidate from former test_darts_sc_nas.py."""

from __future__ import annotations

from darts_sc_nas_support import *  # noqa: F403


class TestBitstreamCandidate(unittest.TestCase):
    def test_eval_passthrough(self) -> None:
        op = BitstreamCandidate(256, 100.0, 1.0)
        op.eval()
        x = torch.tensor([0.3, 0.5, 0.7])
        out = op(x)
        self.assertTrue(torch.equal(x, out))

    def test_train_adds_noise(self) -> None:
        torch.manual_seed(42)
        op = BitstreamCandidate(64, 100.0, 1.0)
        op.train()
        x = torch.full((100,), 0.5)
        out = op(x)
        self.assertFalse(torch.equal(x, out))

    def test_output_clamped_to_unit(self) -> None:
        torch.manual_seed(0)
        op = BitstreamCandidate(64, 100.0, 1.0)
        op.train()
        x = torch.full((1000,), 0.5)
        out = op(x)
        self.assertTrue((out >= 0.0).all())
        self.assertTrue((out <= 1.0).all())

    def test_longer_bitstream_less_noise(self) -> None:
        torch.manual_seed(42)
        op_short = BitstreamCandidate(64, 100.0, 1.0)
        op_short.train()
        x = torch.full((1000,), 0.5)
        out_short = op_short(x)
        var_short = out_short.var().item()

        torch.manual_seed(42)
        op_long = BitstreamCandidate(4096, 100.0, 1.0)
        op_long.train()
        out_long = op_long(x)
        var_long = out_long.var().item()

        self.assertGreater(var_short, var_long)

    def test_cost_attributes(self) -> None:
        op = BitstreamCandidate(256, 42.0, 3.14)
        self.assertEqual(op.length, 256)
        self.assertAlmostEqual(op.lut_cost, 42.0)
        self.assertAlmostEqual(op.power_cost, 3.14)
