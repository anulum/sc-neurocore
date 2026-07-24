# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSyntheticEEGGenerator from former test_tcbo_demo.py

"""Focused suite: TestSyntheticEEGGenerator from former test_tcbo_demo.py."""

from __future__ import annotations

from tests.tcbo_demo_support import *  # noqa: F403


class TestSyntheticEEGGenerator(unittest.TestCase):
    def test_init(self):
        gen = SyntheticEEGGenerator(N=16)
        self.assertEqual(gen.theta.shape, (16,))

    def test_step_returns_phases(self):
        gen = SyntheticEEGGenerator(N=16)
        theta = gen.step()
        self.assertEqual(theta.shape, (16,))
        self.assertTrue(np.all(theta >= 0))
        self.assertTrue(np.all(theta < 2 * np.pi))

    def test_run_returns_history(self):
        gen = SyntheticEEGGenerator(N=16)
        h = gen.run(100)
        self.assertEqual(h.shape, (100, 16))

    def test_coupling_scale(self):
        gen = SyntheticEEGGenerator(N=16)
        gen.set_coupling_scale(2.0)
        self.assertTrue(np.allclose(gen.K, gen._K_base * 2.0))

    def test_anesthesia(self):
        gen = SyntheticEEGGenerator(N=16)
        k_before = gen.K.sum()
        gen.apply_anesthesia(0.9)
        self.assertLess(gen.K.sum(), k_before * 0.2)

    def test_alpha_boost(self):
        gen = SyntheticEEGGenerator(N=16)
        k_before = gen.K[1, :].sum()
        gen.apply_alpha_boost(2.0)
        self.assertGreater(gen.K[1, :].sum(), k_before)

    def test_order_parameter(self):
        gen = SyntheticEEGGenerator(N=16)
        R = gen.get_order_parameter()
        self.assertGreaterEqual(R, 0.0)
        self.assertLessEqual(R, 1.0)

    def test_reset(self):
        gen = SyntheticEEGGenerator(N=16, seed=42)
        gen.run(100)
        gen.reset(seed=42)
        self.assertEqual(gen._step_count, 0)
