# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestP_h1Lightweight from former test_tcbo_demo.py

"""Focused suite: TestP_h1Lightweight from former test_tcbo_demo.py."""

from __future__ import annotations

from tests.tcbo_demo_support import *  # noqa: F403


class TestP_h1Lightweight(unittest.TestCase):
    def test_coherent_high(self):
        history = np.tile(np.ones(16) * 1.0, (60, 1))
        history += np.random.default_rng(0).normal(0, 0.05, history.shape)
        p = _compute_p_h1_lightweight(history)
        self.assertGreater(p, 0.5)

    def test_incoherent_lower(self):
        rng = np.random.default_rng(42)
        history = rng.uniform(0, 2 * np.pi, (60, 16))
        p = _compute_p_h1_lightweight(history)
        self.assertLess(p, 0.8)

    def test_short_history_returns_zero(self):
        history = np.zeros((5, 16))
        p = _compute_p_h1_lightweight(history)
        self.assertEqual(p, 0.0)
