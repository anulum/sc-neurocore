# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSSGFEngine from former test_adaptive_audio.py

"""Focused suite: TestSSGFEngine from former test_adaptive_audio.py."""

from __future__ import annotations

from tests.adaptive_audio_support import *  # noqa: F403


class TestSSGFEngine(unittest.TestCase):
    def test_init(self) -> None:
        eng = SSGFEngine()
        self.assertEqual(eng.theta.shape[0], 16)

    def test_outer_step(self) -> None:
        eng = SSGFEngine()
        cost = eng.outer_step()
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0)

    def test_W_symmetric(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        W = eng.W
        self.assertTrue(np.allclose(W, W.T, atol=1e-10))

    def test_W_non_negative(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        self.assertTrue(np.all(eng.W >= -1e-10))

    def test_W_zero_diagonal(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        self.assertTrue(np.allclose(np.diag(eng.W), 0))

    def test_audio_mapping(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        m = eng.get_audio_mapping()
        self.assertIn("binaural_hz", m)
        self.assertIn("intensity", m)
        self.assertGreaterEqual(m["binaural_hz"], 0.5)
        self.assertLessEqual(m["binaural_hz"], 40.0)

    def test_convergence(self) -> None:
        eng = SSGFEngine(SSGFConfig(seed=42))
        costs = [eng.outer_step() for _ in range(20)]
        self.assertLess(costs[-1], costs[0] * 2)  # Should not diverge

    def test_state(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        s = eng.get_state()
        self.assertIn("R_global", s)
        self.assertIn("audio", s)
        self.assertIn("eigvals", s)

    def test_R_range(self) -> None:
        eng = SSGFEngine()
        eng.outer_step()
        self.assertGreaterEqual(eng.get_state()["R_global"], 0)
        self.assertLessEqual(eng.get_state()["R_global"], 1)
