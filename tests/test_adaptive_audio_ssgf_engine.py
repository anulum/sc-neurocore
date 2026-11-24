# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSSGFEngine from former test_adaptive_audio.py

"""Focused suite: TestSSGFEngine from former test_adaptive_audio.py."""

from __future__ import annotations

import numpy as np
import unittest
from sc_neurocore.audio import (
    SSGFEngine,
)
from sc_neurocore.audio.ssgf_engine import (
    SSGFConfig,
)


class TestSSGFEngine(unittest.TestCase):
    """The SSGF geometry engine and its audio mapping."""

    def test_init(self) -> None:
        """A new engine has 16 oscillator phases."""
        eng = SSGFEngine()
        self.assertEqual(eng.theta.shape[0], 16)

    def test_outer_step(self) -> None:
        """An outer step returns a positive float cost."""
        eng = SSGFEngine()
        cost = eng.outer_step()
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0)

    def test_W_symmetric(self) -> None:
        """The coupling matrix is symmetric."""
        eng = SSGFEngine()
        eng.outer_step()
        W = eng.W
        self.assertTrue(np.allclose(W, W.T, atol=1e-10))

    def test_W_non_negative(self) -> None:
        """The coupling matrix is non-negative."""
        eng = SSGFEngine()
        eng.outer_step()
        self.assertTrue(np.all(eng.W >= -1e-10))

    def test_W_zero_diagonal(self) -> None:
        """The coupling matrix has a zero diagonal."""
        eng = SSGFEngine()
        eng.outer_step()
        self.assertTrue(np.allclose(np.diag(eng.W), 0))

    def test_audio_mapping(self) -> None:
        """The audio mapping gives a binaural frequency inside [0.5, 40] Hz and an intensity."""
        eng = SSGFEngine()
        eng.outer_step()
        m = eng.get_audio_mapping()
        self.assertIn("binaural_hz", m)
        self.assertIn("intensity", m)
        self.assertGreaterEqual(m["binaural_hz"], 0.5)
        self.assertLessEqual(m["binaural_hz"], 40.0)

    def test_convergence(self) -> None:
        """Twenty outer steps end with a cost below twice the first."""
        eng = SSGFEngine(SSGFConfig(seed=42))
        costs = [eng.outer_step() for _ in range(20)]
        self.assertLess(costs[-1], costs[0] * 2)  # Should not diverge

    def test_state(self) -> None:
        """The state reports the global order, the audio mapping and the eigenvalues."""
        eng = SSGFEngine()
        eng.outer_step()
        s = eng.get_state()
        self.assertIn("R_global", s)
        self.assertIn("audio", s)
        self.assertIn("eigvals", s)

    def test_R_range(self) -> None:
        """The global order lies inside [0, 1]."""
        eng = SSGFEngine()
        eng.outer_step()
        self.assertGreaterEqual(eng.get_state()["R_global"], 0)
        self.assertLessEqual(eng.get_state()["R_global"], 1)
