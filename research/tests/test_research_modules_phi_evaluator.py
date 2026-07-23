# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPhiEvaluator from former test_research_modules.py

"""Focused suite: TestPhiEvaluator from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestPhiEvaluator:
    def test_entropy_all_ones(self):
        bs = np.ones(100)
        assert PhiEvaluator.entropy(bs) == 0.0

    def test_entropy_all_zeros(self):
        bs = np.zeros(100)
        assert PhiEvaluator.entropy(bs) == 0.0

    def test_entropy_balanced(self):
        bs = np.array([0, 1] * 500)
        assert PhiEvaluator.entropy(bs) == pytest.approx(1.0, abs=0.01)

    def test_phi_1d_returns_zero(self):
        """1D snapshot should return 0."""
        assert PhiEvaluator.calculate_phi(np.array([0.5, 0.3])) == 0.0

    def test_phi_independent_neurons(self):
        """Independent random neurons should have low Phi."""
        rng = np.random.default_rng(42)
        data = (rng.random((4, 1000)) < 0.5).astype(np.uint8)
        phi = PhiEvaluator.calculate_phi(data)
        assert phi >= 0
        assert phi < 0.5  # weak integration

    def test_phi_correlated_neurons(self):
        """Perfectly correlated neurons should have higher Phi."""
        row = np.array([0, 1] * 500, dtype=np.uint8)
        data = np.stack([row, row, row])
        phi = PhiEvaluator.calculate_phi(data)
        # H(each) = 1.0, H(joint) = 1.0, so phi = 3*1 - 1 = 2
        assert phi > 1.0
