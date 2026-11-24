# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPhiEvaluator from former test_research_modules.py

"""Focused suite: TestPhiEvaluator from former test_research_modules.py."""

from __future__ import annotations

from speculative.analysis_consciousness import PhiEvaluator
import numpy as np
import pytest


class TestPhiEvaluator:
    """The entropy measure of the integrated-information evaluator."""

    def test_entropy_all_ones(self) -> None:
        """An all-ones stream has zero entropy."""
        bs = np.ones(100)
        assert PhiEvaluator.entropy(bs) == 0.0

    def test_entropy_all_zeros(self) -> None:
        """An all-zeros stream has zero entropy."""
        bs = np.zeros(100)
        assert PhiEvaluator.entropy(bs) == 0.0

    def test_entropy_balanced(self) -> None:
        """A balanced stream has an entropy of about one bit."""
        bs = np.array([0, 1] * 500)
        assert PhiEvaluator.entropy(bs) == pytest.approx(1.0, abs=0.01)

    def test_phi_1d_returns_zero(self) -> None:
        """1D snapshot should return 0."""
        assert PhiEvaluator.calculate_phi(np.array([0.5, 0.3])) == 0.0

    def test_phi_independent_neurons(self) -> None:
        """Independent random neurons should have low Phi."""
        rng = np.random.default_rng(42)
        data = (rng.random((4, 1000)) < 0.5).astype(np.uint8)
        phi = PhiEvaluator.calculate_phi(data)
        assert phi >= 0
        assert phi < 0.5  # weak integration

    def test_phi_correlated_neurons(self) -> None:
        """Perfectly correlated neurons should have higher Phi."""
        row = np.array([0, 1] * 500, dtype=np.uint8)
        data = np.stack([row, row, row])
        phi = PhiEvaluator.calculate_phi(data)
        # H(each) = 1.0, H(joint) = 1.0, so phi = 3*1 - 1 = 2
        assert phi > 1.0
