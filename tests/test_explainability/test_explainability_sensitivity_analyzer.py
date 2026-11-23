# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSensitivityAnalyzer from former test_explainability.py

"""Focused suite: TestSensitivityAnalyzer from former test_explainability.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.explainability.explainability import (
    ExplainabilityEngine,
    SensitivityAnalyzer,
    SpikeDecisionTree,
)


class TestSensitivityAnalyzer:
    """Sensitivity of spike decisions to perturbation."""

    def test_basic_sensitivity(self) -> None:
        """The default perturbations include one that flips the decision."""
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        bs[:60] = 1
        node = tree.add_decision("n0", bs, threshold=55)
        results = SensitivityAnalyzer.analyze(node)
        assert len(results) == 6  # default perturbations
        assert any(r.flipped for r in results)

    def test_custom_perturbations(self) -> None:
        """Custom perturbations give one result each."""
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=50)
        results = SensitivityAnalyzer.analyze(node, perturbations=[-1, 1])
        assert len(results) == 2

    def test_critical_delta_spike(self) -> None:
        """A spike five above threshold needs six to flip."""
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        bs[:60] = 1
        node = tree.add_decision("n0", bs, threshold=55)
        cd = SensitivityAnalyzer.critical_delta(node)
        assert cd == 6  # margin 5 → need +6 to flip

    def test_critical_delta_no_spike(self) -> None:
        """A silent neuron twenty below needs twenty to flip."""
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        bs[:30] = 1
        node = tree.add_decision("n0", bs, threshold=50)
        cd = SensitivityAnalyzer.critical_delta(node)
        assert cd == -20

    def test_engine_sensitivity(self) -> None:
        """The engine reports sensitivity results."""
        engine = ExplainabilityEngine(seed=0xACE1)
        node = engine.explain_spike("n0", 32768, 256, 100)
        results = engine.sensitivity(node)
        assert len(results) > 0
