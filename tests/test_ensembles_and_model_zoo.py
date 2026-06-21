# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the ensemble orchestrator and the model zoo

"""Contracts for the multi-agent ensemble orchestrator and the SC model zoo."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.core.orchestrator import CognitiveOrchestrator
from sc_neurocore.core.tensor_stream import TensorStream
from sc_neurocore.ensembles import EnsembleOrchestrator
from sc_neurocore.models import SCDigitClassifier, SCKeywordSpotter


class _ScaleModule:
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def forward(self, values: Any) -> Any:
        return values * self.factor


def _agent_with_scale(factor: float) -> CognitiveOrchestrator:
    agent = CognitiveOrchestrator()
    agent.register_module("scale", _ScaleModule(factor))
    return agent


def test_ensemble_consensus_averages_agent_outputs() -> None:
    """run_consensus runs the same pipeline on every agent and averages the outputs."""
    ensemble = EnsembleOrchestrator()
    ensemble.add_agent("a", _agent_with_scale(2.0))
    ensemble.add_agent("b", _agent_with_scale(4.0))

    out = ensemble.run_consensus(["scale"], TensorStream.from_prob(np.array([0.1])))

    np.testing.assert_allclose(out, np.array([0.3]))  # mean(0.2, 0.4)


def test_ensemble_coordinated_mission_assigns_subtasks() -> None:
    """coordinated_mission derives a per-agent sub-goal from the mission goal."""
    ensemble = EnsembleOrchestrator()
    agent = CognitiveOrchestrator()
    ensemble.add_agent("a", agent)

    ensemble.coordinated_mission("explore")

    assert agent.active_goals == ["explore_subtask"]


def test_digit_classifier_accepts_2d_and_3d_images() -> None:
    """SCDigitClassifier.forward returns a digit class 0-9 for 2-D and 3-D inputs."""
    classifier = SCDigitClassifier()

    label_2d = classifier.forward(np.zeros((28, 28)))
    label_3d = classifier.forward(np.zeros((1, 28, 28)))

    for label in (label_2d, label_3d):
        assert isinstance(label, int)
        assert 0 <= label <= 9


def test_keyword_spotter_returns_class_within_vocabulary() -> None:
    """SCKeywordSpotter.predict returns a keyword index within the configured vocabulary."""
    spotter = SCKeywordSpotter(n_keywords=3)

    label = spotter.predict(np.zeros(16))

    assert isinstance(label, int)
    assert 0 <= label < 3
