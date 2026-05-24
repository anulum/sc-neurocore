# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for cognitive orchestrator contracts

"""Contracts for CognitiveOrchestrator module registration and pipeline execution."""

from __future__ import annotations

import numpy as np

from sc_neurocore.core.orchestrator import CognitiveOrchestrator
from sc_neurocore.core.tensor_stream import TensorStream


class _ForwardModule:
    def __init__(self, transform_fn):
        self.transform_fn = transform_fn

    def forward(self, values):
        return self.transform_fn(values)


class _StepModule:
    def __init__(self, factor=2.0):
        self.factor = factor
        self.v = 0.0

    def step(self, value):
        return value * self.factor

    def get_state(self):
        return {"v": self.v}


def test_orchestrator_executes_forward_pipeline_in_order() -> None:
    orchestrator = CognitiveOrchestrator()
    orchestrator.register_module("add", _ForwardModule(lambda values: values + 0.1))
    orchestrator.register_module("double", _ForwardModule(lambda values: values * 2.0))

    result = orchestrator.execute_pipeline(
        ["add", "double"],
        TensorStream.from_prob(np.array([0.2])),
    )

    np.testing.assert_allclose(result.to_prob(), np.array([0.6]))


def test_orchestrator_skips_missing_modules_without_mutating_stream() -> None:
    orchestrator = CognitiveOrchestrator()
    orchestrator.register_module("double", _ForwardModule(lambda values: values * 2.0))

    result = orchestrator.execute_pipeline(
        ["missing", "double"],
        TensorStream.from_prob(np.array([0.3])),
    )

    np.testing.assert_allclose(result.to_prob(), np.array([0.6]))


def test_orchestrator_supports_step_interface_modules() -> None:
    orchestrator = CognitiveOrchestrator()
    orchestrator.register_module("stepper", _StepModule(factor=3.0))

    result = orchestrator.execute_pipeline(
        ["stepper"],
        TensorStream.from_prob(np.array([0.1, 0.2])),
    )

    np.testing.assert_allclose(result.to_prob(), np.array([0.3, 0.6]))


def test_orchestrator_rejects_attention_to_unknown_module() -> None:
    orchestrator = CognitiveOrchestrator()

    orchestrator.set_attention("missing")

    assert orchestrator.attention_focus is None
