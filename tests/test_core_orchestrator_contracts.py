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


class _QuantumModule:
    def forward(self, values):
        return np.array([0.6 + 0.8j], dtype=complex)


class _Uint8Module:
    def forward(self, values):
        return np.array([0, 1, 1, 0], dtype=np.uint8)


def test_orchestrator_set_attention_focuses_registered_module() -> None:
    """set_attention focuses on a module that has been registered."""
    orchestrator = CognitiveOrchestrator()
    orchestrator.register_module("vision", _ForwardModule(lambda values: values))

    orchestrator.set_attention("vision")

    assert orchestrator.attention_focus == "vision"


def test_orchestrator_quantum_module_wraps_complex_output_as_quantum_stream() -> None:
    """A Quantum-named module is fed a bitstream and its complex output wraps as quantum."""
    orchestrator = CognitiveOrchestrator()
    orchestrator.register_module("q", _QuantumModule())

    result = orchestrator.execute_pipeline(["q"], TensorStream.from_prob(np.array([0.5])))

    assert result.domain == "quantum"


def test_orchestrator_uint8_output_wraps_as_bitstream_stream() -> None:
    """A uint8 forward output is re-wrapped into the bitstream domain."""
    orchestrator = CognitiveOrchestrator()
    orchestrator.register_module("b", _Uint8Module())

    result = orchestrator.execute_pipeline(["b"], TensorStream.from_prob(np.array([0.5])))

    assert result.domain == "bitstream"


def test_orchestrator_step_module_handles_scalar_stream() -> None:
    """A step module processes a 0-d (scalar) stream via the scalar branch."""
    orchestrator = CognitiveOrchestrator()
    orchestrator.register_module("s", _StepModule(factor=3.0))

    result = orchestrator.execute_pipeline(["s"], TensorStream.from_prob(np.array(0.2)))

    np.testing.assert_allclose(result.to_prob(), np.array(0.6))
