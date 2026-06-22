# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for high-level neuro-symbolic agent API

"""Tests for the hybrid predictive-coding agent wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.neuro_symbolic import (
    NeuroSymbolicPredictiveAgent,
    PredictiveAgentConfig,
    build_sc_error_signature,
)


def test_sc_error_signature_uses_xor_and_popcount() -> None:
    signature = build_sc_error_signature(
        np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
        np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32),
    )

    assert signature.xor_bits == (0, 1, 1, 0)
    assert signature.popcount == 2
    assert signature.normalised_popcount == 0.5
    assert signature.mean_abs_error == 1.0


def test_sc_error_signature_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        build_sc_error_signature([1.0, -1.0], [1.0])


def test_sc_error_signature_rejects_multi_dimensional_inputs() -> None:
    # Matching shapes pass the first guard, but a 2-D observation/prediction
    # pair has no single bitstream to XOR, so it is rejected.
    with pytest.raises(ValueError, match="must be one-dimensional"):
        build_sc_error_signature(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )


def test_agent_rejects_multi_dimensional_observation() -> None:
    agent = NeuroSymbolicPredictiveAgent(
        PredictiveAgentConfig(input_dim=2, hidden_dim=1, lr=0.01, precision=1.0)
    )

    with pytest.raises(ValueError, match="observation must be one-dimensional"):
        agent.observe(np.zeros((2, 2), dtype=np.float32))


def test_agent_observe_returns_symbols_trace_and_signature() -> None:
    agent = NeuroSymbolicPredictiveAgent(
        PredictiveAgentConfig(
            input_dim=4,
            hidden_dim=2,
            symbols=("left", "right", "rest"),
            seed=7,
        )
    )

    result = agent.observe(np.array([0.25, -0.2, 0.1, -0.1], dtype=np.float32), top_k=2)

    assert agent.num_symbols == 3
    assert result.prediction.shape == (4,)
    assert result.error.shape == (4,)
    assert len(result.symbol_scores) == 2
    assert result.trace.is_complete
    assert 0 <= result.signature.popcount <= 4
    assert result.learned_error is None


def test_agent_observe_can_apply_learning_step() -> None:
    agent = NeuroSymbolicPredictiveAgent(PredictiveAgentConfig(input_dim=4, hidden_dim=2))
    before = agent.layer.mu.copy()

    result = agent.observe([0.5, 0.25, -0.25, -0.5], learn=True)

    assert result.learned_error is not None
    assert not np.array_equal(before, agent.layer.mu)


def test_agent_registers_additional_symbols() -> None:
    agent = NeuroSymbolicPredictiveAgent(PredictiveAgentConfig(input_dim=3, hidden_dim=2))

    agent.register_symbols(("go", "stop"))

    assert agent.num_symbols == 2
    result = agent.observe([0.1, -0.1, 0.2], top_k=2)
    assert [symbol for symbol, _score in result.symbol_scores] != []


@pytest.mark.parametrize(
    "config",
    [
        PredictiveAgentConfig(input_dim=1, hidden_dim=1, lr=0.01, precision=1.0),
    ],
)
def test_agent_rejects_invalid_observation_shape(config: PredictiveAgentConfig) -> None:
    agent = NeuroSymbolicPredictiveAgent(config)

    with pytest.raises(ValueError, match="expected 1"):
        agent.observe([0.1, 0.2])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"input_dim": 0, "hidden_dim": 1}, "input_dim"),
        ({"input_dim": 1, "hidden_dim": 0}, "hidden_dim"),
        ({"input_dim": 1, "hidden_dim": 1, "lr": 0}, "lr"),
        ({"input_dim": 1, "hidden_dim": 1, "precision": 0}, "precision"),
    ],
)
def test_predictive_agent_config_validates_values(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PredictiveAgentConfig(**kwargs)


def test_agent_rejects_non_positive_top_k() -> None:
    agent = NeuroSymbolicPredictiveAgent(PredictiveAgentConfig(input_dim=2, hidden_dim=1))

    with pytest.raises(ValueError, match="top_k"):
        agent.observe([0.0, 0.0], top_k=0)
