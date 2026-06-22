# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuro-symbolic self-verification trace tests

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neuro_symbolic import (
    NeuroSymbolicPredictiveAgent,
    NeuroSymbolicSelfVerifier,
    PredictiveAgentConfig,
    SCErrorSignature,
    VerificationStatus,
    build_self_verification_trace,
)
from sc_neurocore.neuro_symbolic.agent import HybridInferenceResult


def test_self_verification_trace_passes_for_agent_result() -> None:
    observation = np.array([0.25, -0.2, 0.1, -0.1], dtype=np.float32)
    agent = NeuroSymbolicPredictiveAgent(
        PredictiveAgentConfig(
            input_dim=4,
            hidden_dim=2,
            symbols=("left", "right", "rest"),
            seed=3,
        )
    )
    result = agent.observe(observation, top_k=2)

    trace = build_self_verification_trace(result, observation=observation)
    payload = trace.to_dict()

    assert trace.passed
    assert trace.schema_version.endswith(".v1")
    assert len(trace.result_digest) == 64
    assert trace.reasoning_steps == result.trace.length
    assert trace.top_symbols == tuple(symbol for symbol, _score in result.symbol_scores)
    assert payload["passed"] is True
    assert payload["failed_obligations"] == []


def test_self_verification_is_deterministic_for_same_result() -> None:
    observation = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    agent = NeuroSymbolicPredictiveAgent(
        PredictiveAgentConfig(input_dim=3, hidden_dim=2, symbols=("go", "stop"), seed=5)
    )
    result = agent.observe(observation, top_k=2)
    verifier = NeuroSymbolicSelfVerifier()

    first = verifier.verify_result(result, observation=observation)
    second = verifier.verify_result(result, observation=observation)

    assert first.result_digest == second.result_digest
    assert first.to_dict() == second.to_dict()


def test_self_verification_detects_tampered_signature() -> None:
    observation = np.array([0.5, -0.5], dtype=np.float32)
    agent = NeuroSymbolicPredictiveAgent(
        PredictiveAgentConfig(input_dim=2, hidden_dim=2, symbols=("safe",), seed=7)
    )
    result = agent.observe(observation, top_k=1)
    tampered = HybridInferenceResult(
        prediction=result.prediction,
        error=result.error,
        signature=SCErrorSignature(
            xor_bits=(0, 0),
            popcount=0,
            normalised_popcount=0.0,
            mean_abs_error=result.signature.mean_abs_error,
        ),
        symbol_scores=result.symbol_scores,
        trace=result.trace,
        learned_error=result.learned_error,
    )

    trace = NeuroSymbolicSelfVerifier().verify_result(tampered, observation=observation)

    assert not trace.passed
    assert "sc_signature_consistency" in trace.failed_obligations
    obligation = next(item for item in trace.obligations if item.name == "sc_signature_consistency")
    assert obligation.status == VerificationStatus.FAIL


def test_self_verification_detects_unsorted_symbol_scores() -> None:
    observation = np.array([0.2, -0.1], dtype=np.float32)
    agent = NeuroSymbolicPredictiveAgent(
        PredictiveAgentConfig(input_dim=2, hidden_dim=2, symbols=("a", "b"), seed=9)
    )
    result = agent.observe(observation, top_k=2)
    tampered = HybridInferenceResult(
        prediction=result.prediction,
        error=result.error,
        signature=result.signature,
        symbol_scores=(("low", -0.5), ("high", 0.5)),
        trace=result.trace,
        learned_error=result.learned_error,
    )

    trace = NeuroSymbolicSelfVerifier().verify_result(tampered, observation=observation)

    assert not trace.passed
    assert "symbol_score_ordering" in trace.failed_obligations


def test_trace_only_mode_records_symbolic_evidence() -> None:
    observation = np.array([0.4, -0.4], dtype=np.float32)
    agent = NeuroSymbolicPredictiveAgent(
        PredictiveAgentConfig(input_dim=2, hidden_dim=2, symbols=("x", "y"), seed=11)
    )
    result = agent.observe(observation, top_k=2)

    trace = NeuroSymbolicSelfVerifier().verify_trace_only(
        result.trace,
        symbol_scores=result.symbol_scores,
        signature=result.signature,
    )

    assert trace.reasoning_steps == result.trace.length
    assert trace.top_symbols == tuple(symbol for symbol, _score in result.symbol_scores)
    assert trace.sc_popcount == result.signature.popcount


def test_json_default_serialises_numpy_and_rejects_unknown_types() -> None:
    from sc_neurocore.neuro_symbolic.self_verification import _json_default

    assert _json_default(np.array([1, 2])) == [1, 2]
    assert _json_default(np.float64(1.5)) == 1.5
    with pytest.raises(TypeError, match="cannot serialise"):
        _json_default(object())
