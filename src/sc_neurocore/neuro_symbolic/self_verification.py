# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuro-symbolic self-verification trace

"""Checked self-verification traces for neuro-symbolic inference results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np

from sc_neurocore.neuro_symbolic.agent import HybridInferenceResult, SCErrorSignature
from sc_neurocore.neuro_symbolic.predictive_coding import ReasoningTrace


SCHEMA_VERSION = "sc-neurocore.neuro-symbolic.self-verification.v1"


class VerificationStatus(Enum):
    """Status of one self-verification obligation."""

    # Verification outcome label, not a credential.
    PASS = "pass"  # nosec B105
    FAIL = "fail"


@dataclass(frozen=True)
class VerificationObligation:
    """One checked condition in a self-verification trace."""

    name: str
    status: VerificationStatus
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready obligation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class NeuroSymbolicSelfVerificationTrace:
    """Machine-checkable summary of a neuro-symbolic inference result."""

    schema_version: str
    result_digest: str
    obligations: tuple[VerificationObligation, ...]
    reasoning_steps: int
    top_symbols: tuple[str, ...]
    sc_popcount: int
    sc_normalised_popcount: float

    @property
    def passed(self) -> bool:
        """Whether every obligation passed."""
        return all(item.status == VerificationStatus.PASS for item in self.obligations)

    @property
    def failed_obligations(self) -> tuple[str, ...]:
        """Names of failed obligations."""
        return tuple(
            item.name for item in self.obligations if item.status == VerificationStatus.FAIL
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready trace."""
        return {
            "schema_version": self.schema_version,
            "result_digest": self.result_digest,
            "passed": self.passed,
            "failed_obligations": list(self.failed_obligations),
            "reasoning_steps": self.reasoning_steps,
            "top_symbols": list(self.top_symbols),
            "sc_popcount": self.sc_popcount,
            "sc_normalised_popcount": self.sc_normalised_popcount,
            "obligations": [item.to_dict() for item in self.obligations],
        }


class NeuroSymbolicSelfVerifier:
    """Build checked self-verification traces for inference outputs."""

    def verify_result(
        self,
        result: HybridInferenceResult,
        *,
        observation: np.ndarray[Any, Any] | Sequence[float],
    ) -> NeuroSymbolicSelfVerificationTrace:
        """Verify a high-level hybrid inference result against its observation."""
        obs = self._validate_vector(observation, "observation")
        prediction = self._validate_vector(result.prediction, "prediction")
        error = self._validate_vector(result.error, "error")
        obligations = (
            self._check_shape("prediction_shape", prediction, obs),
            self._check_shape("error_shape", error, obs),
            self._check_prediction_error(obs, prediction, error),
            self._check_signature(obs, prediction, result.signature),
            self._check_reasoning_trace(result.trace),
            self._check_symbol_scores(result.symbol_scores),
        )
        return self._build_trace(result, obligations)

    def verify_trace_only(
        self,
        trace: ReasoningTrace,
        *,
        symbol_scores: Sequence[tuple[str, float]] = (),
        signature: SCErrorSignature | None = None,
    ) -> NeuroSymbolicSelfVerificationTrace:
        """Verify a trace when only symbolic evidence is available."""
        obligations = (
            self._check_reasoning_trace(trace),
            self._check_symbol_scores(tuple(symbol_scores)),
        )
        payload = {
            "trace": trace.to_dict(),
            "symbol_scores": [(name, float(score)) for name, score in symbol_scores],
            "signature": signature.to_dict() if signature is not None else None,
        }
        digest = _stable_digest(payload)
        return NeuroSymbolicSelfVerificationTrace(
            schema_version=SCHEMA_VERSION,
            result_digest=digest,
            obligations=obligations,
            reasoning_steps=trace.length,
            top_symbols=tuple(name for name, _score in symbol_scores),
            sc_popcount=signature.popcount if signature is not None else 0,
            sc_normalised_popcount=signature.normalised_popcount if signature is not None else 0.0,
        )

    @staticmethod
    def _validate_vector(
        values: np.ndarray[Any, Any] | Sequence[float], name: str
    ) -> np.ndarray[Any, Any]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")
        return arr

    @staticmethod
    def _check_shape(
        name: str, lhs: np.ndarray[Any, Any], rhs: np.ndarray[Any, Any]
    ) -> VerificationObligation:
        status = VerificationStatus.PASS if lhs.shape == rhs.shape else VerificationStatus.FAIL
        return VerificationObligation(
            name=name,
            status=status,
            evidence={"lhs_shape": list(lhs.shape), "rhs_shape": list(rhs.shape)},
        )

    @staticmethod
    def _check_prediction_error(
        observation: np.ndarray[Any, Any],
        prediction: np.ndarray[Any, Any],
        error: np.ndarray[Any, Any],
    ) -> VerificationObligation:
        expected = observation - prediction
        residual = float(np.max(np.abs(expected - error))) if expected.size else 0.0
        status = VerificationStatus.PASS if residual <= 1e-6 else VerificationStatus.FAIL
        return VerificationObligation(
            name="prediction_error_consistency",
            status=status,
            evidence={"max_abs_residual": residual, "tolerance": 1e-6},
        )

    @staticmethod
    def _check_signature(
        observation: np.ndarray[Any, Any],
        prediction: np.ndarray[Any, Any],
        signature: SCErrorSignature,
    ) -> VerificationObligation:
        expected_bits = np.logical_xor(observation >= 0.0, prediction >= 0.0).astype(np.uint8)
        expected_tuple = tuple(int(bit) for bit in expected_bits.tolist())
        expected_popcount = int(expected_bits.sum())
        expected_normalised = expected_popcount / max(int(expected_bits.shape[0]), 1)
        valid = (
            signature.xor_bits == expected_tuple
            and signature.popcount == expected_popcount
            and abs(signature.normalised_popcount - expected_normalised) <= 1e-12
            and 0.0 <= signature.normalised_popcount <= 1.0
        )
        return VerificationObligation(
            name="sc_signature_consistency",
            status=VerificationStatus.PASS if valid else VerificationStatus.FAIL,
            evidence={
                "expected_popcount": expected_popcount,
                "actual_popcount": signature.popcount,
                "expected_xor_bits": list(expected_tuple),
                "actual_xor_bits": list(signature.xor_bits),
                "actual_normalised_popcount": signature.normalised_popcount,
            },
        )

    @staticmethod
    def _check_reasoning_trace(trace: ReasoningTrace) -> VerificationObligation:
        confidence_ok = all(0.0 <= step.confidence <= 1.0 for step in trace.steps)
        similarity_ok = all(-1.0 <= step.similarity <= 1.0 for step in trace.steps)
        timestamps_ok = all(step.timestamp_ns >= trace.start_ns for step in trace.steps)
        complete = trace.is_complete
        status = (
            VerificationStatus.PASS
            if confidence_ok and similarity_ok and timestamps_ok and complete
            else VerificationStatus.FAIL
        )
        return VerificationObligation(
            name="reasoning_trace_bounds",
            status=status,
            evidence={
                "complete": complete,
                "length": trace.length,
                "confidence_bounds": confidence_ok,
                "similarity_bounds": similarity_ok,
                "timestamps_after_start": timestamps_ok,
                "mean_confidence": trace.mean_confidence,
            },
        )

    @staticmethod
    def _check_symbol_scores(
        symbol_scores: Sequence[tuple[str, float]],
    ) -> VerificationObligation:
        scores = [float(score) for _name, score in symbol_scores]
        symbols = [name for name, _score in symbol_scores]
        finite = all(np.isfinite(score) for score in scores)
        bounded = all(-1.0 <= score <= 1.0 for score in scores)
        sorted_desc = all(scores[idx] >= scores[idx + 1] for idx in range(len(scores) - 1))
        unique_symbols = len(symbols) == len(set(symbols))
        status = (
            VerificationStatus.PASS
            if finite and bounded and sorted_desc and unique_symbols
            else VerificationStatus.FAIL
        )
        return VerificationObligation(
            name="symbol_score_ordering",
            status=status,
            evidence={
                "scores": scores,
                "finite": finite,
                "bounded": bounded,
                "sorted_descending": sorted_desc,
                "unique_symbols": unique_symbols,
            },
        )

    @staticmethod
    def _build_trace(
        result: HybridInferenceResult,
        obligations: tuple[VerificationObligation, ...],
    ) -> NeuroSymbolicSelfVerificationTrace:
        payload = result.to_dict()
        digest = _stable_digest(payload)
        return NeuroSymbolicSelfVerificationTrace(
            schema_version=SCHEMA_VERSION,
            result_digest=digest,
            obligations=obligations,
            reasoning_steps=result.trace.length,
            top_symbols=tuple(name for name, _score in result.symbol_scores),
            sc_popcount=result.signature.popcount,
            sc_normalised_popcount=result.signature.normalised_popcount,
        )


def build_self_verification_trace(
    result: HybridInferenceResult,
    *,
    observation: np.ndarray[Any, Any] | Sequence[float],
) -> NeuroSymbolicSelfVerificationTrace:
    """Verify a high-level neuro-symbolic inference result against its observation."""
    return NeuroSymbolicSelfVerifier().verify_result(result, observation=observation)


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialise {type(value)!r}")
