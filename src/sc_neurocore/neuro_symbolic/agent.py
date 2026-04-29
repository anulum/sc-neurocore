# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — High-level neuro-symbolic predictive agent

"""High-level neuro-symbolic predictive-coding agent API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sc_neurocore.neuro_symbolic.predictive_coding import (
    PredictiveCodingLayer,
    ReasoningTrace,
    SymbolEncoder,
    VerifiableInference,
)


@dataclass(frozen=True)
class PredictiveAgentConfig:
    """Configuration for a hybrid symbolic-spiking predictive agent."""

    input_dim: int
    hidden_dim: int
    symbols: tuple[str, ...] = ()
    lr: float = 0.01
    precision: float = 1.0
    seed: int = 0
    symbol_seed: int = 42

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.precision <= 0:
            raise ValueError("precision must be positive")


@dataclass(frozen=True)
class SCErrorSignature:
    """SC-domain prediction-error signature.

    `xor_bits` is the stochastic-computing error carrier. `popcount`
    is the integer error magnitude used by hardware-friendly decision
    logic.
    """

    xor_bits: tuple[int, ...]
    popcount: int
    normalised_popcount: float
    mean_abs_error: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "xor_bits": list(self.xor_bits),
            "popcount": self.popcount,
            "normalised_popcount": self.normalised_popcount,
            "mean_abs_error": self.mean_abs_error,
        }


@dataclass(frozen=True)
class HybridInferenceResult:
    """Result of one high-level neuro-symbolic predictive pass."""

    prediction: np.ndarray[Any, Any]
    error: np.ndarray[Any, Any]
    signature: SCErrorSignature
    symbol_scores: tuple[tuple[str, float], ...]
    trace: ReasoningTrace
    learned_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-compatible summary."""

        return {
            "error": self.error.tolist(),
            "learned_error": self.learned_error,
            "prediction": self.prediction.tolist(),
            "signature": self.signature.to_dict(),
            "symbol_scores": [
                {"symbol": symbol, "score": score} for symbol, score in self.symbol_scores
            ],
            "trace": self.trace.to_dict(),
        }


class NeuroSymbolicPredictiveAgent:
    """Hybrid predictive-coding agent for symbolic-spiking workflows."""

    def __init__(self, config: PredictiveAgentConfig):
        self.config = config
        self.encoder = SymbolEncoder(base_seed=config.symbol_seed)
        self.layer = PredictiveCodingLayer(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            lr=config.lr,
            precision=config.precision,
            seed=config.seed,
        )
        self.inference = VerifiableInference(self.encoder, self.layer)
        self.inference.register_symbols(config.symbols)

    @property
    def num_symbols(self) -> int:
        """Number of registered symbolic labels."""

        return self.inference.num_symbols

    def register_symbols(self, symbols: Sequence[str]) -> None:
        """Register additional symbolic labels."""

        self.inference.register_symbols(tuple(symbols))

    def observe(
        self,
        observation: np.ndarray[Any, Any] | Sequence[float],
        *,
        top_k: int = 1,
        learn: bool = False,
    ) -> HybridInferenceResult:
        """Run one predictive-symbolic observation pass."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        obs = self._validate_observation(observation)
        prediction = self.layer.predict()
        error = self.config.precision * (obs - prediction)
        signature = build_sc_error_signature(obs, prediction)
        scores, trace = self.inference.infer(obs, top_k=top_k)
        learned_error = self.layer.update(obs) if learn else None
        return HybridInferenceResult(
            prediction=prediction,
            error=error,
            signature=signature,
            symbol_scores=tuple((name, float(score)) for name, score in scores),
            trace=trace,
            learned_error=learned_error,
        )

    def _validate_observation(
        self, observation: np.ndarray[Any, Any] | Sequence[float]
    ) -> np.ndarray[Any, Any]:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim != 1:
            raise ValueError("observation must be one-dimensional")
        if obs.shape[0] != self.config.input_dim:
            raise ValueError(
                f"observation has {obs.shape[0]} elements, expected {self.config.input_dim}"
            )
        return obs


def build_sc_error_signature(
    observation: np.ndarray[Any, Any] | Sequence[float],
    prediction: np.ndarray[Any, Any] | Sequence[float],
) -> SCErrorSignature:
    """Build an XOR/popcount error signature from observation and prediction."""

    obs = np.asarray(observation, dtype=np.float32)
    pred = np.asarray(prediction, dtype=np.float32)
    if obs.shape != pred.shape:
        raise ValueError("observation and prediction must have matching shapes")
    if obs.ndim != 1:
        raise ValueError("observation and prediction must be one-dimensional")

    obs_bits = obs >= 0.0
    pred_bits = pred >= 0.0
    xor = np.logical_xor(obs_bits, pred_bits).astype(np.uint8)
    popcount = int(xor.sum())
    length = int(xor.shape[0])
    return SCErrorSignature(
        xor_bits=tuple(int(bit) for bit in xor.tolist()),
        popcount=popcount,
        normalised_popcount=popcount / max(length, 1),
        mean_abs_error=float(np.mean(np.abs(obs - pred))) if length else 0.0,
    )
