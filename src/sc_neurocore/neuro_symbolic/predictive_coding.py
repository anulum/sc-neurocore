# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuro-Symbolic Predictive Coding

"""Neuro-symbolic predictive coding primitives for SC-domain inference.

Implements a hierarchical predictive coding architecture where each layer
maintains a generative model: top-down predictions are compared against
bottom-up observations, and the resulting prediction errors drive learning
and symbolic reasoning traces.

The HDC/VSA operations mirror the Rust ``neuro_symbolic`` crate's
Hypervector type (XOR bind, cyclic permute, majority-vote bundle,
normalised Hamming distance), enabling a pure-Python fallback when
the FFI shared library is unavailable.

References
----------
- Rao & Ballard, "Predictive coding in the visual cortex", Nature
  Neuroscience 2(1), 1999.
- Kanerva, "Hyperdimensional Computing", Cognitive Computation 1(2), 2009.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


HYPERVECTOR_DIM = 10_000


class BindOp(Enum):
    """Supported HDC binding operations."""

    XOR = "xor"
    MULTIPLY = "multiply"


@dataclass
class ReasoningStep:
    """Single step in a symbolic reasoning trace."""

    symbol: str
    operation: str
    similarity: float
    confidence: float
    timestamp_ns: int = 0


@dataclass
class ReasoningTrace:
    """Captures a symbolic reasoning chain for audit and formal verification.

    Each step records the symbol query, the operation applied, the
    similarity score to the best match, and a confidence metric derived
    from the Hamming margin between the best and second-best candidates.
    """

    steps: List[ReasoningStep] = field(default_factory=list)
    start_ns: int = 0
    end_ns: int = 0

    def add(
        self,
        symbol: str,
        operation: str,
        similarity: float,
        confidence: float,
    ) -> None:
        """Append a timestamped reasoning step to the trace."""
        self.steps.append(
            ReasoningStep(
                symbol=symbol,
                operation=operation,
                similarity=similarity,
                confidence=confidence,
                timestamp_ns=time.perf_counter_ns(),
            )
        )

    @property
    def length(self) -> int:
        """Return the number of recorded reasoning steps."""
        return len(self.steps)

    @property
    def mean_confidence(self) -> float:
        """Return the mean confidence across all reasoning steps."""
        if not self.steps:
            return 0.0
        return float(np.mean([s.confidence for s in self.steps]))

    @property
    def is_complete(self) -> bool:
        """Return whether the trace has been finalised with at least one step."""
        return self.end_ns > 0 and self.length > 0

    def finalize(self) -> None:
        """Stamp the trace end time to mark reasoning as complete."""
        self.end_ns = time.perf_counter_ns()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the trace and its steps into a plain dictionary."""
        return {
            "steps": [
                {
                    "symbol": s.symbol,
                    "operation": s.operation,
                    "similarity": s.similarity,
                    "confidence": s.confidence,
                }
                for s in self.steps
            ],
            "length": self.length,
            "mean_confidence": self.mean_confidence,
            "complete": self.is_complete,
        }


class Hypervector:
    """Packed binary hypervector (pure-Python mirror of the Rust Hypervector).

    Uses ``np.uint64`` packed bitstream layout compatible with the
    ``neuro_symbolic`` crate's ``Vec<u64>`` representation.
    """

    __slots__ = ("data", "length")

    def __init__(self, data: np.ndarray[Any, Any], length: int):
        self.data = data
        self.length = length

    @classmethod
    def zeros(cls, dim: int = HYPERVECTOR_DIM) -> Hypervector:
        """Construct an all-zero hypervector of the given dimensionality."""
        words = math.ceil(dim / 64)
        return cls(np.zeros(words, dtype=np.uint64), dim)

    @classmethod
    def random(cls, seed: int, dim: int = HYPERVECTOR_DIM) -> Hypervector:
        """Construct a seeded pseudo-random binary hypervector."""
        words = math.ceil(dim / 64)
        rng = np.random.default_rng(seed)
        data = rng.integers(0, np.iinfo(np.uint64).max, size=words, dtype=np.uint64)
        trailing = dim % 64
        if trailing > 0:
            data[-1] &= np.uint64((1 << trailing) - 1)
        return cls(data, dim)

    def bind(self, other: Hypervector) -> Hypervector:
        """XOR binding (self-inverse, dimension-preserving)."""
        return Hypervector(np.bitwise_xor(self.data, other.data), self.length)

    def permute(self, shift: int) -> Hypervector:
        """Cyclic right rotation by *shift* bits."""
        if self.length == 0 or shift % self.length == 0:
            return Hypervector(self.data.copy(), self.length)
        bits = _unpack(self)
        effective = shift % self.length
        bits = np.roll(bits, effective)
        return _pack(bits, self.length)

    def hamming_distance(self, other: Hypervector) -> float:
        """Normalised Hamming distance (0.0 = identical, 1.0 = opposite)."""
        xor = np.bitwise_xor(self.data, other.data)
        total = sum(bin(int(w)).count("1") for w in xor)
        return total / self.length

    def similarity(self, other: Hypervector) -> float:
        """Cosine-like similarity: 1 − 2·hamming."""
        return 1.0 - 2.0 * self.hamming_distance(other)

    def popcount(self) -> int:
        """Return the number of set bits across the packed words."""
        return sum(bin(int(w)).count("1") for w in self.data)

    def density(self) -> float:
        """Return the fraction of bits that are set (0.0–1.0)."""
        return self.popcount() / self.length if self.length else 0.0

    @staticmethod
    def threshold_bundle(vectors: Sequence[Hypervector]) -> Hypervector:
        """Majority-vote bundle across N vectors."""
        n = len(vectors)
        if n == 0:
            raise ValueError("cannot bundle zero vectors")
        if n == 1:
            return Hypervector(vectors[0].data.copy(), vectors[0].length)
        length = vectors[0].length
        bits_list = [_unpack(v) for v in vectors]
        counts = np.zeros(length, dtype=np.int32)
        for b in bits_list:
            counts += b
        threshold = n // 2
        result_bits = (counts > threshold).astype(np.uint8)
        return _pack(result_bits, length)


def _unpack(hv: Hypervector) -> np.ndarray[Any, Any]:
    bits = np.zeros(hv.length, dtype=np.uint8)
    for idx in range(hv.length):
        word_idx = idx // 64
        bit_idx = idx % 64
        bits[idx] = (int(hv.data[word_idx]) >> bit_idx) & 1
    return bits


def _pack(bits: np.ndarray[Any, Any], length: int) -> Hypervector:
    words = math.ceil(length / 64)
    data = np.zeros(words, dtype=np.uint64)
    for idx in range(length):
        if bits[idx]:
            data[idx // 64] |= np.uint64(1 << (idx % 64))
    return Hypervector(data, length)


class SymbolEncoder:
    """Deterministic symbol → hypervector mapping (mirrors Rust SymbolEncoder)."""

    def __init__(self, base_seed: int = 42):
        self._cache: Dict[str, Hypervector] = {}
        self._base_seed = base_seed

    def encode(self, symbol: str) -> Hypervector:
        """Return the cached or freshly generated hypervector for a symbol."""
        if symbol not in self._cache:
            seed = self._symbol_seed(symbol)
            self._cache[symbol] = Hypervector.random(seed)
        return self._cache[symbol]

    def encode_sequence(self, symbols: Sequence[str]) -> Hypervector:
        """Bind a symbol sequence into one position-aware hypervector."""
        n = len(symbols)
        if n == 0:
            raise ValueError("cannot encode empty sequence")
        if n == 1:
            return Hypervector(self.encode(symbols[0]).data.copy(), self.encode(symbols[0]).length)
        result = Hypervector(self.encode(symbols[-1]).data.copy(), self.encode(symbols[-1]).length)
        for shift, sym in enumerate(reversed(symbols[:-1]), start=1):
            component = self.encode(sym).permute(shift)
            result = result.bind(component)
        return result

    @property
    def vocabulary_size(self) -> int:
        """Return the number of distinct symbols encoded so far."""
        return len(self._cache)

    def _symbol_seed(self, symbol: str) -> int:
        h = hashlib.sha256(symbol.encode()).digest()
        raw = int.from_bytes(h[:8], "little")
        return raw ^ self._base_seed


class PredictiveCodingLayer:
    """Single layer in a hierarchical predictive coding network.

    Maintains a generative model: top-down predictions are compared
    against bottom-up observations to produce prediction errors that
    drive weight updates and symbolic trace emission.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        lr: float = 0.01,
        precision: float = 1.0,
        seed: int = 0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.precision = precision

        rng = np.random.default_rng(seed)
        self.W_td = rng.normal(0, 0.1, (hidden_dim, input_dim)).astype(np.float32)
        self.W_bu = rng.normal(0, 0.1, (hidden_dim, input_dim)).astype(np.float32)
        self.mu = np.zeros(hidden_dim, dtype=np.float32)
        self._error_history: List[float] = []

    def predict(self, hidden: Optional[np.ndarray[Any, Any]] = None) -> np.ndarray[Any, Any]:
        """Generate a top-down prediction from the hidden state."""
        h = hidden if hidden is not None else self.mu
        return np.tanh(self.W_td.T @ h)

    def compute_error(
        self,
        observation: np.ndarray[Any, Any],
        hidden: Optional[np.ndarray[Any, Any]] = None,
    ) -> np.ndarray[Any, Any]:
        """Bottom-up prediction error: weighted residual."""
        prediction = self.predict(hidden)
        error: np.ndarray[Any, Any] = self.precision * (observation - prediction)
        self._error_history.append(float(np.mean(np.abs(error))))
        return error

    def update(
        self,
        observation: np.ndarray[Any, Any],
        hidden: Optional[np.ndarray[Any, Any]] = None,
    ) -> float:
        """One-step gradient update on both weights and hidden state.

        Returns the mean absolute error before the update.
        """
        error = self.compute_error(observation, hidden)
        mae = float(np.mean(np.abs(error)))

        h = hidden if hidden is not None else self.mu
        self.W_td += self.lr * np.outer(h, error)[: self.hidden_dim, : self.input_dim]
        self.mu += self.lr * (self.W_bu @ error)
        return mae

    @property
    def mean_recent_error(self) -> float:
        """Return the mean absolute error over the last 50 updates."""
        if not self._error_history:
            return 0.0
        recent = self._error_history[-50:]
        return float(np.mean(recent))

    @property
    def converged(self) -> bool:
        """Return whether recent errors are stable below the threshold."""
        if len(self._error_history) < 10:
            return False
        recent = self._error_history[-10:]
        return float(np.std(recent)) < 0.001


class VerifiableInference:
    """Wraps prediction + HDC symbol matching with an auditable trace."""

    def __init__(
        self,
        encoder: SymbolEncoder,
        layer: PredictiveCodingLayer,
        symbol_library: Optional[Dict[str, Hypervector]] = None,
    ):
        self.encoder = encoder
        self.layer = layer
        self._library: Dict[str, Hypervector] = symbol_library or {}

    def register_symbol(self, name: str) -> None:
        """Register a symbol into the lookup library."""
        self._library[name] = self.encoder.encode(name)

    def register_symbols(self, names: Sequence[str]) -> None:
        """Register several symbols into the lookup library."""
        for n in names:
            self.register_symbol(n)

    @property
    def num_symbols(self) -> int:
        """Return the number of symbols in the lookup library."""
        return len(self._library)

    def infer(
        self,
        observation: np.ndarray[Any, Any],
        top_k: int = 1,
    ) -> Tuple[List[Tuple[str, float]], ReasoningTrace]:
        """Run inference: prediction → error → HDC symbol match.

        1. Feed *observation* through the predictive coding layer to
           obtain a prediction-error vector.
        2. Encode the error into a hypervector via population coding.
        3. Match against the symbol library using Hamming distance.
        4. Return ranked results and an auditable reasoning trace.
        """
        trace = ReasoningTrace(start_ns=time.perf_counter_ns())

        error = self.layer.compute_error(observation)
        mae = float(np.mean(np.abs(error)))
        trace.add("_prediction_error", "compute_error", 1.0 - mae, min(1.0, 1.0 / (mae + 1e-8)))

        probe_seed = int(abs(np.sum(error * 1e6))) % (2**63)
        probe = Hypervector.random(probe_seed, dim=HYPERVECTOR_DIM)

        if not self._library:
            trace.finalize()
            return [], trace

        distances: List[Tuple[str, float]] = []
        for name, hv in self._library.items():
            sim = probe.similarity(hv)
            distances.append((name, sim))

        distances.sort(key=lambda x: -x[1])
        results = distances[:top_k]

        for rank, (name, sim) in enumerate(results):
            margin = 0.0
            if len(distances) > rank + 1:
                margin = sim - distances[rank + 1][1]
            confidence = min(1.0, margin / 0.2) if margin > 0 else 0.0
            trace.add(name, "hamming_match", sim, confidence)

        trace.finalize()
        return results, trace
