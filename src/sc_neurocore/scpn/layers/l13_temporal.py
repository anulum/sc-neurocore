# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L13 Source / Temporal Binding Layer (Stochastic

"""
SCPN L13: Source / Temporal Binding Layer (Stochastic Implementation)

Cross-correlation temporal binding window: measures synchrony between
oscillator signals within a short time window to determine binding
strength.

Binding(i,j) = max_τ |Corr(x_i(t), x_j(t+τ))|,  |τ| < T_window

Ref: Paper 13 — Source Field and Temporal Binding.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L13_StochasticParameters:
    n_channels: int = 64
    bitstream_length: int = 1024
    binding_window: int = 10  # timesteps
    binding_threshold: float = 0.5
    quantum_info_coupling: float = 0.1  # from L12
    rng_seed: Optional[int] = None


class L13_TemporalLayer:
    """Temporal binding via cross-correlation within a sliding window."""

    def __init__(self, params: Optional[L13_StochasticParameters] = None):
        self.params = params or L13_StochasticParameters()
        self._validate_params(self.params)
        n = self.params.n_channels
        w = self.params.binding_window
        self.history = np.zeros((n, w))
        self.binding_matrix = np.zeros((n, n))
        self.step_count = 0
        self.time = 0.0
        self._rng = np.random.default_rng(self.params.rng_seed)

    def step(
        self,
        dt: float,
        l12_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.time += dt
        self.step_count += 1
        n = self.params.n_channels

        # Shift history and add current state
        signal = np.zeros(n, dtype=np.float64)
        if l12_input is not None and "coherence" in l12_input:
            signal = self._coherence_signal(l12_input["coherence"], n)

        self.history = np.roll(self.history, -1, axis=1)
        self.history[:, -1] = signal

        # Max-lag cross-correlation binding over the temporal window.
        if self.step_count >= self.params.binding_window:
            self.binding_matrix = self._max_lag_binding_matrix(self.history)

        bound_pairs = np.sum(np.abs(self.binding_matrix) > self.params.binding_threshold) - n
        binding_strength = float(bound_pairs / max(n * (n - 1), 1))

        activation = np.clip(np.diag(self.binding_matrix) * 0.5 + 0.5, 0, 1)
        rands = self._rng.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < activation[:, None]).astype(np.uint8)

        return {
            "binding_matrix": self.binding_matrix.copy(),
            "binding_strength": binding_strength,
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        n = self.params.n_channels
        off_diag = self.binding_matrix[~np.eye(n, dtype=bool)]
        return float(np.mean(np.abs(off_diag))) if len(off_diag) > 0 else 0.0

    @staticmethod
    def _validate_params(params: L13_StochasticParameters) -> None:
        if params.n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if params.binding_window <= 1:
            raise ValueError("binding_window must be greater than one")
        if not 0.0 <= params.binding_threshold <= 1.0:
            raise ValueError("binding_threshold must be in [0, 1]")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    @staticmethod
    def _coherence_signal(coherence: Any, n_channels: int) -> np.ndarray:
        values = np.asarray(coherence, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("coherence must contain at least one value")
        if not np.all(np.isfinite(values)):
            raise ValueError("coherence must contain only finite values")
        if values.size >= n_channels:
            return values[:n_channels].copy()
        return np.pad(values, (0, n_channels - values.size))

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or b.size < 2:
            return 0.0
        a0 = a - float(np.mean(a))
        b0 = b - float(np.mean(b))
        denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a0, b0) / denom)

    def _max_lag_binding_matrix(self, history: np.ndarray) -> np.ndarray:
        n, window = history.shape
        max_lag = min(self.params.binding_window - 1, window - 1)
        matrix = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                best = 0.0
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        corr = self._pearson(history[i, :lag], history[j, -lag:])
                    elif lag > 0:
                        corr = self._pearson(history[i, lag:], history[j, :-lag])
                    else:
                        corr = self._pearson(history[i], history[j])
                    if abs(corr) > abs(best):
                        best = corr
                matrix[i, j] = best
                matrix[j, i] = best
        return matrix
