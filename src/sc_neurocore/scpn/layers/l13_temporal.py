# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L13 Source / Temporal Binding Layer (Stochastic

"""SCPN L13: source / temporal binding layer (stochastic implementation).

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
    """Stochastic configuration parameters for the SCPN source / temporal binding layer."""

    n_channels: int = 64
    bitstream_length: int = 1024
    binding_window: int = 10  # timesteps
    binding_threshold: float = 0.5
    quantum_info_coupling: float = 0.1  # from L12
    source_decoherence_coupling: float = 0.1
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
        """Advance the source / temporal binding layer one timestep and return its output state."""
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.time += dt
        self.step_count += 1
        n = self.params.n_channels

        # Shift history and add current state
        signal = np.zeros(n, dtype=np.float64)
        source_sampling_gain = 0.0
        temporal_decoherence_load = 0.0
        boundary_context_id: Optional[str] = None
        boundary_terminals: tuple[str, ...] = ()
        source_terminal_set: tuple[str, ...] = ()
        source_sampling_bandwidth = 1.0
        if l12_input is not None:
            signal = self._coherence_signal(l12_input.get("coherence", np.zeros(n)), n)
            l12_effect = self._l12_source_sampling_effect(l12_input)
            source_sampling_gain = l12_effect["source_sampling_gain"]
            temporal_decoherence_load = l12_effect["temporal_decoherence_load"]
            boundary_context_id = l12_effect["boundary_context_id"]
            boundary_terminals = l12_effect["boundary_terminals"]
            source_terminal_set = l12_effect["source_terminal_set"]
            source_sampling_bandwidth = l12_effect["source_sampling_bandwidth"]
            signal = np.clip(
                signal
                + source_sampling_gain
                - self.params.source_decoherence_coupling * temporal_decoherence_load,
                0.0,
                1.0,
            )

        self.history[...] = np.roll(self.history, -1, axis=1)
        self.history[:, -1] = signal

        # Max-lag cross-correlation binding over the temporal window.
        if self.step_count >= self.params.binding_window:
            self.binding_matrix = self._max_lag_binding_matrix(self.history)

        off_diagonal = ~np.eye(n, dtype=bool)
        bound_pairs = np.count_nonzero(
            np.abs(self.binding_matrix[off_diagonal]) > self.params.binding_threshold
        )
        binding_strength = float(bound_pairs / max(n * (n - 1), 1))

        activation = np.clip(np.diag(self.binding_matrix) * 0.5 + 0.5, 0, 1)
        rands = self._rng.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < activation[:, None]).astype(np.uint8)

        return {
            "binding_matrix": self.binding_matrix.copy(),
            "binding_strength": binding_strength,
            "source_sampling_signal": signal.copy(),
            "source_sampling_gain": source_sampling_gain,
            "temporal_decoherence_load": temporal_decoherence_load,
            "boundary_context_id": boundary_context_id,
            "boundary_terminals": boundary_terminals,
            "source_terminal_set": source_terminal_set,
            "source_sampling_bandwidth": source_sampling_bandwidth,
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        """Return the scalar global metric summarising this layer's state."""
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
        if (
            not math.isfinite(float(params.quantum_info_coupling))
            or params.quantum_info_coupling < 0.0
        ):
            raise ValueError("quantum_info_coupling must be finite and non-negative")
        if (
            not math.isfinite(float(params.source_decoherence_coupling))
            or params.source_decoherence_coupling < 0.0
        ):
            raise ValueError("source_decoherence_coupling must be finite and non-negative")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    @staticmethod
    def _coherence_signal(coherence: Any, n_channels: int) -> np.ndarray[Any, Any]:
        values = np.asarray(coherence, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("coherence must contain at least one value")
        if not np.all(np.isfinite(values)):
            raise ValueError("coherence must contain only finite values")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("coherence values must be within [0, 1]")
        if values.size >= n_channels:
            return values[:n_channels].copy()
        return np.pad(values, (0, n_channels - values.size))

    def _l12_source_sampling_effect(self, l12_input: Dict[str, Any]) -> Dict[str, Any]:
        source_context = self._source_context(l12_input)
        source_sampling_gain = (
            self.params.quantum_info_coupling
            * self._scalar(
                l12_input.get("gaian_stabilization_drive", 0.0),
                "gaian_stabilization_drive",
                lower_bound=None,
            )
            * source_context["source_sampling_bandwidth"]
        )
        noospheric_entropy_load = self._scalar(
            l12_input.get("noospheric_entropy_load", 0.0), "noospheric_entropy_load"
        )
        effective_dephasing_gamma = self._scalar(
            l12_input.get("effective_dephasing_gamma", 0.0), "effective_dephasing_gamma"
        )
        return {
            "source_sampling_gain": source_sampling_gain,
            "temporal_decoherence_load": noospheric_entropy_load + effective_dephasing_gamma,
            "boundary_context_id": source_context["boundary_context_id"],
            "boundary_terminals": source_context["boundary_terminals"],
            "source_terminal_set": source_context["source_terminal_set"],
            "source_sampling_bandwidth": source_context["source_sampling_bandwidth"],
        }

    @staticmethod
    def _source_context(l12_input: Dict[str, Any]) -> Dict[str, Any]:
        has_context_id = "boundary_context_id" in l12_input
        has_terminals = "boundary_terminals" in l12_input
        if not has_context_id and not has_terminals:
            return {
                "boundary_context_id": None,
                "boundary_terminals": (),
                "source_terminal_set": (),
                "source_sampling_bandwidth": 1.0,
            }
        if not has_context_id or not has_terminals:
            raise ValueError("boundary context requires boundary_context_id and boundary_terminals")

        raw_context_id = l12_input["boundary_context_id"]
        terminals = tuple(l12_input["boundary_terminals"])
        if raw_context_id is None and not terminals:
            return {
                "boundary_context_id": None,
                "boundary_terminals": (),
                "source_terminal_set": (),
                "source_sampling_bandwidth": 1.0,
            }
        context_id = str(raw_context_id)
        if not context_id:
            raise ValueError("boundary_context_id must be non-empty")
        valid_terminals = {"T1", "T2", "T3", "T4", "T5", "T6", "T7"}
        if not terminals or any(terminal not in valid_terminals for terminal in terminals):
            raise ValueError("boundary_terminals must contain valid T1-T7 terminal identifiers")

        source_terminals = tuple(terminal for terminal in terminals if terminal in {"T5", "T6"})
        return {
            "boundary_context_id": context_id,
            "boundary_terminals": terminals,
            "source_terminal_set": source_terminals,
            "source_sampling_bandwidth": float(len(source_terminals) / 2.0),
        }

    @staticmethod
    def _scalar(value: Any, name: str, *, lower_bound: Optional[float] = 0.0) -> float:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != ():
            raise ValueError(f"{name} must be a finite scalar")
        scalar = float(values)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be a finite scalar")
        if lower_bound is not None and scalar < lower_bound:
            raise ValueError(f"{name} must be finite and non-negative")
        return scalar

    @staticmethod
    def _pearson(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        if a.size < 2 or b.size < 2:
            return 0.0
        a0 = a - float(np.mean(a))
        b0 = b - float(np.mean(b))
        denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a0, b0) / denom)

    def _max_lag_binding_matrix(self, history: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
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
