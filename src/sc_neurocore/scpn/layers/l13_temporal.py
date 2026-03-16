# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

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
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L13_StochasticParameters:
    n_channels: int = 64
    bitstream_length: int = 1024
    binding_window: int = 10  # timesteps
    binding_threshold: float = 0.5
    quantum_info_coupling: float = 0.1  # from L12


class L13_TemporalLayer:
    """Temporal binding via cross-correlation within a sliding window."""

    def __init__(self, params: Optional[L13_StochasticParameters] = None):
        self.params = params or L13_StochasticParameters()
        n = self.params.n_channels
        w = self.params.binding_window
        self.history = np.zeros((n, w))
        self.binding_matrix = np.zeros((n, n))
        self.step_count = 0
        self.time = 0.0

    def step(
        self,
        dt: float,
        l12_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        self.time += dt
        self.step_count += 1
        n = self.params.n_channels

        # Shift history and add current state
        signal = np.random.uniform(0, 1, n)
        if l12_input is not None and "coherence" in l12_input:
            coh = l12_input["coherence"]
            signal[: len(coh)] = coh[:n] if len(coh) >= n else np.pad(coh, (0, n - len(coh)))

        self.history = np.roll(self.history, -1, axis=1)
        self.history[:, -1] = signal

        # Cross-correlation binding (simplified: Pearson on history)
        if self.step_count >= self.params.binding_window:
            normed = self.history - self.history.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(normed, axis=1, keepdims=True) + 1e-10
            normed /= norms
            self.binding_matrix = normed @ normed.T

        bound_pairs = np.sum(np.abs(self.binding_matrix) > self.params.binding_threshold) - n
        binding_strength = float(bound_pairs / max(n * (n - 1), 1))

        activation = np.clip(np.diag(self.binding_matrix) * 0.5 + 0.5, 0, 1)
        rands = np.random.random((n, self.params.bitstream_length))
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
