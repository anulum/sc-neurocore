# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L15 Meta-Cognitive / Consilium Layer (Stochastic

"""
SCPN L15: Meta-Cognitive / Consilium Layer (Stochastic Implementation)

Self-monitoring layer: compares integrated coherence against a target
attractor and computes the deviation signal for L16 Director feedback.

Error = |I_target - I_actual|
GCI = 1 - Error  (Global Coherence Index)

Ref: Paper 15 — Consilium and Universal Metric.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L15_StochasticParameters:
    n_monitors: int = 16  # one per SCPN layer
    bitstream_length: int = 1024
    target_coherence: float = 0.8
    smoothing_alpha: float = 0.1
    integration_coupling: float = 0.2  # from L14


class L15_MetaLayer:
    """Self-monitoring meta-cognitive layer with GCI computation."""

    def __init__(self, params: Optional[L15_StochasticParameters] = None):
        self.params = params or L15_StochasticParameters()
        self.gci = 0.5
        self.error_history = np.zeros(self.params.n_monitors)
        self.time = 0.0

    def step(
        self,
        dt: float,
        l14_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.time += dt

        actual = 0.5
        if l14_input is not None and "integrated_coherence" in l14_input:
            actual = l14_input["integrated_coherence"]

        error = abs(self.params.target_coherence - actual)
        self.gci = (1 - self.params.smoothing_alpha) * self.gci + self.params.smoothing_alpha * (
            1 - error
        )

        # Per-monitor error tracking (shift and append)
        self.error_history = np.roll(self.error_history, -1)  # type: ignore[assignment]
        self.error_history[-1] = error

        activation = np.full(self.params.n_monitors, np.clip(self.gci, 0, 1))
        rands = np.random.random((self.params.n_monitors, self.params.bitstream_length))
        output_bitstreams = (rands < activation[:, None]).astype(np.uint8)

        return {
            "gci": self.gci,
            "error": error,
            "error_trend": float(np.mean(self.error_history)),
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        return self.gci
