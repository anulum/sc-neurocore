# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L14 Transdimensional Integration Layer (Stochastic

"""
SCPN L14: Transdimensional Integration Layer (Stochastic Implementation)

Weighted aggregation across all lower layers to produce a unified
coherence metric. Acts as the integration hub of the SCPN stack.

I_global = sum_n w_n * M_n  (weighted layer metrics)

Ref: Paper 14 — Transdimensional Resonance.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# Default weights for 13 lower-layer metrics (L1-L13)
_DEFAULT_WEIGHTS = np.array(
    [
        0.10,
        0.08,
        0.06,
        0.10,
        0.08,  # L1-L5
        0.06,
        0.08,
        0.08,
        0.07,
        0.07,  # L6-L10
        0.06,
        0.08,
        0.08,  # L11-L13
    ]
)


@dataclass
class L14_StochasticParameters:
    n_dimensions: int = 13  # one per lower layer
    bitstream_length: int = 1024
    integration_weights: Optional[np.ndarray] = None
    temporal_coupling: float = 0.1  # from L13

    def __post_init__(self) -> None:
        if self.integration_weights is None:
            self.integration_weights = _DEFAULT_WEIGHTS.copy()


class L14_IntegrationLayer:
    """Weighted integration across SCPN layer metrics."""

    def __init__(self, params: Optional[L14_StochasticParameters] = None):
        self.params = params or L14_StochasticParameters()
        self.layer_metrics = np.zeros(self.params.n_dimensions)
        self.integrated_coherence = 0.5
        self.time = 0.0

    def step(
        self,
        dt: float,
        layer_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        self.time += dt

        if layer_metrics is not None:
            values = list(layer_metrics.values())[: self.params.n_dimensions]
            self.layer_metrics[: len(values)] = values

        w = self.params.integration_weights
        self.integrated_coherence = float(np.dot(w, self.layer_metrics))

        activation = np.full(self.params.n_dimensions, self.integrated_coherence)
        activation = np.clip(activation, 0, 1)

        rands = np.random.random((self.params.n_dimensions, self.params.bitstream_length))
        output_bitstreams = (rands < activation[:, None]).astype(np.uint8)

        return {
            "integrated_coherence": self.integrated_coherence,
            "layer_metrics": self.layer_metrics.copy(),
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        return self.integrated_coherence
