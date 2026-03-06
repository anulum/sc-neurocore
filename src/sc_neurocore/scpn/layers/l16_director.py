# SPDX-License-Identifier: AGPL-3.0-or-later
"""
SCPN L16: Director / Cybernetic Closure Layer (Stochastic Implementation)

PI controller with Lyapunov-monitored recursive self-refinement.
The Director receives GCI from L15 and adjusts system-wide coupling
to maintain coherence above the target threshold.

H_rec = alignment_error + (1 - R_global) + entropy_flux  (Lyapunov candidate)
u(t) = Kp * e(t) + Ki * integral(e)  (PI control law)
Veto: active when entropy_proxy > threshold.

Ref: Paper 16 / SSGF l16_closure.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class L16_StochasticParameters:
    n_control_nodes: int = 10
    bitstream_length: int = 1024
    kp: float = 2.0
    ki: float = 0.5
    veto_threshold: float = 0.8
    target_gci: float = 0.8
    integral_clamp: float = 5.0
    meta_coupling: float = 0.2  # from L15


class L16_DirectorLayer:
    """Cybernetic closure with PI control and Lyapunov monitoring."""

    def __init__(self, params: Optional[L16_StochasticParameters] = None):
        self.params = params or L16_StochasticParameters()
        n = self.params.n_control_nodes
        self.will = np.full(n, 0.9)
        self.integral_error = 0.0
        self.entropy_proxy = 0.0
        self.veto_active = False
        self.h_rec = 0.0
        self.time = 0.0

    def step(
        self,
        dt: float,
        l15_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        self.time += dt
        n = self.params.n_control_nodes

        gci = 0.5
        if l15_input is not None and "gci" in l15_input:
            gci = l15_input["gci"]

        # PI controller
        error = self.params.target_gci - gci
        self.integral_error = np.clip(
            self.integral_error + error * dt,
            -self.params.integral_clamp,
            self.params.integral_clamp,
        )
        u = self.params.kp * error + self.params.ki * self.integral_error
        u = np.clip(u, -1, 1)

        # Entropy proxy (inverse of coherence stability)
        self.entropy_proxy = 0.9 * self.entropy_proxy + 0.1 * (1.0 - gci)

        # Veto
        self.veto_active = self.entropy_proxy > self.params.veto_threshold

        # Lyapunov candidate
        self.h_rec = abs(error) + (1 - gci) + self.entropy_proxy

        # Will update
        d_will = 0.1 * gci - 0.2 * self.entropy_proxy + 0.05 * u
        self.will = np.clip(self.will + d_will * dt, 0, 1)

        effective_will = self.will * (0.0 if self.veto_active else 1.0)
        rands = np.random.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < effective_will[:, None]).astype(np.uint8)

        return {
            "will": self.will.copy(),
            "control_signal": float(u),
            "veto_active": self.veto_active,
            "h_rec": self.h_rec,
            "entropy_proxy": self.entropy_proxy,
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        return float(np.mean(self.will))
