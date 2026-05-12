# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L16 Director / Cybernetic Closure Layer (Stochastic

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
import math
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
    bridge_alignment_coupling: float = 0.2
    bridge_protection_coupling: float = 0.2
    rng_seed: Optional[int] = None


class L16_DirectorLayer:
    """Cybernetic closure with PI control and Lyapunov monitoring."""

    def __init__(self, params: Optional[L16_StochasticParameters] = None):
        self.params = params or L16_StochasticParameters()
        self._validate_params(self.params)
        self._rng = np.random.default_rng(self.params.rng_seed)
        n = self.params.n_control_nodes
        self.will = np.full(n, 0.9)
        self.integral_error = 0.0
        self.entropy_proxy = 0.0
        self.entropy_flux = 0.0
        self.veto_active = False
        self.h_rec = 0.0
        self.time = 0.0

    def step(
        self,
        dt: float,
        l15_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        (
            gci,
            ethical_dissonance,
            free_energy,
            bridge_alignment_credit,
            bridge_protection_penalty,
        ) = self._validate_step_inputs(dt, l15_input, self.params)
        self.time += dt
        n = self.params.n_control_nodes

        # PI controller
        coherence_error = self.params.target_gci - gci
        audited_error = (
            coherence_error
            + self.params.meta_coupling * ethical_dissonance
            + bridge_protection_penalty
            - bridge_alignment_credit
        )
        self.integral_error = np.clip(
            self.integral_error + audited_error * dt,
            -self.params.integral_clamp,
            self.params.integral_clamp,
        )
        u = self.params.kp * audited_error + self.params.ki * self.integral_error
        u = np.clip(u, -1, 1)

        # Entropy proxy (inverse coherence plus L15 free-energy leakage)
        self.entropy_flux = float(
            np.clip(
                (1.0 - gci)
                + self.params.meta_coupling * free_energy
                + bridge_protection_penalty,
                0.0,
                1.0,
            )
        )
        self.entropy_proxy = float(np.clip(0.9 * self.entropy_proxy + 0.1 * self.entropy_flux, 0.0, 1.0))

        # Veto
        self.veto_active = self.entropy_proxy > self.params.veto_threshold

        # Lyapunov candidate
        alignment_cost = max(
            0.0,
            abs(coherence_error)
            + ethical_dissonance
            + bridge_protection_penalty
            - bridge_alignment_credit,
        )
        self.h_rec = float(alignment_cost + self.entropy_proxy)

        # Will update
        d_will = self.params.meta_coupling * (0.1 * gci - 0.2 * self.entropy_proxy + 0.05 * u)
        self.will = np.clip(self.will + d_will * dt, 0, 1)

        effective_will = self.will * (0.0 if self.veto_active else 1.0)
        qecc_syndrome = np.full(
            n,
            1 if self.veto_active or bridge_protection_penalty > 0.0 else 0,
            dtype=np.uint8,
        )
        rands = self._rng.random((n, self.params.bitstream_length))
        output_bitstreams = (rands < effective_will[:, None]).astype(np.uint8)

        return {
            "will": self.will.copy(),
            "control_signal": float(u),
            "veto_active": self.veto_active,
            "h_rec": self.h_rec,
            "recursive_hamiltonian": self.h_rec,
            "entropy_flux": self.entropy_flux,
            "entropy_proxy": self.entropy_proxy,
            "closure_bridge_alignment_credit": bridge_alignment_credit,
            "closure_bridge_protection_penalty": bridge_protection_penalty,
            "effective_will": effective_will.copy(),
            "qecc_syndrome": qecc_syndrome,
            "output_bitstreams": output_bitstreams,
        }

    def get_global_metric(self) -> float:
        return float(np.mean(self.will))

    @staticmethod
    def _validate_params(params: L16_StochasticParameters) -> None:
        if params.n_control_nodes <= 0:
            raise ValueError("n_control_nodes must be positive")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")
        if not math.isfinite(params.kp):
            raise ValueError("kp must be finite")
        if not math.isfinite(params.ki):
            raise ValueError("ki must be finite")
        if not math.isfinite(params.veto_threshold) or not 0.0 <= params.veto_threshold <= 1.0:
            raise ValueError("veto_threshold must be finite and in [0, 1]")
        if not math.isfinite(params.target_gci) or not 0.0 <= params.target_gci <= 1.0:
            raise ValueError("target_gci must be finite and in [0, 1]")
        if not math.isfinite(params.integral_clamp) or params.integral_clamp <= 0.0:
            raise ValueError("integral_clamp must be finite and positive")
        if not math.isfinite(params.meta_coupling) or not 0.0 <= params.meta_coupling <= 1.0:
            raise ValueError("meta_coupling must be finite and in [0, 1]")
        if (
            not math.isfinite(params.bridge_alignment_coupling)
            or not 0.0 <= params.bridge_alignment_coupling <= 1.0
        ):
            raise ValueError("bridge_alignment_coupling must be finite and in [0, 1]")
        if (
            not math.isfinite(params.bridge_protection_coupling)
            or not 0.0 <= params.bridge_protection_coupling <= 1.0
        ):
            raise ValueError("bridge_protection_coupling must be finite and in [0, 1]")
        if params.rng_seed is not None and (
            isinstance(params.rng_seed, bool)
            or not isinstance(params.rng_seed, int)
            or params.rng_seed < 0
        ):
            raise ValueError("rng_seed must be non-negative when provided")

    @staticmethod
    def _validate_step_inputs(
        dt: float, l15_input: Optional[Dict[str, Any]], params: L16_StochasticParameters
    ) -> tuple[float, float, float, float, float]:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        if l15_input is None:
            return 0.5, 0.0, 0.0, 0.0, 0.0

        if "gci" not in l15_input:
            raise ValueError("l15_input must include gci")

        gci = float(l15_input["gci"])
        if not math.isfinite(gci) or not 0.0 <= gci <= 1.0:
            raise ValueError("l15 gci must be finite and in [0, 1]")

        ethical_dissonance = float(l15_input.get("ethical_dissonance", 0.0))
        if not math.isfinite(ethical_dissonance) or not 0.0 <= ethical_dissonance <= 1.0:
            raise ValueError("l15 ethical_dissonance must be finite and in [0, 1]")

        free_energy = float(l15_input.get("free_energy", 0.0))
        if not math.isfinite(free_energy) or not 0.0 <= free_energy <= 1.0:
            raise ValueError("l15 free_energy must be finite and in [0, 1]")

        bridge_alignment_credit = params.bridge_alignment_coupling * L16_DirectorLayer._unit_scalar(
            l15_input.get("bridge_alignment_credit", 0.0),
            "l15 bridge_alignment_credit",
        )
        bridge_protection_penalty = (
            params.bridge_protection_coupling
            * L16_DirectorLayer._nonnegative_scalar(
                l15_input.get("bridge_protection_penalty", 0.0),
                "l15 bridge_protection_penalty",
            )
        )

        return (
            gci,
            ethical_dissonance,
            free_energy,
            bridge_alignment_credit,
            bridge_protection_penalty,
        )

    @staticmethod
    def _nonnegative_scalar(value: Any, name: str) -> float:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != ():
            raise ValueError(f"{name} must be a finite scalar")
        scalar = float(values)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return scalar

    @classmethod
    def _unit_scalar(cls, value: Any, name: str) -> float:
        scalar = cls._nonnegative_scalar(value, name)
        if scalar > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return scalar
