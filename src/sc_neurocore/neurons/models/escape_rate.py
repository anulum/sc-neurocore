# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gerstner 2000 — stochastic threshold (escape noise model)

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sc_neurocore.utils.numerics import safe_exp


@dataclass
class EscapeRateNeuron:
    """Gerstner 2000 — stochastic threshold (escape noise model).

    Membrane dynamics use the exact constant-current RC flow before evaluating
    the finite-step escape hazard.

    Reference: Gerstner, W. (2000). Neural Comput. 12:43–89.
    """

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    rho_0: float = 0.001
    delta_u: float = 3.0
    resistance: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        self._validate_runtime_state()

    def _validate_runtime_state(self) -> None:
        for field in ("v", "v_rest", "v_reset", "v_threshold"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in ("tau_m", "rho_0", "delta_u", "resistance", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def _spike_probability(self, voltage: float) -> float:
        if not math.isfinite(voltage):
            raise ValueError("voltage candidate must be finite")
        rate = self.rho_0 * safe_exp((voltage - self.v_threshold) / self.delta_u)
        hazard = rate * self.dt
        if not math.isfinite(hazard) or hazard < 0.0:
            raise ValueError("escape hazard must be finite and non-negative")
        probability = -math.expm1(-hazard)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("spike probability must remain finite and bounded")
        return probability

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        voltage = self._exact_voltage_candidate(current)
        p_spike = self._spike_probability(voltage)
        if np.random.random() < p_spike:
            self.v = self.v_reset
            return 1
        self.v = voltage
        return 0

    def reset(self) -> None:
        self.v = self.v_rest

    def _exact_voltage_candidate(self, current: float) -> float:
        steady_state = self.v_rest + self.resistance * current
        decay = math.exp(-self.dt / self.tau_m)
        voltage = steady_state + (self.v - steady_state) * decay
        if (
            not math.isfinite(steady_state)
            or not math.isfinite(decay)
            or not math.isfinite(voltage)
        ):
            raise ValueError("voltage candidate must be finite")
        return voltage
