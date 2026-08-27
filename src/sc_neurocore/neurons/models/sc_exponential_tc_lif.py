# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC exponential two-compartment LIF (preserved engine recurrence)

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SCExponentialTwoCompartmentLIFNeuron:
    """SC exponential two-compartment LIF — preserved engine recurrence.

    Historical production-engine model formerly published under the
    ``TwoCompartmentLIFNeuron`` name. It is structurally distinct from
    both the Zhang et al. (2024) TC-LIF and the SC leaky variant:
    per-step exponential decay factors, additive wholesale coupling of
    the freshly decayed dendrite into the soma, per-compartment external
    currents, and a HARD soma reset:

    V_d[t] = exp(-dt/tau_d) * V_d[t-1] + I_dend[t]
    V_s[t] = exp(-dt/tau_s) * V_s[t-1] + I_soma[t] + kappa * V_d[t]
    Spike when V_s >= theta; V_s -> V_reset, V_d unchanged.

    Count-neutral SC identity: it consumes no source-catalogue slot and
    makes no publication-exact claim. The production Rust engine keeps
    this recurrence verbatim as ``SCExponentialTwoCompartmentLIF``,
    anchored to the pre-2026-08-27 built engine trajectories.
    """

    v_s: float = 0.0
    v_d: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    theta: float = 1.0
    tau_s: float = 2.0
    tau_d: float = 20.0
    kappa: float = 0.5
    dt: float = 1.0

    def __post_init__(self) -> None:
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        values = (
            self.v_s,
            self.v_d,
            self.v_rest,
            self.v_reset,
            self.theta,
            self.tau_s,
            self.tau_d,
            self.kappa,
            self.dt,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SC exponential TC-LIF state and parameters must be finite")
        if not (-1e6 <= self.v_s <= 1e6 and -1e6 <= self.v_d <= 1e6):
            raise ValueError("v_s and v_d must be within [-1e6, 1e6]")
        if not (-100.0 <= self.v_rest <= 100.0 and -100.0 <= self.v_reset <= 100.0):
            raise ValueError("v_rest and v_reset must be within [-100, 100]")
        if not 0.0 < self.theta <= 100.0:
            raise ValueError("theta must be within (0, 100]")
        if not (0.1 <= self.tau_s <= 1000.0 and 0.1 <= self.tau_d <= 1000.0):
            raise ValueError("tau_s and tau_d must be within [0.1, 1000]")
        if not 0.0 <= self.kappa <= 10.0:
            raise ValueError("kappa must be within [0, 10]")
        if not 0.0 < self.dt <= 10.0:
            raise ValueError("dt must be within (0, 10]")

    def step(self, i_soma: float, i_dend: float = 0.0) -> int:
        if not (math.isfinite(i_soma) and math.isfinite(i_dend)):
            raise ValueError("i_soma and i_dend must be finite")
        self._validate_configuration()

        alpha_s = math.exp(-self.dt / self.tau_s)
        alpha_d = math.exp(-self.dt / self.tau_d)
        v_d_candidate = alpha_d * self.v_d + i_dend
        v_s_candidate = alpha_s * self.v_s + i_soma + self.kappa * v_d_candidate
        if not (math.isfinite(v_d_candidate) and math.isfinite(v_s_candidate)):
            raise ValueError("SC exponential TC-LIF candidate state became non-finite")

        self.v_d = v_d_candidate
        if v_s_candidate >= self.theta:
            self.v_s = self.v_reset
            return 1
        self.v_s = v_s_candidate
        return 0

    def reset(self) -> None:
        self.v_s = self.v_rest
        self.v_d = self.v_rest
