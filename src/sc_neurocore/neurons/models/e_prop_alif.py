# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bellec et al. 2020 — ALIF with eligibility traces for e-prop

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class EPropALIFNeuron:
    """Bellec et al. 2020 — ALIF with eligibility traces for e-prop.

    Adaptive LIF: threshold increases after each spike and decays.
    Eligibility trace e_t tracks how synaptic weight changes affect
    future spiking, enabling three-factor learning.

    Reference: Bellec, G. et al. (2020). Nat. Commun. 11:3625.
    """

    v: float = 0.0
    a: float = 0.0  # adaptive threshold component
    e_trace: float = 0.0  # eligibility trace
    tau_m: float = 20.0  # ms
    tau_a: float = 200.0  # ms, adaptation time constant
    v_threshold_base: float = 1.0
    beta: float = 0.07  # threshold adaptation coupling
    v_reset: float = 0.0
    dt: float = 1.0
    alpha_m: float = field(init=False)
    alpha_a: float = field(init=False)

    def __post_init__(self) -> None:
        self.alpha_m = np.exp(-self.dt / self.tau_m)
        self.alpha_a = np.exp(-self.dt / self.tau_a)

    def step(self, current: float) -> int:
        self.v = self.alpha_m * self.v + current
        threshold = self.v_threshold_base + self.beta * self.a
        # Bellec 2020 Eq. 4: pseudo-derivative for eligibility
        psi = max(0.0, 1.0 - abs(self.v - threshold)) * 0.3
        self.e_trace = self.alpha_a * self.e_trace + psi
        if self.v >= threshold:
            self.v = self.v_reset
            self.a = self.alpha_a * self.a + 1.0
            return 1
        self.a *= self.alpha_a
        return 0

    def reset(self) -> None:
        self.v, self.a, self.e_trace = 0.0, 0.0, 0.0
