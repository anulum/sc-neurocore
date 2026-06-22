# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pagkalos, Chavlis & Poirazi 2023 — Dendrify active-dendrite model

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DendrifyNeuron:
    """Pagkalos, Chavlis & Poirazi 2023 — Dendrify active-dendrite 2-compartment model.

    Soma: standard LIF with reset.
    Dendrite: has a dendritic spike mechanism (NMDA-like) that produces
    a supralinear calcium plateau when input exceeds d_threshold.

    Reference: Pagkalos, M. et al. (2023). Nat. Commun. 14:3234.
    """

    v_s: float = -65.0
    v_d: float = -65.0
    d_active: bool = False  # dendritic plateau state
    tau_s: float = 10.0
    tau_d: float = 20.0
    g_c: float = 0.8  # soma-dendrite coupling
    d_threshold: float = -35.0  # mV, dendritic spike threshold
    d_amplitude: float = 30.0  # mV, plateau amplitude
    d_duration: float = 10.0  # ms
    d_timer: float = 0.0
    v_rest: float = -65.0
    v_threshold: float = -50.0
    v_reset: float = -65.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        v_s_prev = self.v_s
        # Dendritic compartment
        dv_d = (-(self.v_d - self.v_rest) + current - self.g_c * (self.v_d - self.v_s)) / self.tau_d
        self.v_d += dv_d * self.dt
        # Dendritic spike initiation
        if not self.d_active and self.v_d >= self.d_threshold:
            self.d_active = True
            self.d_timer = self.d_duration
        if self.d_active:
            self.d_timer -= self.dt
            d_inject = self.d_amplitude
            if self.d_timer <= 0.0:
                self.d_active = False
        else:
            d_inject = 0.0
        # Soma
        dv_s = (
            -(self.v_s - self.v_rest) + self.g_c * (self.v_d - self.v_s) + d_inject
        ) / self.tau_s
        self.v_s += dv_s * self.dt
        if self.v_s >= self.v_threshold and v_s_prev < self.v_threshold:
            self.v_s = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        self.v_s, self.v_d = -65.0, -65.0
        self.d_active, self.d_timer = False, 0.0
