# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wang 1999 NMDA-autapse pyramidal neuron

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class NMDANeuron:
    """Wang (1999) pyramidal LIF neuron with an NMDA autapse.

    The scalar model combines Wang's equations (1), (2), (4), and (5):
    a pyramidal leaky integrate-and-fire membrane, optional calcium-activated
    potassium adaptation, and the two-stage saturating NMDA gate. The recurrent
    gate receives the neuron's own emitted events, matching the NMDA-only
    autapse experiment in Figure 3. ``current`` is the paper's applied current
    ``I_app`` in nA.

    The source used second-order Runge--Kutta integration at 0.02--0.05 ms and
    interpolated spike times. This deterministic scalar specialization uses
    midpoint RK2 at 0.05 ms and sampled upward threshold detection.

    Reference: Wang, X.-J. (1999), J Neurosci 19(21):9587--9603;
    Jahr & Stevens (1990), J Neurosci 10(9):3178--3182.
    """

    v: float = -70.0
    x_nmda: float = 0.0
    s_nmda: float = 0.0
    ca: float = 0.0
    refractory_remaining: float = 0.0
    c_m: float = 0.5
    g_l: float = 0.025
    v_l: float = -70.0
    g_nmda: float = 0.1
    e_nmda: float = 0.0
    mg_conc: float = 1.0
    alpha_x: float = 1.0
    tau_x: float = 2.0
    alpha_s: float = 1.0
    tau_s: float = 80.0
    kinetic_scale: float = 1.0
    g_ahp: float = 0.0
    v_k: float = -85.0
    alpha_ca: float = 0.2
    tau_ca: float = 80.0
    dt: float = 0.05
    v_threshold: float = -52.0
    v_reset: float = -59.0
    refractory_period: float = 2.0

    def __post_init__(self) -> None:
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        values = (
            self.v,
            self.x_nmda,
            self.s_nmda,
            self.ca,
            self.refractory_remaining,
            self.c_m,
            self.g_l,
            self.v_l,
            self.g_nmda,
            self.e_nmda,
            self.mg_conc,
            self.alpha_x,
            self.tau_x,
            self.alpha_s,
            self.tau_s,
            self.kinetic_scale,
            self.g_ahp,
            self.v_k,
            self.alpha_ca,
            self.tau_ca,
            self.dt,
            self.v_threshold,
            self.v_reset,
            self.refractory_period,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("NMDA state and parameters must be finite")
        if not -120.0 <= self.v <= 80.0:
            raise ValueError("v must be within [-120, 80] mV")
        if self.x_nmda < 0.0 or not 0.0 <= self.s_nmda <= 1.0 or self.ca < 0.0:
            raise ValueError("x_nmda and ca must be non-negative and s_nmda within [0, 1]")
        if not 0.0 <= self.refractory_remaining <= self.refractory_period:
            raise ValueError("refractory_remaining must be within [0, refractory_period]")
        if not (0.01 <= self.c_m <= 10.0 and 0.0 <= self.g_l <= 1.0):
            raise ValueError("c_m or g_l is outside the public bounds")
        if not (-100.0 <= self.v_l <= -40.0 and -10.0 <= self.e_nmda <= 10.0):
            raise ValueError("v_l or e_nmda is outside the public bounds")
        if not (0.0 <= self.g_nmda <= 2.0 and 0.0 <= self.mg_conc <= 5.0):
            raise ValueError("g_nmda or mg_conc is outside the public bounds")
        if not (0.0 <= self.alpha_x <= 10.0 and 0.01 <= self.tau_x <= 100.0):
            raise ValueError("alpha_x or tau_x is outside the public bounds")
        if not (0.0 <= self.alpha_s <= 10.0 and 1.0 <= self.tau_s <= 1000.0):
            raise ValueError("alpha_s or tau_s is outside the public bounds")
        if not 0.01 <= self.kinetic_scale <= 100.0:
            raise ValueError("kinetic_scale is outside the public bounds")
        if not (0.0 <= self.g_ahp <= 10.0 and -120.0 <= self.v_k <= -40.0):
            raise ValueError("g_ahp or v_k is outside the public bounds")
        if not (0.0 <= self.alpha_ca <= 10.0 and 1.0 <= self.tau_ca <= 1000.0):
            raise ValueError("alpha_ca or tau_ca is outside the public bounds")
        if not 0.0 < self.dt <= 0.05:
            raise ValueError("dt must be within (0, 0.05] ms")
        if not -80.0 <= self.v_threshold <= -30.0:
            raise ValueError("v_threshold is outside the public bounds")
        if not -100.0 <= self.v_reset < self.v_threshold:
            raise ValueError("v_reset must be below v_threshold and within [-100, -30) mV")
        if not 0.0 <= self.refractory_period <= 20.0:
            raise ValueError("refractory_period is outside the public bounds")

    def _derivatives(
        self, v: float, x_nmda: float, s_nmda: float, ca: float, current: float
    ) -> tuple[float, float, float, float]:
        mg_block = 1.0 / (1.0 + self.mg_conc * math.exp(-0.062 * v) / 3.57)
        i_l = self.g_l * (v - self.v_l)
        i_ahp = self.g_ahp * ca * (v - self.v_k)
        i_nmda = self.g_nmda * s_nmda * mg_block * (v - self.e_nmda)
        dv = (-i_l - i_ahp - i_nmda + current) / self.c_m
        dx = self.kinetic_scale * (-x_nmda / self.tau_x)
        ds = self.kinetic_scale * (self.alpha_s * x_nmda * (1.0 - s_nmda) - s_nmda / self.tau_s)
        dca = -ca / self.tau_ca
        return dv, dx, ds, dca

    def _rk2_candidate(
        self, v: float, x_nmda: float, s_nmda: float, ca: float, current: float
    ) -> tuple[float, float, float, float]:
        k1 = self._derivatives(v, x_nmda, s_nmda, ca, current)
        half_dt = 0.5 * self.dt
        midpoint = (
            v + half_dt * k1[0],
            x_nmda + half_dt * k1[1],
            s_nmda + half_dt * k1[2],
            ca + half_dt * k1[3],
        )
        k2 = self._derivatives(*midpoint, current)
        return (
            v + self.dt * k2[0],
            x_nmda + self.dt * k2[1],
            s_nmda + self.dt * k2[2],
            ca + self.dt * k2[3],
        )

    def step(self, current: float = 0.0) -> int:
        """Advance one source-grid step and return the sampled spike event."""

        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_configuration()
        held = self.refractory_remaining > 0.0
        voltage = self.v_reset if held else self.v
        v_candidate, x_candidate, s_candidate, ca_candidate = self._rk2_candidate(
            voltage, self.x_nmda, self.s_nmda, self.ca, current
        )
        refractory_candidate = max(0.0, self.refractory_remaining - self.dt)
        fired = 0
        if held:
            v_candidate = self.v_reset
        elif v_candidate >= self.v_threshold:
            fired = 1
            v_candidate = self.v_reset
            refractory_candidate = self.refractory_period
            x_candidate += self.kinetic_scale * self.alpha_x
            ca_candidate += self.alpha_ca
        if not all(
            math.isfinite(value)
            for value in (v_candidate, x_candidate, s_candidate, ca_candidate, refractory_candidate)
        ):
            raise ValueError("NMDA candidate state became non-finite")
        self.v = max(-120.0, min(80.0, v_candidate))
        self.x_nmda = max(0.0, x_candidate)
        self.s_nmda = max(0.0, min(1.0, s_candidate))
        self.ca = max(0.0, ca_candidate)
        self.refractory_remaining = refractory_candidate
        return fired

    def reset(self) -> None:
        """Restore dynamic state while preserving the configured source profile."""

        self.v = self.v_l
        self.x_nmda = 0.0
        self.s_nmda = 0.0
        self.ca = 0.0
        self.refractory_remaining = 0.0
