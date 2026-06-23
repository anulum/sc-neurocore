# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chay & Keizer 1983 — pancreatic beta-cell square-wave burster

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ChayKeizerNeuron:
    """Chay & Keizer 1983 pancreatic beta-cell minimal model (five-state burster).

    The original five-dimensional model: membrane potential ``v``, the
    Hodgkin-Huxley activation/inactivation gates ``m``/``h`` of the inward
    calcium current, the delayed-rectifier potassium activation ``n``, and the
    free cytosolic calcium concentration ``ca`` (the slow variable that packages
    spikes into bursts). Calcium enters through the voltage-gated calcium channel
    during the active phase, gradually activating a calcium-dependent potassium
    conductance until it terminates the burst; calcium then decays through the
    silent phase until the next burst begins. With the published parameters the
    model produces square-wave bursts with a period of order ten to twenty
    seconds and a cytosolic calcium oscillation of order one micromolar.

    The conductances follow the Hodgkin-Huxley convention written as
    ``g (E_rev - V)`` (inward positive). The gate rate functions are the
    Hodgkin-Huxley 1952 forms with the membrane potential shifted by ``v_prime``
    for the calcium gates and ``v_star`` for the potassium gate, scaled by the
    temperature factor ``phi``; calcium influx is the surface-to-volume scaled
    calcium current minus a first-order pump removal.

    Reference: Chay, T.R. & Keizer, J. (1983). Minimal model for membrane
    oscillations in the pancreatic beta-cell. Biophys. J. 42:181-190.
    DOI 10.1016/S0006-3495(83)84384-7. Parameters are the paper's Table I with
    the burst calcium-removal rate of Fig. 1b; cross-checked against the Wolfram
    Demonstrations reference implementation.
    """

    v: float = -54.774
    m: float = 0.029725
    h: float = 0.747865
    n: float = 0.061079
    ca: float = 0.8
    g_ca: float = 6.5
    g_k: float = 12.0
    g_kca: float = 0.09
    g_l: float = 0.04
    e_ca: float = 100.0
    e_k: float = -75.0
    e_l: float = -40.0
    c_m: float = 1.0
    v_prime: float = 50.0
    v_star: float = 30.0
    k_dis: float = 1.0
    radius_cm: float = 8.9e-4
    faraday: float = 96487.0
    f_ca: float = 0.004
    k_ca: float = 0.04
    temp_celsius: float = 20.0
    dt: float = 0.05
    # Spike peaks sit on the burst plateau near -25 mV with troughs near -39 mV
    # (the paper's ~12 mV spikes); detect upward crossings between the two.
    v_threshold: float = -30.0

    _MAX_SUBSTEP: float = 0.01
    _V_MIN: float = -200.0
    _V_MAX: float = 200.0
    _CA_MAX: float = 1000.0

    @staticmethod
    def _finite(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    @classmethod
    def _nonnegative(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @classmethod
    def _probability(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return value

    @classmethod
    def _checked_exp(cls, exponent: float) -> float:
        if exponent < -700.0:
            return 0.0
        if exponent > 700.0:
            return math.exp(700.0)
        return math.exp(exponent)

    def _alpha_m(self, v: float) -> float:
        d = (v + self.v_prime) - 25.0
        if abs(d) < 1e-7:
            return 1.0
        return -0.1 * d / (self._checked_exp(-d / 10.0) - 1.0)

    def _beta_m(self, v: float) -> float:
        return 4.0 * self._checked_exp(-(v + self.v_prime) / 18.0)

    def _alpha_h(self, v: float) -> float:
        return 0.07 * self._checked_exp(-(v + self.v_prime) / 20.0)

    def _beta_h(self, v: float) -> float:
        return 1.0 / (self._checked_exp(-((v + self.v_prime) - 30.0) / 10.0) + 1.0)

    def _alpha_n(self, v: float) -> float:
        d = (v + self.v_star) - 10.0
        if abs(d) < 1e-7:
            return 0.1
        return -0.01 * d / (self._checked_exp(-d / 10.0) - 1.0)

    def _beta_n(self, v: float) -> float:
        return 0.125 * self._checked_exp(-(v + self.v_star) / 80.0)

    def _validated_state(
        self,
    ) -> tuple[float, float, float, float, float, int, float, float, float]:
        v = self._finite(self.v, "v")
        if not self._V_MIN <= v <= self._V_MAX:
            raise ValueError("v outside Chay-Keizer safety envelope")
        m = self._probability(self.m, "m")
        h = self._probability(self.h, "h")
        n = self._probability(self.n, "n")
        ca = self._nonnegative(self.ca, "ca")
        if ca > self._CA_MAX:
            raise ValueError("ca outside Chay-Keizer safety envelope")

        self._nonnegative(self.g_ca, "g_ca")
        self._nonnegative(self.g_k, "g_k")
        self._nonnegative(self.g_kca, "g_kca")
        self._nonnegative(self.g_l, "g_l")
        self._finite(self.e_ca, "e_ca")
        self._finite(self.e_k, "e_k")
        self._finite(self.e_l, "e_l")
        self._positive(self.c_m, "c_m")
        self._finite(self.v_prime, "v_prime")
        self._finite(self.v_star, "v_star")
        self._positive(self.k_dis, "k_dis")
        self._positive(self.radius_cm, "radius_cm")
        self._positive(self.faraday, "faraday")
        self._nonnegative(self.f_ca, "f_ca")
        self._nonnegative(self.k_ca, "k_ca")
        self._finite(self.temp_celsius, "temp_celsius")
        dt = self._positive(self.dt, "dt")
        self._finite(self.v_threshold, "v_threshold")

        # Temperature factor (Q10 = 3, Hodgkin-Huxley reference 6.3 degrees C) and
        # the surface-to-volume calcium influx coefficient, both from the paper.
        phi = 3.0 ** ((self.temp_celsius - 6.3) / 10.0)
        ca_influx = 3.0 / (self.radius_cm * self.faraday)

        substeps = max(1, math.ceil(dt / self._MAX_SUBSTEP))
        if substeps > 100000:
            raise ValueError("dt requires too many Chay-Keizer integration substeps")
        return v, m, h, n, ca, substeps, dt / substeps, phi, ca_influx

    def _candidate(
        self,
        v: float,
        m: float,
        h: float,
        n: float,
        ca: float,
        current: float,
        step_dt: float,
        phi: float,
        ca_influx: float,
    ) -> tuple[float, float, float, float, float]:
        g_ca_open = self.g_ca * m * m * m * h
        i_ca = g_ca_open * (self.e_ca - v)
        i_k = self.g_k * n * n * n * n * (self.e_k - v)
        i_kca = self.g_kca * (ca / (ca + self.k_dis)) * (self.e_k - v)
        i_l = self.g_l * (self.e_l - v)

        v_next = v + (current + 2.0 * i_ca + i_k + i_kca + i_l) / self.c_m * step_dt
        m_next = m + phi * (self._alpha_m(v) * (1.0 - m) - self._beta_m(v) * m) * step_dt
        h_next = h + phi * (self._alpha_h(v) * (1.0 - h) - self._beta_h(v) * h) * step_dt
        n_next = n + phi * (self._alpha_n(v) * (1.0 - n) - self._beta_n(v) * n) * step_dt
        ca_next = ca + self.f_ca * (ca_influx * i_ca - self.k_ca * ca) * step_dt

        if not math.isfinite(v_next) or not self._V_MIN <= v_next <= self._V_MAX:
            raise ValueError("Chay-Keizer voltage candidate outside safety envelope")
        m_next = min(max(m_next, 0.0), 1.0)
        h_next = min(max(h_next, 0.0), 1.0)
        n_next = min(max(n_next, 0.0), 1.0)
        if not math.isfinite(ca_next) or not 0.0 <= ca_next <= self._CA_MAX:
            raise ValueError("Chay-Keizer calcium candidate outside safety envelope")
        return v_next, m_next, h_next, n_next, ca_next

    def step(self, current: float = 0.0) -> int:
        """Advance one timestep and return an upward-threshold spike flag.

        The default zero current is the autonomous glucose-stimulated regime in
        which the cell bursts on its own; a non-zero ``current`` is an applied
        membrane current (for example the negative pump-mimicking current of the
        paper's Na/K-pump extension).
        """

        current = self._finite(current, "current")
        v_initial = self.v
        v, m, h, n, ca, substeps, step_dt, phi, ca_influx = self._validated_state()

        crossed = False
        for _ in range(substeps):
            v_next, m, h, n, ca = self._candidate(v, m, h, n, ca, current, step_dt, phi, ca_influx)
            crossed = crossed or (v_next >= self.v_threshold and v < self.v_threshold)
            v = v_next

        self.v, self.m, self.h, self.n, self.ca = v, m, h, n, ca
        return 1 if crossed and v_initial < self.v_threshold else 0

    def reset(self) -> None:
        self.v, self.m, self.h, self.n, self.ca = -54.774, 0.029725, 0.747865, 0.061079, 0.8
