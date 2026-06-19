# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Booth & Rinzel 1995 — bistable motoneuron, 2-compartment

from __future__ import annotations

import math
from dataclasses import dataclass

_STATE_NAMES = ("vs", "vd", "h", "n", "q", "ca")
_PARAM_NAMES = (
    "p",
    "gc",
    "g_na",
    "g_k",
    "g_ca",
    "g_kca",
    "g_l",
    "e_na",
    "e_k",
    "e_ca",
    "e_l",
    "c_m",
    "alpha_ca",
    "k_ca",
    "f_ca",
    "dt",
    "v_threshold",
)
_STRICTLY_POSITIVE_PARAMS = (
    "gc",
    "g_na",
    "g_k",
    "g_ca",
    "g_kca",
    "g_l",
    "c_m",
    "alpha_ca",
    "k_ca",
    "f_ca",
    "dt",
)
_GATE_NAMES = ("h", "n", "q")


@dataclass
class BoothRinzelNeuron:
    """Booth & Rinzel 1995 — bistable motoneuron, 2-compartment.

    C dVs/dt = -I_Na(Vs) - I_K(Vs) - I_L(Vs) - gc*(Vs - Vd)/p + I/p
    C dVd/dt = -I_Ca(Vd) - I_KCa(Vd) - I_L(Vd) - gc*(Vd - Vs)/(1-p)
    dq/dt   = (q_inf(Vd) - q) / tau_q
    dCa/dt  = -f * (alpha_Ca * I_Ca + k_Ca * Ca)

    Reference: Booth, V. & Rinzel, J. (1995). J. Neurophysiol. 73:1934–1945.
    """

    vs: float = -65.0
    vd: float = -65.0
    h: float = 0.9
    n: float = 0.0
    q: float = 0.0
    ca: float = 0.0
    p: float = 0.5
    gc: float = 0.1
    g_na: float = 120.0
    g_k: float = 20.0
    g_ca: float = 14.0
    g_kca: float = 5.0
    g_l: float = 0.51
    e_na: float = 55.0
    e_k: float = -80.0
    e_ca: float = 80.0
    e_l: float = -60.0
    c_m: float = 1.0
    alpha_ca: float = 0.009
    k_ca: float = 0.18
    f_ca: float = 0.0025
    dt: float = 0.025
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        self._validate_configuration(coerce=True)

    def _validate_configuration(self, *, coerce: bool = False) -> None:
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if coerce:
                setattr(self, name, value)
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.p < 1.0:
            raise ValueError("p must be in (0, 1)")
        if self.ca < 0.0:
            raise ValueError("ca must be non-negative")
        for name in _GATE_NAMES:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} gate must remain in [0, 1]")

    @staticmethod
    def _safe_exp(x: float) -> float:
        return math.exp(max(-500.0, min(500.0, x)))

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _validate_candidate(
        values: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Booth-Rinzel candidate state became non-finite")
        vs, vd, h, n, q, ca = values
        if ca < 0.0:
            raise FloatingPointError("Booth-Rinzel calcium concentration became negative")
        for name, value in zip(_GATE_NAMES, (h, n, q)):
            if not 0.0 <= value <= 1.0:
                raise FloatingPointError(f"{name} gate left [0, 1]")
        if not (-200.0 <= vs <= 100.0 and -200.0 <= vd <= 100.0):
            raise FloatingPointError("Booth-Rinzel voltage left safety envelope")
        return values

    def _substep(
        self, vs: float, vd: float, h: float, n: float, q: float, ca: float, current: float
    ) -> tuple[float, float, float, float, float, float]:
        m_inf = 1.0 / (1.0 + self._safe_exp(-(vs + 35.0) / 7.8))
        h_inf = 1.0 / (1.0 + self._safe_exp((vs + 55.0) / 7.0))
        tau_h = 30.0 / (
            self._safe_exp((vs + 50.0) / 15.0) + self._safe_exp(-(vs + 50.0) / 16.0) + 1e-12
        )
        n_inf = 1.0 / (1.0 + self._safe_exp(-(vs + 28.0) / 15.0))
        tau_n = 7.0 / (
            self._safe_exp((vs + 40.0) / 40.0) + self._safe_exp(-(vs + 40.0) / 50.0) + 1e-12
        )

        next_h = self._clip(h + (h_inf - h) / tau_h * self.dt, 0.0, 1.0)
        next_n = self._clip(n + (n_inf - n) / tau_n * self.dt, 0.0, 1.0)

        i_na = self.g_na * m_inf**3 * next_h * (vs - self.e_na)
        i_k = self.g_k * next_n**4 * (vs - self.e_k)
        i_ls = self.g_l * (vs - self.e_l)
        i_coup_s = self.gc * (vs - vd) / self.p
        dvs = (-i_na - i_k - i_ls - i_coup_s + current / self.p) / self.c_m * self.dt

        s_inf = 1.0 / (1.0 + self._safe_exp(-(vd + 22.0) / 5.0))
        q_inf = 1.0 / (1.0 + self._safe_exp(-(vd + 35.0) / 2.0))
        next_q = self._clip(q + (q_inf - q) / 400.0 * self.dt, 0.0, 1.0)

        i_ca = self.g_ca * s_inf**2 * (vd - self.e_ca)
        chi = min(ca / 250.0, 1.0)
        i_kca = self.g_kca * chi * (vd - self.e_k)
        i_ld = self.g_l * (vd - self.e_l)
        i_coup_d = self.gc * (vd - vs) / (1.0 - self.p)
        dvd = (-i_ca - i_kca - i_ld - i_coup_d) / self.c_m * self.dt
        next_ca = max(0.0, ca + self.f_ca * (-self.alpha_ca * i_ca - self.k_ca * ca) * self.dt)

        return self._validate_candidate(
            (
                self._clip(vs + dvs, -200.0, 100.0),
                self._clip(vd + dvd, -200.0, 100.0),
                next_h,
                next_n,
                next_q,
                next_ca,
            )
        )

    def step(self, current: float) -> int:
        current = float(current)
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_configuration()

        vs_prev = self.vs
        candidate: tuple[float, float, float, float, float, float] = (
            self.vs,
            self.vd,
            self.h,
            self.n,
            self.q,
            self.ca,
        )
        for _ in range(4):
            candidate = self._substep(*candidate, current)
        self.vs, self.vd, self.h, self.n, self.q, self.ca = candidate
        return 1 if (self.vs >= self.v_threshold and vs_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.vs = -65.0
        self.vd = -65.0
        self.h, self.n, self.q = 0.9, 0.0, 0.0
        self.ca = 0.0
