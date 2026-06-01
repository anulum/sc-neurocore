# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sherman, Rinzel & Keizer 1988 beta-cell burster

from __future__ import annotations

from dataclasses import dataclass
import math

_TAU_N = 9.09
_EXP_LIMIT = 80.0
_V_MIN = -200.0
_V_MAX = 200.0


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _gate(value: float) -> bool:
    return _finite(value) and 0.0 <= value <= 1.0


def _sigmoid(arg: float) -> float:
    arg = max(-_EXP_LIMIT, min(_EXP_LIMIT, arg))
    return 1.0 / (1.0 + math.exp(-arg))


@dataclass
class ShermanRinzelKeizerNeuron:
    """Sherman, Rinzel & Keizer 1988 reduced pancreatic beta-cell burster."""

    v: float = -50.0
    n: float = 0.1
    s: float = 0.1
    g_ca: float = 3.6
    g_k: float = 10.0
    g_s: float = 4.0
    e_ca: float = 25.0
    e_k: float = -75.0
    tau_s: float = 5000.0
    dt: float = 0.5
    v_threshold: float = -20.0

    def _validate(self) -> None:
        finite_scalars = (
            self.v,
            self.g_ca,
            self.g_k,
            self.g_s,
            self.e_ca,
            self.e_k,
            self.tau_s,
            self.dt,
            self.v_threshold,
        )
        if not all(_finite(value) for value in finite_scalars):
            raise ValueError("Sherman-Rinzel-Keizer state and parameters must be finite")
        if not (_V_MIN <= self.v <= _V_MAX):
            raise ValueError("Sherman-Rinzel-Keizer voltage outside safety envelope")
        if not (_gate(self.n) and _gate(self.s)):
            raise ValueError("Sherman-Rinzel-Keizer gates must stay within [0, 1]")
        if self.g_ca <= 0.0 or self.g_k <= 0.0 or self.g_s < 0.0:
            raise ValueError("Sherman-Rinzel-Keizer conductances must be physical")
        if self.tau_s <= 0.0 or self.dt <= 0.0:
            raise ValueError("Sherman-Rinzel-Keizer time constants must be positive")

    def _derivatives(
        self, v: float, n_gate: float, s_gate: float, current: float
    ) -> tuple[float, float, float]:
        if not (_finite(v) and _finite(n_gate) and _finite(s_gate) and _finite(current)):
            raise ValueError("Sherman-Rinzel-Keizer derivative input is invalid")
        m_inf = _sigmoid((v + 20.0) / 12.0)
        n_inf = _sigmoid((v + 16.0) / 5.0)
        s_inf = _sigmoid((v + 35.0) / 10.0)
        i_ca = self.g_ca * m_inf * (v - self.e_ca)
        i_k = self.g_k * n_gate * (v - self.e_k)
        i_s = self.g_s * s_gate * (v - self.e_k)
        dv = -i_ca - i_k - i_s + current
        dn = (n_inf - n_gate) / _TAU_N
        ds = (s_inf - s_gate) / self.tau_s
        if not (_finite(dv) and _finite(dn) and _finite(ds)):
            raise ValueError("Sherman-Rinzel-Keizer derivative became non-finite")
        return dv, dn, ds

    def _rk4_candidate(self, current: float) -> tuple[float, float, float]:
        half_dt = 0.5 * self.dt
        k1 = self._derivatives(self.v, self.n, self.s, current)
        k2 = self._derivatives(
            self.v + half_dt * k1[0],
            self.n + half_dt * k1[1],
            self.s + half_dt * k1[2],
            current,
        )
        k3 = self._derivatives(
            self.v + half_dt * k2[0],
            self.n + half_dt * k2[1],
            self.s + half_dt * k2[2],
            current,
        )
        k4 = self._derivatives(
            self.v + self.dt * k3[0],
            self.n + self.dt * k3[1],
            self.s + self.dt * k3[2],
            current,
        )
        next_v = self.v + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        next_n = self.n + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        next_s = self.s + self.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        if not (_finite(next_v) and _gate(next_n) and _gate(next_s)):
            raise ValueError("Sherman-Rinzel-Keizer RK4 candidate is invalid")
        if not (_V_MIN <= next_v <= _V_MAX):
            raise ValueError("Sherman-Rinzel-Keizer RK4 voltage candidate escaped envelope")
        return next_v, next_n, next_s

    def step(self, current: float) -> int:
        """Advance one constant-current RK4 step and return threshold crossing."""

        if not _finite(current):
            raise ValueError("Sherman-Rinzel-Keizer current must be finite")
        self._validate()
        v_prev = self.v
        next_v, next_n, next_s = self._rk4_candidate(current)
        self.v = next_v
        self.n = next_n
        self.s = next_s
        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        self.v, self.n, self.s = -50.0, 0.1, 0.1
