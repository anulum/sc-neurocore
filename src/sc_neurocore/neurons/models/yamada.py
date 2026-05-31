# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Yamada, Kashimori & Kambara 1989 — subcritical Hopf burster

from __future__ import annotations

from dataclasses import dataclass
import math


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _tau_n(v: float) -> float:
    x = (v + 40.0) / 12.0
    if not math.isfinite(x):
        return math.nan
    if x > 709.0:
        return 1.0
    return 1.0 + 7.5 / (1.0 + math.exp(x))


@dataclass
class YamadaNeuron:
    """Yamada, Kashimori & Kambara 1989 — subcritical Hopf burster.

    3 ODEs: V, n (fast K recovery), q (slow variable for bursting).
    Exhibits square-wave bursting via slow modulation of a Hopf bifurcation.

    Reference: Yamada, W.M. et al. (1989). In: Methods in Neuronal Modeling. MIT Press, pp. 97–133.
    """

    v: float = -60.0
    n: float = 0.1
    q: float = 0.0
    g_na: float = 20.0
    g_k: float = 10.0
    g_q: float = 5.0
    g_l: float = 0.5
    e_na: float = 60.0
    e_k: float = -80.0
    e_q: float = -80.0
    e_l: float = -60.0
    tau_q: float = 300.0
    dt: float = 0.05
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        for name in (
            "v",
            "n",
            "q",
            "g_na",
            "g_k",
            "g_q",
            "g_l",
            "e_na",
            "e_k",
            "e_q",
            "e_l",
            "tau_q",
            "dt",
            "v_threshold",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("g_na", "g_k", "g_q", "g_l"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("tau_q", "dt"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("n", "q"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self.__post_init__()

        v_prev = self.v
        next_v, next_n, next_q = self._rk4_candidate(current)

        self.v = next_v
        self.n = next_n
        self.q = next_q
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def _derivatives(self, v: float, n: float, q: float, current: float) -> tuple[float, float, float]:
        if not all(math.isfinite(value) for value in (v, n, q, current)):
            raise ValueError("Yamada RK4 candidate must remain finite")
        if not 0.0 <= n <= 1.0 or not 0.0 <= q <= 1.0:
            raise ValueError("Yamada RK4 candidate must keep gates in [0, 1]")

        m_inf = _sigmoid((v + 30.0) / 9.5)
        n_inf = _sigmoid((v + 30.0) / 10.0)
        q_inf = _sigmoid((v + 50.0) / 10.0)
        tau_n = _tau_n(v)

        i_na = self.g_na * m_inf**3 * (1.0 - n) * (v - self.e_na)
        i_k = self.g_k * n**4 * (v - self.e_k)
        i_q = self.g_q * q * (v - self.e_q)
        i_l = self.g_l * (v - self.e_l)

        dv = -i_na - i_k - i_q - i_l + current
        dn = (n_inf - n) / tau_n
        dq = (q_inf - q) / self.tau_q
        if not all(
            math.isfinite(value)
            for value in (m_inf, n_inf, q_inf, tau_n, i_na, i_k, i_q, i_l, dv, dn, dq)
        ):
            raise ValueError("Yamada RK4 candidate must remain finite")
        return dv, dn, dq

    def _rk4_candidate(self, current: float) -> tuple[float, float, float]:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self.__post_init__()

        v0, n0, q0 = self.v, self.n, self.q
        dt = self.dt
        k1 = self._derivatives(v0, n0, q0, current)
        k2 = self._derivatives(
            v0 + 0.5 * dt * k1[0],
            n0 + 0.5 * dt * k1[1],
            q0 + 0.5 * dt * k1[2],
            current,
        )
        k3 = self._derivatives(
            v0 + 0.5 * dt * k2[0],
            n0 + 0.5 * dt * k2[1],
            q0 + 0.5 * dt * k2[2],
            current,
        )
        k4 = self._derivatives(v0 + dt * k3[0], n0 + dt * k3[1], q0 + dt * k3[2], current)
        next_v = v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        next_n = n0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        next_q = q0 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        if not all(math.isfinite(value) for value in (next_v, next_n, next_q)):
            raise ValueError("Yamada RK4 candidate must remain finite")
        if not 0.0 <= next_n <= 1.0 or not 0.0 <= next_q <= 1.0:
            raise ValueError("Yamada RK4 candidate must keep gates in [0, 1]")
        return next_v, next_n, next_q

    def reset(self) -> None:
        self.v, self.n, self.q = -60.0, 0.1, 0.0
