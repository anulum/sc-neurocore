# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Renshaw Cell (Spinal Inhibitory Interneuron)

from __future__ import annotations

import math
from dataclasses import dataclass


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


def _all_finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _exact_gate(previous: float, alpha: float, beta: float, phi: float, dt: float) -> float | None:
    total = phi * (alpha + beta)
    if not _all_finite(previous, alpha, beta, total, dt) or total <= 0.0:
        return None
    steady = alpha / (alpha + beta)
    return _clamp01(steady + (previous - steady) * math.exp(-total * dt))


def _exact_relax(previous: float, steady: float, tau: float, dt: float) -> float | None:
    if not _all_finite(previous, steady, tau, dt) or tau <= 0.0:
        return None
    return _clamp01(steady + (previous - steady) * math.exp(-dt / tau))


def _probability(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


def _physiological_voltage(value: float) -> bool:
    return math.isfinite(value) and -150.0 <= value <= 100.0


@dataclass
class RenshawCell:
    """Renshaw cell — spinal inhibitory interneuron for recurrent inhibition.

    WB gating core with strong adaptation to produce burst-then-decay
    response to motor axon collateral input.

    Reference: Renshaw (1941); Windhorst (1996) Prog Neurobiol 46(5).
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    adapt: float = 0.0
    g_na: float = 35.0
    g_k: float = 9.0
    g_adapt: float = 5.0
    g_l: float = 0.12
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    tau_adapt: float = 50.0
    dt: float = 0.01
    v_threshold: float = -20.0

    def _valid_state(self) -> bool:
        return (
            _physiological_voltage(self.v)
            and _probability(self.h)
            and _probability(self.n)
            and _probability(self.adapt)
            and _all_finite(
                self.g_na,
                self.g_k,
                self.g_adapt,
                self.g_l,
                self.e_na,
                self.e_k,
                self.e_l,
                self.c_m,
                self.phi,
                self.tau_adapt,
                self.dt,
                self.v_threshold,
            )
            and self.g_na >= 0.0
            and self.g_k >= 0.0
            and self.g_adapt >= 0.0
            and self.g_l >= 0.0
            and self.c_m > 0.0
            and self.phi > 0.0
            and self.tau_adapt > 0.0
            and self.dt > 0.0
        )

    def step(self, current: float = 0.0) -> int:
        if not math.isfinite(current) or not self._valid_state():
            return 0

        v_prev = self.v
        v = self.v
        h = self.h
        n = self.n
        adapt = self.adapt
        n_sub = max(1, int(0.5 / max(self.dt, 0.001)))
        for _ in range(n_sub):
            am = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            bm = 4.0 * math.exp(-(v + 60.0) / 18.0)
            m_inf = am / (am + bm)
            ah = 0.07 * math.exp(-(v + 58.0) / 20.0)
            bh = 1.0 / (1.0 + math.exp(-(v + 28.0) / 10.0))
            an = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            bn = 0.125 * math.exp(-(v + 44.0) / 80.0)

            h_next = _exact_gate(h, ah, bh, self.phi, self.dt)
            n_next = _exact_gate(n, an, bn, self.phi, self.dt)
            if h_next is None or n_next is None:
                return 0

            adapt_inf = 1.0 / (1.0 + math.exp(-(v + 30.0) / 5.0))
            adapt_next = _exact_relax(adapt, adapt_inf, self.tau_adapt, self.dt)
            if adapt_next is None:
                return 0

            g_na = self.g_na * m_inf**3 * h_next
            g_k = self.g_k * n_next**4
            g_adapt = self.g_adapt * adapt_next
            g_total = g_na + g_k + g_adapt + self.g_l
            if not math.isfinite(g_total) or g_total <= 0.0:
                return 0

            steady_v = (
                current
                + g_na * self.e_na
                + g_k * self.e_k
                + g_adapt * self.e_k
                + self.g_l * self.e_l
            ) / g_total
            v_next = steady_v + (v - steady_v) * math.exp(-(g_total / self.c_m) * self.dt)
            if not (
                _physiological_voltage(v_next)
                and _probability(h_next)
                and _probability(n_next)
                and _probability(adapt_next)
            ):
                return 0

            v = v_next
            h = h_next
            n = n_next
            adapt = adapt_next

        self.v = v
        self.h = h
        self.n = n
        self.adapt = adapt

        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.8
        self.n = 0.1
        self.adapt = 0.0
