# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cerebellar Stellate Cell

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    z = -d / k
    if z > 60.0:
        return 0.0
    if z < -60.0:
        return a * d
    return a * d / (1.0 - math.exp(z))


def _boltz(v: float, vh: float, k: float) -> float:
    z = -(v - vh) / k
    if z > 60.0:
        return 0.0
    if z < -60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def _safe_exp(value: float) -> float:
    return math.exp(max(-60.0, min(60.0, value)))


def _exact_relax(value: float, target: float, tau: float, dt: float) -> float:
    return target + (value - target) * math.exp(-dt / tau)


def _exact_hh_gate(value: float, alpha: float, beta: float, phi: float, dt: float) -> float:
    rate = phi * (alpha + beta)
    target = alpha / (alpha + beta)
    return target + (value - target) * math.exp(-rate * dt)


def _exact_voltage_step(
    v: float,
    input_current: float,
    conductances: tuple[tuple[float, float], ...],
    c_m: float,
    dt: float,
) -> float:
    g_total = sum(g for g, _ in conductances)
    if g_total <= 0.0:
        return v + dt * input_current / c_m
    reversal_drive = sum(g * e_rev for g, e_rev in conductances)
    v_inf = (input_current + reversal_drive) / g_total
    return v_inf + (v - v_inf) * math.exp(-dt * g_total / c_m)


@dataclass
class StellateCell:
    """Cerebellar stellate cell — fast-spiking molecular layer interneuron.

    WB Na⁺/K⁺ core + Kv3.1 for narrow APs. Feedforward inhibition onto
    Purkinje cell dendrites. Smaller than basket cells.

    Reference: Sultan & Bower (1999) J Comp Neurol 409:63;
    Häusser & Clark (1997) Neuron 19:665.
    """

    v: float = -65.0
    h: float = 0.6
    n: float = 0.32
    p: float = 0.0
    g_na: float = 35.0
    g_k: float = 9.0
    g_kv3: float = 3.0
    g_l: float = 0.1
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 0.5
    phi: float = 5.0
    dt: float = 0.5
    v_threshold: float = -20.0
    gain: float = 1.0
    _sub_steps: int = field(default=50, repr=False)

    def __post_init__(self) -> None:
        self._validate_state()

    def step(self, current: float = 0.0) -> int:
        self._validate_state()
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        inp = self.gain * current
        sub_dt = self.dt / self._sub_steps
        fired = 0
        v = self.v
        h = self.h
        n = self.n
        p = self.p

        for _ in range(self._sub_steps):
            alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            beta_m = 4.0 * _safe_exp(-(v + 60.0) / 18.0)
            m_inf = alpha_m / (alpha_m + beta_m)
            alpha_h = 0.07 * _safe_exp(-(v + 58.0) / 20.0)
            beta_h = _boltz(v, -28.0, 10.0)
            alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            beta_n = 0.125 * _safe_exp(-(v + 44.0) / 80.0)

            p_inf = _boltz(v, -10.0, 10.0)
            tau_p = 1.0 + 4.0 / (1.0 + _safe_exp((v + 20.0) / 15.0))

            h = max(0.0, min(1.0, _exact_hh_gate(h, alpha_h, beta_h, self.phi, sub_dt)))
            n = max(0.0, min(1.0, _exact_hh_gate(n, alpha_n, beta_n, self.phi, sub_dt)))
            p = max(0.0, min(1.0, _exact_relax(p, p_inf, tau_p, sub_dt)))

            g_na_eff = self.g_na * m_inf**3 * h
            g_k_eff = self.g_k * n**4
            g_kv3_eff = self.g_kv3 * p**2
            v = max(
                -100.0,
                min(
                    60.0,
                    _exact_voltage_step(
                        v,
                        inp,
                        (
                            (g_na_eff, self.e_na),
                            (g_k_eff, self.e_k),
                            (g_kv3_eff, self.e_k),
                            (self.g_l, self.e_l),
                        ),
                        self.c_m,
                        sub_dt,
                    ),
                ),
            )
            if not all(math.isfinite(x) for x in (v, h, n, p)):
                raise ValueError("stellate cell integration produced non-finite state")
            if v >= self.v_threshold:
                fired = 1
                v = -65.0

        self.v = v
        self.h = h
        self.n = n
        self.p = p
        return fired

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.6
        self.n = 0.32
        self.p = 0.0

    def _validate_state(self) -> None:
        finite_values = (
            self.v,
            self.h,
            self.n,
            self.p,
            self.g_na,
            self.g_k,
            self.g_kv3,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_l,
            self.c_m,
            self.phi,
            self.dt,
            self.v_threshold,
            self.gain,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("stellate cell state and parameters must be finite")
        if not -100.0 <= self.v <= 60.0:
            raise ValueError("stellate cell v must stay in [-100, 60]")
        if not all(0.0 <= gate <= 1.0 for gate in (self.h, self.n, self.p)):
            raise ValueError("stellate cell gates must stay in [0, 1]")
        if not all(
            conductance >= 0.0 for conductance in (self.g_na, self.g_k, self.g_kv3, self.g_l)
        ):
            raise ValueError("stellate cell conductances must be non-negative")
        if self.c_m <= 0.0 or self.phi <= 0.0 or self.dt <= 0.0:
            raise ValueError("stellate cell capacitance, rate scale, and timestep must be positive")
        if not isinstance(self._sub_steps, int) or self._sub_steps <= 0:
            raise ValueError("stellate cell sub-step count must be a positive integer")
        if self.gain < 0.0:
            raise ValueError("stellate cell gain must be non-negative")
