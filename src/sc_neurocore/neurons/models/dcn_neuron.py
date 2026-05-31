# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Deep Cerebellar Nuclei (DCN) Neuron

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


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
class DCNNeuron:
    """Deep cerebellar nuclei neuron — main output of the cerebellum.

    WB Na⁺/K⁺ core + T-type Ca²⁺ (rebound bursting), Ih (pacemaker),
    persistent Na⁺ (subthreshold), Ca²⁺-dependent AHP. 7 currents total.

    Reference: Llinás & Mühlethaler (1988) J Physiol 404:241;
    Jahnsen (1986) J Physiol 372:129.
    """

    v: float = -60.0
    h: float = 0.6
    n: float = 0.32
    p: float = 0.01
    s: float = 0.8
    r: float = 0.1
    ca: float = 0.05
    g_na: float = 35.0
    g_nap: float = 0.5
    g_k: float = 9.0
    g_t: float = 0.1
    g_ahp: float = 2.0
    g_h: float = 0.02
    g_l: float = 0.2
    e_na: float = 55.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_h: float = -40.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    tau_ca: float = 150.0
    kd_ahp: float = 0.5
    dt: float = 0.5
    v_threshold: float = -20.0
    gain: float = 1.0
    _sub_steps: int = field(default=20, repr=False)

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
        s = self.s
        r = self.r
        ca = self.ca

        for _ in range(self._sub_steps):
            alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            beta_m = 4.0 * math.exp(-(v + 60.0) / 18.0)
            m_inf = alpha_m / (alpha_m + beta_m)
            alpha_h = 0.07 * math.exp(-(v + 58.0) / 20.0)
            beta_h = 1.0 / (1.0 + math.exp(-(v + 28.0) / 10.0))
            alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            beta_n = 0.125 * math.exp(-(v + 44.0) / 80.0)

            p_inf = 1.0 / (1.0 + math.exp(-(v + 48.0) / 5.0))
            tau_p = 5.0 + 15.0 / max(0.01, 1.0 + ((v + 48.0) / 10.0) ** 2)

            m_t_inf = 1.0 / (1.0 + math.exp(-(v + 52.0) / 5.0))
            s_inf = 1.0 / (1.0 + math.exp((v + 60.0) / 6.5))
            tau_s = 20.0 + 50.0 / (1.0 + math.exp((v + 65.0) / 10.0))

            r_inf = 1.0 / (1.0 + math.exp((v + 80.0) / 10.0))
            tau_r = 100.0 + 200.0 / (1.0 + math.exp((v + 70.0) / 10.0))

            h = _exact_hh_gate(h, alpha_h, beta_h, self.phi, sub_dt)
            n = _exact_hh_gate(n, alpha_n, beta_n, self.phi, sub_dt)
            p = _exact_relax(p, p_inf, tau_p, sub_dt)
            s = _exact_relax(s, s_inf, tau_s, sub_dt)
            r = _exact_relax(r, r_inf, tau_r, sub_dt)

            i_t = self.g_t * m_t_inf**2 * s * (v - self.e_ca)
            ca_entry = -i_t * 0.001 if i_t < 0.0 else 0.0
            ca = _exact_relax(ca, ca_entry * self.tau_ca, self.tau_ca, sub_dt)
            ca = max(0.0, ca)

            ahp_inf = ca**2 / (ca**2 + self.kd_ahp**2)

            g_na_eff = self.g_na * m_inf**3 * h
            g_nap_eff = self.g_nap * p
            g_k_eff = self.g_k * n**4
            g_t_eff = self.g_t * m_t_inf**2 * s
            g_ahp_eff = self.g_ahp * ahp_inf
            g_h_eff = self.g_h * r
            v = _exact_voltage_step(
                v,
                inp,
                (
                    (g_na_eff, self.e_na),
                    (g_nap_eff, self.e_na),
                    (g_k_eff, self.e_k),
                    (g_t_eff, self.e_ca),
                    (g_ahp_eff, self.e_k),
                    (g_h_eff, self.e_h),
                    (self.g_l, self.e_l),
                ),
                self.c_m,
                sub_dt,
            )

            if v >= self.v_threshold:
                fired = 1
                v = -60.0
                s *= 0.5
                ca += 0.5

        candidates = (v, h, n, p, s, r, ca)
        if not all(math.isfinite(value) for value in candidates):
            raise ValueError("DCN candidate state must be finite")

        self.v = max(-100.0, min(60.0, v))
        self.h = max(0.0, min(1.0, h))
        self.n = max(0.0, min(1.0, n))
        self.p = max(0.0, min(1.0, p))
        self.s = max(0.0, min(1.0, s))
        self.r = max(0.0, min(1.0, r))
        self.ca = max(0.0, ca)
        return fired

    def reset(self) -> None:
        self.v = -60.0
        self.h = 0.6
        self.n = 0.32
        self.p = 0.01
        self.s = 0.8
        self.r = 0.1
        self.ca = 0.05
        self._validate_state()

    def _validate_state(self) -> None:
        values = (
            self.v,
            self.h,
            self.n,
            self.p,
            self.s,
            self.r,
            self.ca,
            self.g_na,
            self.g_nap,
            self.g_k,
            self.g_t,
            self.g_ahp,
            self.g_h,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_ca,
            self.e_h,
            self.e_l,
            self.c_m,
            self.phi,
            self.tau_ca,
            self.kd_ahp,
            self.dt,
            self.v_threshold,
            self.gain,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("DCN state and parameters must be finite")
        if not -100.0 <= self.v <= 60.0:
            raise ValueError("v must be in the physical clamp interval [-100, 60]")
        for name in ("h", "n", "p", "s", "r"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} gate must be in [0, 1]")
        if self.ca < 0.0:
            raise ValueError("ca must be non-negative")
        if any(
            value < 0.0
            for value in (
                self.g_na,
                self.g_nap,
                self.g_k,
                self.g_t,
                self.g_ahp,
                self.g_h,
                self.g_l,
            )
        ):
            raise ValueError("conductances must be non-negative")
        if self.c_m <= 0.0:
            raise ValueError("c_m must be positive")
        if self.phi <= 0.0:
            raise ValueError("phi must be positive")
        if self.tau_ca <= 0.0:
            raise ValueError("tau_ca must be positive")
        if self.kd_ahp <= 0.0:
            raise ValueError("kd_ahp must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.gain < 0.0:
            raise ValueError("gain must be non-negative")
        if self._sub_steps <= 0:
            raise ValueError("_sub_steps must be positive")
