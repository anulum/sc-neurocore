# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cerebellar Granule Cell

from __future__ import annotations

import math
from dataclasses import dataclass


def _exact_relax(value: float, target: float, tau: float, dt: float) -> float:
    return target + (value - target) * math.exp(-dt / tau)


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
class GranuleCell:
    """Cerebellar granule cell — D'Angelo et al. 2001 model.

    Smallest and most numerous neuron in the brain. 7 ionic currents:
    INa (m³h), IKdr (n⁴), IKA (a³b), ICaT (mT²s), IKCa (Hill), Ih (r),
    plus tonic GABA. Ca²⁺ dynamics with KCa half-saturation.

    Reference: D'Angelo et al. (2001) J Neurosci 21:759–770.
    """

    v: float = -70.0
    m: float = 0.02
    h: float = 0.85
    n: float = 0.05
    a: float = 0.1
    b: float = 0.8
    m_t: float = 0.01
    s: float = 0.95
    ca: float = 0.05
    r: float = 0.1
    c_m: float = 1.0
    g_na: float = 17.0
    g_kdr: float = 9.0
    g_ka: float = 1.0
    g_t: float = 0.5
    g_kca: float = 3.5
    g_h: float = 0.03
    g_l: float = 0.1
    g_tonic: float = 0.2
    e_na: float = 87.4
    e_k: float = -84.7
    e_ca: float = 129.3
    e_h: float = -40.0
    e_l: float = -58.0
    e_gaba: float = -75.0
    tau_ca: float = 10.0
    kd_kca: float = 0.2
    dt: float = 0.5
    sub_steps: int = 4
    gain: float = 1.0

    def __post_init__(self) -> None:
        self._validate_state()

    @staticmethod
    def _boltz(v: float, vh: float, k: float) -> float:
        z = -(v - vh) / k
        if z > 60.0:
            return 0.0
        if z < -60.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(z))

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def step(self, current: float = 0.0) -> int:
        self._validate_state()
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        inp = self.gain * current
        dt_sub = self.dt / float(self.sub_steps)
        v_prev = self.v
        v = self.v
        m = self.m
        h = self.h
        n = self.n
        a = self.a
        b = self.b
        m_t = self.m_t
        s = self.s
        ca = self.ca
        r = self.r

        for _ in range(self.sub_steps):
            bz = self._boltz

            m_inf = bz(v, -30.0, 7.0)
            tau_m = 0.1 + 0.3 / max(0.01, 1.0 + ((v + 30.0) / 10.0) ** 2)
            m = self._clamp01(_exact_relax(m, m_inf, tau_m, dt_sub))

            h_inf = bz(v, -52.0, -6.0)
            tau_h = 0.5 + 5.0 / max(0.01, 1.0 + ((v + 50.0) / 15.0) ** 2)
            h = self._clamp01(_exact_relax(h, h_inf, tau_h, dt_sub))

            n_inf = bz(v, -35.0, 8.0)
            tau_n = 1.0 + 5.0 / max(0.01, 1.0 + ((v + 35.0) / 15.0) ** 2)
            n = self._clamp01(_exact_relax(n, n_inf, tau_n, dt_sub))

            a_inf = bz(v, -50.0, 20.0)
            a = self._clamp01(_exact_relax(a, a_inf, 2.0, dt_sub))

            b_inf = bz(v, -70.0, -6.0)
            b = self._clamp01(_exact_relax(b, b_inf, 50.0, dt_sub))

            mt_inf = bz(v, -52.0, 5.0)
            m_t = self._clamp01(_exact_relax(m_t, mt_inf, 1.0, dt_sub))

            s_inf = bz(v, -60.0, -6.5)
            tau_s = 20.0 + 50.0 / max(0.01, 1.0 + ((v + 65.0) / 10.0) ** 2)
            s = self._clamp01(_exact_relax(s, s_inf, tau_s, dt_sub))

            r_inf = bz(v, -80.0, -10.0)
            tau_r = 50.0 + 200.0 / max(0.01, 1.0 + ((v + 80.0) / 20.0) ** 2)
            r = self._clamp01(_exact_relax(r, r_inf, tau_r, dt_sub))

            i_ca_t = self.g_t * m_t**2 * s * (v - self.e_ca)
            ca_entry = -i_ca_t * 0.001 if i_ca_t < 0.0 else 0.0
            ca = max(0.0, _exact_relax(ca, ca_entry * self.tau_ca, self.tau_ca, dt_sub))

            kca_inf = ca**2 / (ca**2 + self.kd_kca**2)

            g_na_eff = self.g_na * m**3 * h
            g_kdr_eff = self.g_kdr * n**4
            g_ka_eff = self.g_ka * a**3 * b
            g_t_eff = self.g_t * m_t**2 * s
            g_kca_eff = self.g_kca * kca_inf
            g_h_eff = self.g_h * r
            v = max(
                -100.0,
                min(
                    60.0,
                    _exact_voltage_step(
                        v,
                        inp,
                        (
                            (g_na_eff, self.e_na),
                            (g_kdr_eff, self.e_k),
                            (g_ka_eff, self.e_k),
                            (g_t_eff, self.e_ca),
                            (g_kca_eff, self.e_k),
                            (g_h_eff, self.e_h),
                            (self.g_l, self.e_l),
                            (self.g_tonic, self.e_gaba),
                        ),
                        self.c_m,
                        dt_sub,
                    ),
                ),
            )

            if not all(math.isfinite(x) for x in (v, m, h, n, a, b, m_t, s, ca, r)):
                raise ValueError("granule cell integration produced non-finite state")

        self.v = v
        self.m = m
        self.h = h
        self.n = n
        self.a = a
        self.b = b
        self.m_t = m_t
        self.s = s
        self.ca = ca
        self.r = r

        return 1 if v >= 0.0 and v_prev < 0.0 else 0

    def reset(self) -> None:
        self.v = -70.0
        self.m = 0.02
        self.h = 0.85
        self.n = 0.05
        self.a = 0.1
        self.b = 0.8
        self.m_t = 0.01
        self.s = 0.95
        self.ca = 0.05
        self.r = 0.1

    def _validate_state(self) -> None:
        finite_values = (
            self.v,
            self.m,
            self.h,
            self.n,
            self.a,
            self.b,
            self.m_t,
            self.s,
            self.ca,
            self.r,
            self.c_m,
            self.g_na,
            self.g_kdr,
            self.g_ka,
            self.g_t,
            self.g_kca,
            self.g_h,
            self.g_l,
            self.g_tonic,
            self.e_na,
            self.e_k,
            self.e_ca,
            self.e_h,
            self.e_l,
            self.e_gaba,
            self.tau_ca,
            self.kd_kca,
            self.dt,
            self.gain,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("granule cell state and parameters must be finite")
        if not -100.0 <= self.v <= 60.0:
            raise ValueError("granule cell v must stay in [-100, 60]")

        gates = (self.m, self.h, self.n, self.a, self.b, self.m_t, self.s, self.r)
        if not all(0.0 <= gate <= 1.0 for gate in gates):
            raise ValueError("granule cell gates must stay in [0, 1]")
        if self.ca < 0.0:
            raise ValueError("granule cell calcium concentration must be non-negative")
        if not all(
            conductance >= 0.0
            for conductance in (
                self.g_na,
                self.g_kdr,
                self.g_ka,
                self.g_t,
                self.g_kca,
                self.g_h,
                self.g_l,
                self.g_tonic,
            )
        ):
            raise ValueError("granule cell conductances must be non-negative")
        if self.c_m <= 0.0 or self.tau_ca <= 0.0 or self.kd_kca <= 0.0 or self.dt <= 0.0:
            raise ValueError(
                "granule cell capacitance, calcium, and timestep parameters must be positive"
            )
        if not isinstance(self.sub_steps, int) or self.sub_steps <= 0:
            raise ValueError("granule cell sub_steps must be a positive integer")
        if self.gain < 0.0:
            raise ValueError("granule cell gain must be non-negative")
