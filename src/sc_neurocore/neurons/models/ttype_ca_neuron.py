# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — T-type Ca²⁺ (IT) Neuron

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


@dataclass
class TTypeCaNeuron:
    """T-type Ca²⁺ (IT) neuron — WB base + low-voltage-activated Ca²⁺ current.

    IT activates at subthreshold voltages (-65 to -50 mV) and inactivates
    slowly. When de-inactivated by hyperpolarisation, IT produces a
    low-threshold spike (LTS) that can trigger a burst of Na⁺ spikes.

    m_T∞ = 1 / (1 + exp(-(v + 52) / 5))
    s∞ = 1 / (1 + exp((v + 81) / 4))
    τ_s = 30 + 100 / (1 + exp((v + 75) / 10))

    Reference: Huguenard (1996) Annu Rev Physiol 58:329;
    Destexhe, Bal, McCormick & Sejnowski (1996) J Neurophysiol 76:2049.
    The WB spiking base follows Wang & Buzsáki (1996) J Neurosci 16:6402;
    the threshold-reset event and the spike-triggered inactivation
    collapse (s ×= 0.3) are repository-specific specialisations, not
    publication-exact claims.
    """

    v: float = -65.0
    h: float = 0.6
    n: float = 0.32
    s: float = 0.9
    g_na: float = 35.0
    g_k: float = 9.0
    g_t: float = 0.1
    g_l: float = 0.2
    e_na: float = 55.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    dt: float = 0.5
    v_threshold: float = -20.0
    gain: float = 1.0
    _sub_steps: int = field(default=50, repr=False)

    def __post_init__(self) -> None:
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        values = (
            self.v,
            self.h,
            self.n,
            self.s,
            self.g_na,
            self.g_k,
            self.g_t,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_ca,
            self.e_l,
            self.c_m,
            self.phi,
            self.dt,
            self.v_threshold,
            self.gain,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("T-type state and parameters must be finite")
        if not -100.0 <= self.v <= 60.0:
            raise ValueError("v must be within [-100, 60] mV")
        if not all(0.0 <= gate <= 1.0 for gate in (self.h, self.n, self.s)):
            raise ValueError("h, n, and s must be within [0, 1]")
        if not (0.0 <= self.g_na <= 200.0 and 0.0 <= self.g_k <= 100.0):
            raise ValueError("g_na and g_k exceed the public conductance bounds")
        if not (0.0 <= self.g_t <= 20.0 and 0.0 <= self.g_l <= 5.0):
            raise ValueError("g_t and g_l exceed the public conductance bounds")
        if not (30.0 <= self.e_na <= 70.0 and -100.0 <= self.e_k <= -70.0):
            raise ValueError("e_na or e_k is outside the public reversal bounds")
        if not (60.0 <= self.e_ca <= 150.0 and -80.0 <= self.e_l <= -40.0):
            raise ValueError("e_ca or e_l is outside the public reversal bounds")
        if not (0.5 <= self.c_m <= 2.0 and 0.5 <= self.phi <= 10.0):
            raise ValueError("c_m or phi is outside the public bounds")
        if not (0.0 < self.dt <= 1.0 and -20.0 <= self.v_threshold <= 20.0):
            raise ValueError("dt or v_threshold is outside the public bounds")
        if not 0.0 <= self.gain <= 10.0:
            raise ValueError("gain must be within [0, 10]")
        if not isinstance(self._sub_steps, int) or isinstance(self._sub_steps, bool):
            raise TypeError("_sub_steps must be an integer")
        if not 1 <= self._sub_steps <= 10_000:
            raise ValueError("_sub_steps must be within [1, 10000]")

    def step(self, current: float = 0.0) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_configuration()

        inp = self.gain * current
        sub_dt = self.dt / self._sub_steps
        fired = 0
        v_candidate = self.v
        h_candidate = self.h
        n_candidate = self.n
        s_candidate = self.s

        for _ in range(self._sub_steps):
            v = v_candidate

            alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            beta_m = 4.0 * math.exp(-(v + 60.0) / 18.0)
            m_inf = alpha_m / (alpha_m + beta_m)

            alpha_h = 0.07 * math.exp(-(v + 58.0) / 20.0)
            beta_h = 1.0 / (1.0 + math.exp(-(v + 28.0) / 10.0))

            alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            beta_n = 0.125 * math.exp(-(v + 44.0) / 80.0)

            m_t_inf = 1.0 / (1.0 + math.exp(-(v + 52.0) / 5.0))
            s_inf = 1.0 / (1.0 + math.exp((v + 81.0) / 4.0))
            tau_s = 30.0 + 100.0 / (1.0 + math.exp((v + 75.0) / 10.0))

            h_candidate += (
                sub_dt * self.phi * (alpha_h * (1.0 - h_candidate) - beta_h * h_candidate)
            )
            n_candidate += (
                sub_dt * self.phi * (alpha_n * (1.0 - n_candidate) - beta_n * n_candidate)
            )
            s_candidate += sub_dt * (s_inf - s_candidate) / tau_s

            i_na = self.g_na * m_inf**3 * h_candidate * (v - self.e_na)
            i_k = self.g_k * n_candidate**4 * (v - self.e_k)
            i_t = self.g_t * m_t_inf**2 * s_candidate * (v - self.e_ca)
            i_l = self.g_l * (v - self.e_l)

            dv = (-i_na - i_k - i_t - i_l + inp) / self.c_m
            v_candidate += sub_dt * dv

            if not all(
                math.isfinite(value)
                for value in (v_candidate, h_candidate, n_candidate, s_candidate)
            ):
                raise ValueError("T-type candidate state became non-finite")

            if v_candidate >= self.v_threshold:
                fired = 1
                v_candidate = -65.0
                s_candidate *= 0.3

        self.v = max(-100.0, min(60.0, v_candidate))
        self.h = max(0.0, min(1.0, h_candidate))
        self.n = max(0.0, min(1.0, n_candidate))
        self.s = max(0.0, min(1.0, s_candidate))

        return fired

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.6
        self.n = 0.32
        self.s = 0.9
