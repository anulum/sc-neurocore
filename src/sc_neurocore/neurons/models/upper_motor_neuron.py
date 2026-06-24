# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Upper Motor Neuron (Corticospinal L5 Pyramidal)

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class UpperMotorNeuron:
    """Upper motor neuron — layer 5 pyramidal cell, corticospinal projection.

    Pospischil 2008 RS parameterisation with Na⁺, K⁺, M-current (Kv7
    for adaptation) + high-threshold Ca²⁺ for dendritic Ca²⁺ spikes.

    Reference: Pospischil et al. (2008) Biol Cybern 99:427–441;
    Larkum (2013) Trends Neurosci 36(3).
    """

    v: float = -70.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.3
    p: float = 0.0
    s: float = 0.0
    g_na: float = 50.0
    g_k: float = 5.0
    g_m: float = 0.07
    g_ca: float = 0.3
    g_l: float = 0.1
    e_na: float = 50.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_l: float = -70.0
    c_m: float = 1.0
    dt: float = 0.025
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        self._validate_configuration()
        self._validate_state()

    @staticmethod
    def _rate_exp(value: float) -> float:
        return math.exp(max(-60.0, min(60.0, value)))

    @staticmethod
    def _gate(previous: float, alpha: float, beta: float, dt: float) -> float:
        total = alpha + beta
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("gate rates must define a positive finite time constant")
        steady = alpha / total
        return min(
            1.0, max(0.0, steady + (previous - steady) * UpperMotorNeuron._rate_exp(-total * dt))
        )

    @staticmethod
    def _gate_inf(previous: float, steady: float, tau: float, dt: float) -> float:
        if tau <= 0.0 or not math.isfinite(tau):
            raise ValueError("gate time constant must be positive and finite")
        return min(
            1.0, max(0.0, steady + (previous - steady) * UpperMotorNeuron._rate_exp(-dt / tau))
        )

    def _validate_configuration(self) -> None:
        for name in (
            "g_na",
            "g_k",
            "g_m",
            "g_ca",
            "g_l",
            "e_na",
            "e_k",
            "e_ca",
            "e_l",
            "c_m",
            "dt",
            "v_threshold",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("g_na", "g_k", "g_m", "g_ca", "g_l"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.c_m <= 0.0:
            raise ValueError("c_m must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

    def _validate_state(self) -> None:
        for name in ("v", "m", "h", "n", "p", "s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} state must be finite")
        if not -150.0 <= self.v <= 100.0:
            raise ValueError("v state must remain within [-150, 100] mV")
        for name in ("m", "h", "n", "p", "s"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} gate state must be within [0, 1]")

    def _step_candidate(
        self, v: float, m: float, h: float, n: float, p: float, s: float, current: float
    ) -> tuple[float, float, float, float, float, float]:
        vt = -56.2
        dv = v - vt
        x_m = dv - 13.0
        alpha_m = (
            0.32 * 4.0 if abs(x_m) < 1e-6 else -0.32 * x_m / (self._rate_exp(-x_m / 4.0) - 1.0)
        )
        # Published Pospischil beta_m numerator is V - V_T - 40; an earlier revision
        # shared the -17 offset of alpha_h, which drove the cell into depolarisation
        # block (three spikes then a fixed point near threshold for any stimulus).
        x_bm = dv - 40.0
        beta_m = (
            0.28 * 5.0 if abs(x_bm) < 1e-6 else 0.28 * x_bm / (self._rate_exp(x_bm / 5.0) - 1.0)
        )
        alpha_h = 0.128 * self._rate_exp(-(dv - 17.0) / 18.0)
        beta_h = 4.0 / (1.0 + self._rate_exp(-(dv - 40.0) / 5.0))
        x_n = dv - 15.0
        alpha_n = (
            0.032 * 5.0 if abs(x_n) < 1e-6 else -0.032 * x_n / (self._rate_exp(-x_n / 5.0) - 1.0)
        )
        beta_n = 0.5 * self._rate_exp(-(dv - 10.0) / 40.0)

        m = self._gate(m, alpha_m, beta_m, self.dt)
        h = self._gate(h, alpha_h, beta_h, self.dt)
        n = self._gate(n, alpha_n, beta_n, self.dt)
        p_inf = 1.0 / (1.0 + self._rate_exp(-(v + 35.0) / 10.0))
        tau_p = 400.0 / (
            3.3 * self._rate_exp((v + 35.0) / 20.0) + self._rate_exp(-(v + 35.0) / 20.0)
        )
        p = self._gate_inf(p, p_inf, tau_p, self.dt)
        s_inf = 1.0 / (1.0 + self._rate_exp(-(v + 20.0) / 5.0))
        s = self._gate_inf(s, s_inf, 10.0, self.dt)

        g_na = self.g_na * m**3 * h
        g_k = self.g_k * n**4
        g_m = self.g_m * p
        g_ca = self.g_ca * s**2
        g_total = g_na + g_k + g_m + g_ca + self.g_l
        if g_total <= 0.0 or not math.isfinite(g_total):
            raise ValueError("total conductance must be positive and finite")
        steady_v = (
            current
            + g_na * self.e_na
            + g_k * self.e_k
            + g_m * self.e_k
            + g_ca * self.e_ca
            + self.g_l * self.e_l
        ) / g_total
        v = steady_v + (v - steady_v) * self._rate_exp(-(g_total / self.c_m) * self.dt)
        return max(-150.0, min(100.0, v)), m, h, n, p, s

    def step(self, current: float = 0.0) -> int:
        self._validate_configuration()
        self._validate_state()
        if not math.isfinite(current):
            raise ValueError("current must be finite")

        v_prev = self.v
        v, m, h, n, p, s = self.v, self.m, self.h, self.n, self.p, self.s
        for _ in range(4):
            v, m, h, n, p, s = self._step_candidate(v, m, h, n, p, s, current)

        for name, value in zip(("v", "m", "h", "n", "p", "s"), (v, m, h, n, p, s), strict=True):
            if not math.isfinite(value):
                raise ValueError(f"{name} candidate state must be finite")

        self.v, self.m, self.h, self.n, self.p, self.s = v, m, h, n, p, s
        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        self._validate_configuration()
        self.v = -70.0
        self.m = 0.05
        self.h = 0.6
        self.n = 0.3
        self.p = 0.0
        self.s = 0.0
