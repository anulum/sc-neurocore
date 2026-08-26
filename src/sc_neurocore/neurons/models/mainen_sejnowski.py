# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mainen & Sejnowski 1996 — axonal Na spike initiation model

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _safe_exp(x: float) -> float:
    return float(np.exp(np.clip(x, -500.0, 500.0)))


def _linoid(x: float, k: float) -> float:
    """Evaluate ``x / (1 - exp(-x / k))`` with its analytic limit ``k`` at 0.

    The ``expm1`` form keeps full precision near the removable singularity
    and saturates (rate → 0) for strongly negative ``x`` instead of
    overflowing.
    """

    if x == 0.0:
        return k
    return x / float(-np.expm1(-x / k))


@dataclass
class MainenSejnowskiNeuron:
    """Mainen & Sejnowski 1996 — axonal Na spike initiation model.

    2-compartment: soma (passive) + axon (active Na + K).
    Axon initiates spike via fast Na kinetics; soma follows passively.
    C_s dV_s/dt = -g_L(V_s - E_L) + gc(V_a - V_s) + I
    C_a dV_a/dt = -I_Na - I_K + gc(V_s - V_a)

    Reference: Mainen, Z.F. & Sejnowski, T.J. (1996). Nature 382:363–366.
    The Euler substepping and the in-loop voltage clips to [-200, 200] mV
    are repository-specific specialisations, not publication-exact
    claims. Canonical rate evaluation uses numerically stable analytic
    removable-singularity limits (``expm1`` linoid form); the historical
    additive 1e-12 denominator regularisation — which returned a zero
    rate exactly at each singular voltage — remains reconstructible via
    ``legacy_epsilon_rates=True`` as a count-neutral legacy
    configuration.
    """

    vs: float = -65.0
    va: float = -65.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.3
    kappa: float = 10.0
    g_na: float = 3000.0
    g_k: float = 1500.0
    g_l: float = 1.0
    e_na: float = 50.0
    e_k: float = -90.0
    e_l: float = -70.0
    c_s: float = 1.0
    c_a: float = 0.1
    dt: float = 0.005
    v_threshold: float = -20.0
    legacy_epsilon_rates: bool = False

    def __post_init__(self) -> None:
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        values = (
            self.vs,
            self.va,
            self.m,
            self.h,
            self.n,
            self.kappa,
            self.g_na,
            self.g_k,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_l,
            self.c_s,
            self.c_a,
            self.dt,
            self.v_threshold,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Mainen-Sejnowski state and parameters must be finite")
        if not (-200.0 <= self.vs <= 200.0 and -200.0 <= self.va <= 200.0):
            raise ValueError("vs and va must be within [-200, 200] mV")
        if not all(0.0 <= gate <= 1.0 for gate in (self.m, self.h, self.n)):
            raise ValueError("m, h, and n must be within [0, 1]")
        if not 0.0 <= self.kappa <= 100.0:
            raise ValueError("kappa is outside the public coupling bounds")
        if not (0.0 <= self.g_na <= 5000.0 and 0.0 <= self.g_k <= 3000.0):
            raise ValueError("g_na and g_k exceed the public conductance bounds")
        if not 0.0 <= self.g_l <= 5.0:
            raise ValueError("g_l exceeds the public conductance bounds")
        if not (30.0 <= self.e_na <= 70.0 and -100.0 <= self.e_k <= -70.0):
            raise ValueError("e_na or e_k is outside the public reversal bounds")
        if not -90.0 <= self.e_l <= -50.0:
            raise ValueError("e_l is outside the public reversal bounds")
        if not (0.5 <= self.c_s <= 2.0 and 0.05 <= self.c_a <= 1.0):
            raise ValueError("c_s or c_a is outside the public capacitance bounds")
        if not 0.0 < self.dt <= 0.1:
            raise ValueError("dt is outside the public bounds")
        if not -40.0 <= self.v_threshold <= 20.0:
            raise ValueError("v_threshold is outside the public bounds")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_configuration()

        vs_prev = self.vs
        vs_candidate = self.vs
        va_candidate = self.va
        m_candidate = self.m
        h_candidate = self.h
        n_candidate = self.n

        for _ in range(20):
            va = va_candidate
            # Axon HH gates (shifted for fast initiation)
            if self.legacy_epsilon_rates:
                am = 0.182 * (va + 25.0) / (1.0 - _safe_exp(-(va + 25.0) / 9.0) + 1e-12)
                bm = -0.124 * (va + 25.0) / (1.0 - _safe_exp((va + 25.0) / 9.0) + 1e-12)
                ah = 0.024 * (va + 40.0) / (1.0 - _safe_exp(-(va + 40.0) / 5.0) + 1e-12)
                bh = -0.0091 * (va + 65.0) / (1.0 - _safe_exp((va + 65.0) / 5.0) + 1e-12)
                an = 0.02 * (va - 20.0) / (1.0 - _safe_exp(-(va - 20.0) / 9.0) + 1e-12)
                bn = -0.002 * (va - 20.0) / (1.0 - _safe_exp((va - 20.0) / 9.0) + 1e-12)
            else:
                am = 0.182 * _linoid(va + 25.0, 9.0)
                bm = 0.124 * _linoid(-(va + 25.0), 9.0)
                ah = 0.024 * _linoid(va + 40.0, 5.0)
                bh = 0.0091 * _linoid(-(va + 65.0), 5.0)
                an = 0.02 * _linoid(va - 20.0, 9.0)
                bn = 0.002 * _linoid(-(va - 20.0), 9.0)

            m_candidate = np.clip(
                m_candidate + (am * (1 - m_candidate) - bm * m_candidate) * self.dt, 0.0, 1.0
            )
            h_candidate = np.clip(
                h_candidate + (ah * (1 - h_candidate) - bh * h_candidate) * self.dt, 0.0, 1.0
            )
            n_candidate = np.clip(
                n_candidate + (an * (1 - n_candidate) - bn * n_candidate) * self.dt, 0.0, 1.0
            )

            i_na = self.g_na * m_candidate**3 * h_candidate * (va - self.e_na)
            i_k = self.g_k * n_candidate * (va - self.e_k)
            i_l = self.g_l * (vs_candidate - self.e_l)

            dvs = (-i_l + self.kappa * (va - vs_candidate) + current) / self.c_s * self.dt
            dva = (-i_na - i_k + self.kappa * (vs_candidate - va)) / self.c_a * self.dt
            vs_candidate = float(np.clip(vs_candidate + dvs, -200.0, 200.0))
            va_candidate = float(np.clip(va_candidate + dva, -200.0, 200.0))

            if not all(
                math.isfinite(value)
                for value in (
                    vs_candidate,
                    va_candidate,
                    float(m_candidate),
                    float(h_candidate),
                    float(n_candidate),
                )
            ):
                raise ValueError("Mainen-Sejnowski candidate state became non-finite")

        self.vs = vs_candidate
        self.va = va_candidate
        self.m = float(m_candidate)
        self.h = float(h_candidate)
        self.n = float(n_candidate)

        return 1 if (self.vs >= self.v_threshold and vs_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.vs = -65.0
        self.va = -65.0
        self.m, self.h, self.n = 0.05, 0.6, 0.3
