# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pinsky-Rinzel 1994 — 2-compartment pyramidal cell

from __future__ import annotations

import math
from dataclasses import dataclass

_STATE_NAMES = ("v_s", "v_d", "h", "n", "s", "c", "q")
_PARAM_NAMES = (
    "gc",
    "p",
    "g_na",
    "g_kdr",
    "g_ca",
    "g_kahp",
    "g_kc",
    "g_l",
    "e_na",
    "e_k",
    "e_ca",
    "e_l",
    "dt",
    "v_threshold",
)
_STRICTLY_POSITIVE_PARAMS = ("gc", "g_na", "g_kdr", "g_ca", "g_kahp", "g_kc", "g_l", "dt")
_GATE_NAMES = ("h", "n", "s", "q")


@dataclass
class PinskyRinzelNeuron:
    """Pinsky-Rinzel 1994 — 2-compartment pyramidal cell.

    Soma (fast Na/K) coupled to dendrite (Ca/KAHP) via gc.
    Minimal model for burst generation in cortical pyramidal cells.

    Reference: Pinsky, P.F. & Rinzel, J. (1994). J. Comput. Neurosci. 1:39–60.
    """

    v_s: float = -60.0
    v_d: float = -60.0
    h: float = 0.9
    n: float = 0.1
    s: float = 0.0
    c: float = 0.0
    q: float = 0.0
    gc: float = 2.1
    p: float = 0.5
    g_na: float = 30.0
    g_kdr: float = 15.0
    g_ca: float = 10.0
    g_kahp: float = 0.8
    g_kc: float = 15.0
    g_l: float = 0.1
    e_na: float = 60.0
    e_k: float = -75.0
    e_ca: float = 80.0
    e_l: float = -60.0
    dt: float = 0.02
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
        if self.c < 0.0:
            raise ValueError("c must be non-negative")
        for name in _GATE_NAMES:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} gate must remain in [0, 1]")

    @staticmethod
    def _exp(value: float) -> float:
        try:
            out = math.exp(value)
        except OverflowError as exc:
            raise FloatingPointError("Pinsky-Rinzel rate exponential overflowed") from exc
        if not math.isfinite(out):
            raise FloatingPointError("Pinsky-Rinzel rate exponential became non-finite")
        return out

    @staticmethod
    def _logistic(value: float) -> float:
        if value >= 0.0:
            exp_neg = PinskyRinzelNeuron._exp(-value)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = PinskyRinzelNeuron._exp(value)
        return exp_pos / (1.0 + exp_pos)

    @staticmethod
    def _validate_candidate(
        values: tuple[float, float, float, float, float, float, float],
    ) -> tuple[float, ...]:
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Pinsky-Rinzel candidate state became non-finite")
        v_s, v_d, h, n, s, c, q = values
        if c < 0.0:
            raise FloatingPointError("Pinsky-Rinzel calcium concentration became negative")
        for name, value in zip(_GATE_NAMES, (h, n, s, q)):
            if not 0.0 <= value <= 1.0:
                raise FloatingPointError(f"{name} gate left [0, 1]")
        return values

    def step(self, current_soma: float, current_dend: float = 0.0) -> int:
        current_soma = float(current_soma)
        current_dend = float(current_dend)
        if not math.isfinite(current_soma) or not math.isfinite(current_dend):
            raise ValueError("current_soma and current_dend must be finite")
        self._validate_configuration()

        v_prev = self.v_s
        am = (
            0.32 * (self.v_s + 54.0) / (1.0 - self._exp(-(self.v_s + 54.0) / 4.0))
            if abs(self.v_s + 54.0) > 1e-6
            else 8.0
        )
        bm = (
            0.28 * (self.v_s + 27.0) / (self._exp((self.v_s + 27.0) / 5.0) - 1.0)
            if abs(self.v_s + 27.0) > 1e-6
            else 5.6
        )
        m_inf = am / (am + bm)

        ah = 0.128 * self._exp(-(self.v_s + 50.0) / 18.0)
        bh = 4.0 * self._logistic((self.v_s + 27.0) / 5.0)
        an = (
            0.032 * (self.v_s + 52.0) / (1.0 - self._exp(-(self.v_s + 52.0) / 5.0))
            if abs(self.v_s + 52.0) > 1e-6
            else 0.32
        )
        bn = 0.5 * self._exp(-(self.v_s + 57.0) / 40.0)

        s_inf = self._logistic((self.v_d + 20.0) / 9.0)

        # Soma (PR 1994, Table 1)
        i_na = self.g_na * m_inf**2 * self.h * (self.v_s - self.e_na)
        i_kdr = self.g_kdr * self.n * (self.v_s - self.e_k)
        i_ls = self.g_l * (self.v_s - self.e_l)
        i_ds = (self.gc / self.p) * (self.v_s - self.v_d)

        # Dendrite (PR 1994, Table 1)
        i_ca = self.g_ca * self.s**2 * (self.v_d - self.e_ca)
        i_kahp = self.g_kahp * self.q * (self.v_d - self.e_k)
        chi = min(self.v_d / 250.0 + 0.5, 1.0) if self.v_d <= 50.0 else 2.0
        i_kc = self.g_kc * self.c * chi * (self.v_d - self.e_k)
        i_ld = self.g_l * (self.v_d - self.e_l)
        i_sd = (self.gc / (1 - self.p)) * (self.v_d - self.v_s)

        next_v_s = self.v_s + (-i_na - i_kdr - i_ls - i_ds + current_soma / self.p) * self.dt
        next_v_d = (
            self.v_d + (-i_ca - i_kahp - i_kc - i_ld - i_sd + current_dend / (1 - self.p)) * self.dt
        )
        next_h = self.h + (ah * (1 - self.h) - bh * self.h) * self.dt
        next_n = self.n + (an * (1 - self.n) - bn * self.n) * self.dt
        next_s = self.s + ((s_inf - self.s) / 5.0) * self.dt
        next_c = max(0.0, self.c + (-0.13 * i_ca - 0.075 * self.c) * self.dt)
        q_inf = min(next_c / (next_c + 2.0), 1.0)
        next_q = self.q + ((q_inf - self.q) / 100.0) * self.dt

        self.v_s, self.v_d, self.h, self.n, self.s, self.c, self.q = self._validate_candidate(
            (next_v_s, next_v_d, next_h, next_n, next_s, next_c, next_q)
        )

        return 1 if (self.v_s >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v_s, self.v_d = -60.0, -60.0
        self.h, self.n, self.s, self.c, self.q = 0.9, 0.1, 0.0, 0.0, 0.0
