# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Butera, Rinzel & Smith 1999 — pre-Botzinger respiratory

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar


@dataclass
class ButeraRespiratoryNeuron:
    """Butera, Rinzel & Smith 1999 pre-Botzinger respiratory neuron.

    The maintained numerical contract is candidate-first RK4 over the published
    three-state Model I ODE: membrane voltage ``v``, potassium activation ``n``,
    and persistent sodium inactivation ``h_nap``. Invalid drive, parameters,
    corrupted state, non-finite rates, or out-of-envelope candidates raise
    before mutating state.
    """

    _EXP_MAX: ClassVar[float] = 700.0

    v: float = -50.0
    n: float = 0.01
    h_nap: float = 0.5
    g_na: float = 28.0
    g_nap: float = 2.8
    g_k: float = 11.2
    g_l: float = 2.8
    e_na: float = 50.0
    e_k: float = -85.0
    e_l: float = -65.0
    e_syn: float = -10.0
    tau_h: float = 10000.0
    dt: float = 0.1
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        self.v = self._finite_float("v", self.v)
        self.n = self._gate_float("n", self.n)
        self.h_nap = self._gate_float("h_nap", self.h_nap)
        self.g_na = self._non_negative_float("g_na", self.g_na)
        self.g_nap = self._non_negative_float("g_nap", self.g_nap)
        self.g_k = self._non_negative_float("g_k", self.g_k)
        self.g_l = self._non_negative_float("g_l", self.g_l)
        self.e_na = self._finite_float("e_na", self.e_na)
        self.e_k = self._finite_float("e_k", self.e_k)
        self.e_l = self._finite_float("e_l", self.e_l)
        self.e_syn = self._finite_float("e_syn", self.e_syn)
        self.tau_h = self._positive_float("tau_h", self.tau_h)
        self.dt = self._positive_float("dt", self.dt)
        self.v_threshold = self._finite_float("v_threshold", self.v_threshold)
        if not -200.0 <= self.v <= 100.0:
            raise ValueError("v must be within the Butera finite voltage envelope [-200, 100]")

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a finite float")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _positive_float(cls, name: str, value: float) -> float:
        value = cls._finite_float(name, value)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    @classmethod
    def _non_negative_float(cls, name: str, value: float) -> float:
        value = cls._finite_float(name, value)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @classmethod
    def _gate_float(cls, name: str, value: float) -> float:
        value = cls._finite_float(name, value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
        return value

    @classmethod
    def _checked_exp(cls, x: float, name: str) -> float:
        if not math.isfinite(x) or x > cls._EXP_MAX:
            raise FloatingPointError(f"{name} exponent is outside finite domain")
        return math.exp(x)

    @staticmethod
    def _checked_cosh(x: float, name: str) -> float:
        if not math.isfinite(x) or abs(x) > 350.0:
            raise FloatingPointError(f"{name} cosh argument is outside finite domain")
        return math.cosh(x)

    @classmethod
    def _sexp(cls, x: float) -> float:
        return cls._checked_exp(float(x), "sexp")

    @classmethod
    def _scosh(cls, x: float) -> float:
        return cls._checked_cosh(float(x), "scosh")

    @classmethod
    def _rates(cls, v: float, tau_h_base: float) -> tuple[float, float, float, float, float, float]:
        m_na_inf = 1.0 / (1.0 + cls._checked_exp(-(v + 34.0) / 5.0, "m_na_inf"))
        m_nap_inf = 1.0 / (1.0 + cls._checked_exp(-(v + 40.0) / 6.0, "m_nap_inf"))
        h_nap_inf = 1.0 / (1.0 + cls._checked_exp((v + 48.0) / 6.0, "h_nap_inf"))
        n_inf = 1.0 / (1.0 + cls._checked_exp(-(v + 29.0) / 4.0, "n_inf"))
        tau_n = max(10.0 / max(cls._checked_cosh((v + 29.0) / 8.0, "tau_n"), 1e-12), 0.01)
        tau_h = max(tau_h_base / max(cls._checked_cosh((v + 48.0) / 12.0, "tau_h"), 1e-12), 0.1)
        rates = (m_na_inf, m_nap_inf, h_nap_inf, n_inf, tau_n, tau_h)
        if not all(math.isfinite(rate) for rate in rates) or tau_n <= 0.0 or tau_h <= 0.0:
            raise FloatingPointError("Butera rates must be finite with positive time constants")
        return rates

    def _validate_runtime_state(self) -> None:
        if not self._candidate_valid((self.v, self.n, self.h_nap)):
            raise FloatingPointError("Butera runtime state is outside finite physical bounds")
        self._non_negative_float("g_na", self.g_na)
        self._non_negative_float("g_nap", self.g_nap)
        self._non_negative_float("g_k", self.g_k)
        self._non_negative_float("g_l", self.g_l)
        self._positive_float("tau_h", self.tau_h)
        self._positive_float("dt", self.dt)
        self._finite_float("v_threshold", self.v_threshold)

    @staticmethod
    def _candidate_valid(state: tuple[float, float, float]) -> bool:
        v, n, h_nap = state
        return (
            all(math.isfinite(x) for x in state)
            and -200.0 <= v <= 100.0
            and -0.05 <= n <= 1.05
            and -0.05 <= h_nap <= 1.05
        )

    def _derivatives(
        self, state: tuple[float, float, float], current: float
    ) -> tuple[float, float, float]:
        v, n, h_nap = state
        if not all(math.isfinite(value) for value in (v, n, h_nap, current)):
            raise FloatingPointError("Butera derivative state and current must be finite")
        v = max(-200.0, min(100.0, v))
        n = max(0.0, min(1.0, n))
        h_nap = max(0.0, min(1.0, h_nap))
        m_na_inf, m_nap_inf, h_nap_inf, n_inf, tau_n, tau_h = self._rates(v, self.tau_h)
        i_na = self.g_na * m_na_inf**3 * (1.0 - n) * (v - self.e_na)
        i_nap = self.g_nap * m_nap_inf * h_nap * (v - self.e_na)
        i_k = self.g_k * n**4 * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)
        derivs = (
            -i_na - i_nap - i_k - i_l + current,
            (n_inf - n) / tau_n,
            (h_nap_inf - h_nap) / tau_h,
        )
        if not all(math.isfinite(value) for value in derivs):
            raise FloatingPointError("Butera derivatives must be finite")
        return derivs

    def _rk4_candidate(self, current: float) -> tuple[float, float, float]:
        current = self._finite_float("current", current)
        self._validate_runtime_state()
        state = (self.v, self.n, self.h_nap)
        dt = self.dt
        k1 = self._derivatives(state, current)
        k2 = self._derivatives(tuple(s + 0.5 * dt * k for s, k in zip(state, k1)), current)
        k3 = self._derivatives(tuple(s + 0.5 * dt * k for s, k in zip(state, k2)), current)
        k4 = self._derivatives(tuple(s + dt * k for s, k in zip(state, k3)), current)
        raw_candidate = tuple(
            s + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for s, a, b, c, d in zip(state, k1, k2, k3, k4)
        )
        if not all(math.isfinite(value) for value in raw_candidate):
            raise FloatingPointError("Butera RK4 candidate must be finite")
        v, n, h_nap = raw_candidate
        return (max(-200.0, min(100.0, v)), max(0.0, min(1.0, n)), max(0.0, min(1.0, h_nap)))

    def step(self, current: float) -> int:
        v_prev = self.v
        self.v, self.n, self.h_nap = self._rk4_candidate(current)
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        self.v, self.n, self.h_nap = -50.0, 0.01, 0.5
