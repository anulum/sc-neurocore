# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A-type K⁺ (IA) Neuron

from __future__ import annotations

import math
from dataclasses import dataclass, field

_EXP_MAX = 709.0
_EXP_MIN = -745.0


def _checked_exp(x: float) -> float:
    if not math.isfinite(x) or x > _EXP_MAX:
        raise ValueError("A-type K neuron exponential argument is non-finite or unstable")
    if x < _EXP_MIN:
        return 0.0
    return math.exp(x)


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    rate = a * d / (1.0 - _checked_exp(-d / k))
    if not math.isfinite(rate):
        raise ValueError("A-type K neuron rate candidate is non-finite")
    return rate


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _nonnegative(name: str, value: float) -> None:
    _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _positive(name: str, value: float) -> None:
    _finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _probability(name: str, value: float) -> None:
    _finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must stay in [0, 1]")


@dataclass
class ATypeKNeuron:
    """A-type K+ neuron with Wang-Buzsaki core and transient IA current."""

    v: float = -65.0
    h: float = 0.6
    n: float = 0.32
    a: float = 0.1
    b: float = 0.8
    g_na: float = 35.0
    g_k: float = 9.0
    g_a: float = 8.0
    g_l: float = 0.1
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    dt: float = 0.5
    v_threshold: float = -20.0
    gain: float = 1.0
    _sub_steps: int = field(default=50, repr=False)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for name in ("v", "e_na", "e_k", "e_l", "v_threshold", "gain"):
            _finite(name, getattr(self, name))
        for name in ("h", "n", "a", "b"):
            _probability(name, getattr(self, name))
        for name in ("g_na", "g_k", "g_a", "g_l"):
            _nonnegative(name, getattr(self, name))
        for name in ("c_m", "phi", "dt"):
            _positive(name, getattr(self, name))
        if not isinstance(self._sub_steps, int) or self._sub_steps <= 0:
            raise ValueError("_sub_steps must be a positive integer")

    @staticmethod
    def _validate_candidates(v: float, h: float, n: float, a: float, b: float) -> None:
        _finite("membrane candidate", v)
        _probability("h candidate", h)
        _probability("n candidate", n)
        _probability("a candidate", a)
        _probability("b candidate", b)

    def step(self, current: float = 0.0) -> int:
        self._validate()
        _finite("current", current)
        inp = self.gain * current
        _finite("input drive", inp)
        sub_dt = self.dt / self._sub_steps
        _positive("sub_dt", sub_dt)

        v = self.v
        h = self.h
        n = self.n
        a = self.a
        b = self.b
        fired = 0

        for _ in range(self._sub_steps):
            alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            beta_m = 4.0 * _checked_exp(-(v + 60.0) / 18.0)
            m_inf = alpha_m / (alpha_m + beta_m)
            _finite("m_inf", m_inf)

            alpha_h = 0.07 * _checked_exp(-(v + 58.0) / 20.0)
            beta_h = 1.0 / (1.0 + _checked_exp(-(v + 28.0) / 10.0))
            alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            beta_n = 0.125 * _checked_exp(-(v + 44.0) / 80.0)
            a_inf = 1.0 / (1.0 + _checked_exp(-(v + 50.0) / 20.0))
            b_inf = 1.0 / (1.0 + _checked_exp((v + 70.0) / 6.0))

            h_next = h + sub_dt * self.phi * (alpha_h * (1.0 - h) - beta_h * h)
            n_next = n + sub_dt * self.phi * (alpha_n * (1.0 - n) - beta_n * n)
            a_next = a + sub_dt * (a_inf - a) / 2.0
            b_next = b + sub_dt * (b_inf - b) / 50.0

            i_na = self.g_na * m_inf**3 * h_next * (v - self.e_na)
            i_k = self.g_k * n_next**4 * (v - self.e_k)
            i_a = self.g_a * a_next**3 * b_next * (v - self.e_k)
            i_l = self.g_l * (v - self.e_l)
            dv = (-i_na - i_k - i_a - i_l + inp) / self.c_m
            v_next = v + sub_dt * dv

            if v_next >= self.v_threshold:
                fired = 1
                v_next = -65.0
            if v_next < -100.0 or v_next > 60.0:
                raise ValueError("membrane candidate left physiological safety bounds")
            self._validate_candidates(v_next, h_next, n_next, a_next, b_next)
            v, h, n, a, b = v_next, h_next, n_next, a_next, b_next

        self.v = v
        self.h = h
        self.n = n
        self.a = a
        self.b = b
        return fired

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.6
        self.n = 0.32
        self.a = 0.1
        self.b = 0.8
