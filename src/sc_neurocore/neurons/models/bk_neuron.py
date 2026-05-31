# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BK (Big Conductance Ca²⁺-Activated K⁺) Neuron

from __future__ import annotations

import math
from dataclasses import dataclass, field

_EXP_MAX = 709.0
_EXP_MIN = -745.0


def _checked_exp(x: float) -> float:
    if not math.isfinite(x) or x > _EXP_MAX:
        raise ValueError("BK neuron exponential argument is non-finite or unstable")
    if x < _EXP_MIN:
        return 0.0
    return math.exp(x)


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    """Rate function with safe handling of (v + vhalf) near zero."""
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    rate = a * d / (1.0 - _checked_exp(-d / k))
    if not math.isfinite(rate):
        raise ValueError("BK neuron rate candidate is non-finite")
    return rate


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _positive(name: str, value: float) -> None:
    _finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _nonnegative(name: str, value: float) -> None:
    _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _probability(name: str, value: float) -> None:
    _finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must stay in [0, 1]")


@dataclass
class BKNeuron:
    """BK calcium-activated K+ channel neuron.

    Runtime calcium, gate, and membrane candidates are computed locally and
    committed only after all finite/probability/bounds checks pass. This keeps
    the documented explicit substep path while preventing non-finite calcium or
    voltage state from being silently reset after poisoning downstream currents.
    """

    v: float = -65.0
    h: float = 0.6
    n: float = 0.32
    ca: float = 0.0
    g_na: float = 35.0
    g_k: float = 9.0
    g_bk: float = 3.0
    g_l: float = 0.1
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    tau_ca: float = 50.0
    dt: float = 0.5
    v_threshold: float = -20.0
    gain: float = 1.0
    _sub_steps: int = field(default=50, repr=False)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for name in ("v", "e_na", "e_k", "e_l", "v_threshold", "gain"):
            _finite(name, getattr(self, name))
        for name in ("h", "n"):
            _probability(name, getattr(self, name))
        _nonnegative("ca", self.ca)
        for name in ("g_na", "g_k", "g_bk", "g_l"):
            _nonnegative(name, getattr(self, name))
        for name in ("c_m", "phi", "tau_ca", "dt"):
            _positive(name, getattr(self, name))
        if not isinstance(self._sub_steps, int) or self._sub_steps <= 0:
            raise ValueError("_sub_steps must be a positive integer")

    @staticmethod
    def _validate_candidates(v: float, h: float, n: float, ca: float) -> None:
        _finite("membrane candidate", v)
        if v < -100.0 or v > 60.0:
            raise ValueError("membrane candidate left physiological safety bounds")
        _probability("h candidate", h)
        _probability("n candidate", n)
        _nonnegative("calcium candidate", ca)

    def step(self, current: float = 0.0) -> int:
        """Advance one dt. Returns 1 if spike, 0 otherwise."""
        self._validate()
        _finite("current", current)
        inp = self.gain * current
        _finite("input drive", inp)
        sub_dt = self.dt / self._sub_steps
        _positive("sub_dt", sub_dt)

        v = self.v
        h = self.h
        n = self.n
        ca = self.ca
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

            ca_next = max(ca + sub_dt * (-ca / self.tau_ca), 0.0)
            _nonnegative("calcium decay candidate", ca_next)
            denom = ca_next + 0.5
            _positive("BK calcium denominator", denom)
            v_half_bk = 10.0 - 30.0 * (ca_next / denom)
            _finite("BK half-activation", v_half_bk)
            bk_inf = 1.0 / (1.0 + _checked_exp(-(v - v_half_bk) / 15.0))
            _probability("BK activation", bk_inf)

            h_next = h + sub_dt * self.phi * (alpha_h * (1.0 - h) - beta_h * h)
            n_next = n + sub_dt * self.phi * (alpha_n * (1.0 - n) - beta_n * n)

            i_na = self.g_na * m_inf**3 * h_next * (v - self.e_na)
            i_k = self.g_k * n_next**4 * (v - self.e_k)
            i_bk = self.g_bk * bk_inf * (v - self.e_k)
            i_l = self.g_l * (v - self.e_l)
            dv = (-i_na - i_k - i_bk - i_l + inp) / self.c_m
            v_next = v + sub_dt * dv

            if v_next >= self.v_threshold:
                fired = 1
                v_next = -65.0
                ca_next += 0.3

            self._validate_candidates(v_next, h_next, n_next, ca_next)
            v, h, n, ca = v_next, h_next, n_next, ca_next

        self.v = v
        self.h = h
        self.n = n
        self.ca = ca
        return fired

    def reset(self) -> None:
        """Reset to default initial conditions."""
        self.v = -65.0
        self.h = 0.6
        self.n = 0.32
        self.ca = 0.0
