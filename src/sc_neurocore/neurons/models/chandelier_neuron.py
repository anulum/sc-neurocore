# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chandelier Cell (Axo-Axonic Interneuron)

from __future__ import annotations

import math
from dataclasses import dataclass

_EXP_MAX = 709.0
_EXP_MIN = -745.0


def _checked_exp(x: float) -> float:
    if not math.isfinite(x) or x > _EXP_MAX:
        raise ValueError("chandelier neuron exponential argument is non-finite or unstable")
    if x < _EXP_MIN:
        return 0.0
    return math.exp(x)


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    rate = a * d / (1.0 - _checked_exp(-d / k))
    if not math.isfinite(rate):
        raise ValueError("chandelier neuron rate candidate is non-finite")
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
class ChandelierNeuron:
    """Axo-axonic fast-spiking interneuron with Kv1 and Kv3 currents."""

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    d: float = 0.0
    p: float = 0.0
    g_na: float = 35.0
    g_k: float = 9.0
    g_kv1: float = 3.0
    g_kv3: float = 4.0
    g_l: float = 0.1
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    dt: float = 0.01
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for name in ("v", "e_na", "e_k", "e_l", "v_threshold"):
            _finite(name, getattr(self, name))
        for name in ("h", "n", "d", "p"):
            _probability(name, getattr(self, name))
        for name in ("g_na", "g_k", "g_kv1", "g_kv3", "g_l"):
            _nonnegative(name, getattr(self, name))
        for name in ("c_m", "phi", "dt"):
            _positive(name, getattr(self, name))

    @staticmethod
    def _validate_candidates(v: float, h: float, n: float, d: float, p: float) -> None:
        _finite("membrane candidate", v)
        if v < -100.0 or v > 60.0:
            raise ValueError("membrane candidate left physiological safety bounds")
        for name, value in (("h", h), ("n", n), ("d", d), ("p", p)):
            _probability(f"{name} candidate", value)

    def step(self, current: float = 0.0) -> int:
        self._validate()
        _finite("current", current)
        v_prev = self.v
        n_sub = max(1, int(0.5 / max(self.dt, 0.001)))

        v = self.v
        h = self.h
        n = self.n
        d_gate = self.d
        p_gate = self.p

        for _ in range(n_sub):
            am = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            bm = 4.0 * _checked_exp(-(v + 60.0) / 18.0)
            m_inf = am / (am + bm)
            _finite("m_inf", m_inf)
            ah = 0.07 * _checked_exp(-(v + 58.0) / 20.0)
            bh = 1.0 / (1.0 + _checked_exp(-(v + 28.0) / 10.0))
            an = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            bn = 0.125 * _checked_exp(-(v + 44.0) / 80.0)

            h_next = h + self.phi * (ah * (1.0 - h) - bh * h) * self.dt
            n_next = n + self.phi * (an * (1.0 - n) - bn * n) * self.dt

            d_inf = 1.0 / (1.0 + _checked_exp(-(v + 50.0) / 10.0))
            d_next = d_gate + (d_inf - d_gate) / 150.0 * self.dt

            p_inf = 1.0 / (1.0 + _checked_exp(-(v + 10.0) / 10.0))
            p_next = p_gate + self.phi * (p_inf - p_gate) * self.dt

            i_na = self.g_na * m_inf**3 * h_next * (v - self.e_na)
            i_k = self.g_k * n_next**4 * (v - self.e_k)
            i_kv1 = self.g_kv1 * d_next**4 * (v - self.e_k)
            i_kv3 = self.g_kv3 * p_next * (v - self.e_k)
            i_l = self.g_l * (v - self.e_l)
            v_next = v + (-i_na - i_k - i_kv1 - i_kv3 - i_l + current) / self.c_m * self.dt

            self._validate_candidates(v_next, h_next, n_next, d_next, p_next)
            v, h, n, d_gate, p_gate = v_next, h_next, n_next, d_next, p_next

        self.v = v
        self.h = h
        self.n = n
        self.d = d_gate
        self.p = p_gate
        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.8
        self.n = 0.1
        self.d = 0.0
        self.p = 0.0
