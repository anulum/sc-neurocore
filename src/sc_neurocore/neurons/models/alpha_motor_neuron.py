# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha Motor Neuron

from __future__ import annotations

import math
from dataclasses import dataclass

_EXP_MAX = 709.0
_EXP_MIN = -745.0


def _checked_exp(x: float) -> float:
    if not math.isfinite(x) or x > _EXP_MAX:
        raise ValueError("alpha motor neuron exponential argument is non-finite or unstable")
    if x < _EXP_MIN:
        return 0.0
    return math.exp(x)


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    rate = a * d / (1.0 - _checked_exp(-d / k))
    if not math.isfinite(rate):
        raise ValueError("alpha motor neuron rate candidate is non-finite")
    return rate


def _check_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _check_probability(name: str, value: float) -> None:
    _check_finite(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must stay in [0, 1]")


def _check_nonnegative(name: str, value: float) -> None:
    _check_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


@dataclass
class AlphaMotorNeuron:
    """Alpha motor neuron with WB Na/K, PIC, and Ca-dependent AHP dynamics.

    The integrator is still the documented explicit substep path, but mutable
    runtime state is validated before integration and all substep candidates are
    computed locally before committing. Invalid state, unstable exponentials,
    or non-finite membrane/calcium candidates fail closed without poisoning the
    neuron state.
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    m_pic: float = 0.0
    h_pic: float = 1.0
    ca: float = 0.0
    ca_buf: float = 0.0
    g_na: float = 35.0
    g_k: float = 9.0
    g_pic: float = 0.15
    g_ahp: float = 3.0
    g_l: float = 0.3
    e_na: float = 55.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_l: float = -65.0
    c_m: float = 1.5
    phi: float = 4.0
    tau_ca: float = 150.0
    buf_ratio: float = 0.003
    dt: float = 0.01
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        for name in ("v", "e_na", "e_k", "e_ca", "e_l", "v_threshold"):
            _check_finite(name, getattr(self, name))
        for name in ("h", "n", "m_pic", "h_pic"):
            _check_probability(name, getattr(self, name))
        for name in ("ca", "ca_buf"):
            _check_nonnegative(name, getattr(self, name))
        for name in ("g_na", "g_k", "g_pic", "g_ahp", "g_l"):
            _check_nonnegative(name, getattr(self, name))
        for name in ("c_m", "phi", "tau_ca", "dt"):
            _check_finite(name, getattr(self, name))
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        _check_finite("buf_ratio", self.buf_ratio)
        if not 0.0 <= self.buf_ratio <= 1.0:
            raise ValueError("buf_ratio must stay in [0, 1]")

    @staticmethod
    def _validate_candidates(values: dict[str, float]) -> None:
        for name, value in values.items():
            _check_finite(name, value)
        for name in ("h", "n", "m_pic", "h_pic"):
            _check_probability(name, values[name])
        _check_nonnegative("ca", values["ca"])
        _check_nonnegative("ca_buf", values["ca_buf"])

    def step(self, current: float = 0.0) -> int:
        self._validate()
        _check_finite("current", current)

        v_prev = self.v
        v = self.v
        h = self.h
        n = self.n
        m_pic = self.m_pic
        h_pic = self.h_pic
        ca = self.ca
        ca_buf = self.ca_buf

        n_sub = max(1, int(0.5 / max(self.dt, 0.001)))
        for _ in range(n_sub):
            am = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            bm = 4.0 * _checked_exp(-(v + 60.0) / 18.0)
            m_inf = am / (am + bm)
            ah = 0.07 * _checked_exp(-(v + 58.0) / 20.0)
            bh = 1.0 / (1.0 + _checked_exp(-(v + 28.0) / 10.0))
            an = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            bn = 0.125 * _checked_exp(-(v + 44.0) / 80.0)

            h_next = h + self.phi * (ah * (1.0 - h) - bh * h) * self.dt
            n_next = n + self.phi * (an * (1.0 - n) - bn * n) * self.dt

            m_pic_inf = 1.0 / (1.0 + _checked_exp(-(v + 40.0) / 5.0))
            m_pic_next = m_pic + (m_pic_inf - m_pic) / 50.0 * self.dt

            h_pic_inf = 1.0 / (1.0 + _checked_exp((v + 40.0) / 8.0))
            tau_h_pic = 200.0 + 100.0 / max(0.01, 1.0 + ((v + 40.0) / 10.0) ** 2)
            h_pic_next = h_pic + (h_pic_inf - h_pic) / tau_h_pic * self.dt
            h_pic_next = max(0.0, min(1.0, h_pic_next))

            i_ca_entry = self.g_pic * m_pic_next * h_pic_next * (v - self.e_ca)
            ca_influx = -i_ca_entry * 0.001 if i_ca_entry < 0.0 else 0.0
            ca_spike = 0.02 if v > -10.0 else 0.0
            free_ca_change = (ca_influx + ca_spike) * self.buf_ratio
            ca_next = max(0.0, ca + (-ca / self.tau_ca + free_ca_change) * self.dt)
            ca_buf_next = max(
                0.0,
                ca_buf
                + ((ca_influx + ca_spike) * (1.0 - self.buf_ratio) - ca_buf / (self.tau_ca * 5.0))
                * self.dt,
            )

            ca_total = ca_next + ca_buf_next * 0.01
            ahp_inf = ca_total**2 / (ca_total**2 + 0.25)

            i_na = self.g_na * m_inf**3 * h_next * (v - self.e_na)
            i_k = self.g_k * n_next**4 * (v - self.e_k)
            i_pic = self.g_pic * m_pic_next * h_pic_next * (v - self.e_ca)
            i_ahp = self.g_ahp * ahp_inf * (v - self.e_k)
            i_l = self.g_l * (v - self.e_l)
            v_next = v + (-i_na - i_k - i_pic - i_ahp - i_l + current) / self.c_m * self.dt

            self._validate_candidates(
                {
                    "v": v_next,
                    "h": h_next,
                    "n": n_next,
                    "m_pic": m_pic_next,
                    "h_pic": h_pic_next,
                    "ca": ca_next,
                    "ca_buf": ca_buf_next,
                }
            )
            v, h, n, m_pic, h_pic, ca, ca_buf = (
                v_next,
                h_next,
                n_next,
                m_pic_next,
                h_pic_next,
                ca_next,
                ca_buf_next,
            )

        self.v = v
        self.h = h
        self.n = n
        self.m_pic = m_pic
        self.h_pic = h_pic
        self.ca = ca
        self.ca_buf = ca_buf
        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        self.v = -65.0
        self.h = 0.8
        self.n = 0.1
        self.m_pic = 0.0
        self.h_pic = 1.0
        self.ca = 0.0
        self.ca_buf = 0.0
