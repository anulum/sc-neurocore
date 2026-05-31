# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chay & Keizer 1983 — pancreatic beta cell with Ca-dependent K

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ChayKeizerNeuron:
    """Chay-Keizer pancreatic beta-cell model with guarded Ca-K dynamics.

    Reference: Chay, T.R. & Keizer, J. (1983). Biophys. J. 42:181-190.
    """

    v: float = -50.0
    n: float = 0.01
    ca: float = 0.1
    g_ca: float = 20.0
    g_k: float = 25.0
    g_kca: float = 12.0
    g_l: float = 0.1
    e_ca: float = 100.0
    e_k: float = -75.0
    e_l: float = -40.0
    k_d: float = 1.0
    f_ca: float = 0.004
    k_ca: float = 0.03
    dt: float = 0.02
    v_threshold: float = -20.0

    _MAX_SUBSTEP: float = 0.001
    _V_MIN: float = -200.0
    _V_MAX: float = 200.0
    _CA_MAX: float = 100.0

    @staticmethod
    def _finite(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    @classmethod
    def _nonnegative(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @classmethod
    def _probability(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return value

    @classmethod
    def _checked_exp(cls, exponent: float, name: str) -> float:
        exponent = cls._finite(exponent, name)
        if exponent < -700.0:
            return 0.0
        if exponent > 700.0:
            return math.exp(700.0)
        return math.exp(exponent)

    @classmethod
    def _gate_inf(cls, exponent: float, name: str) -> float:
        return 1.0 / (1.0 + cls._checked_exp(exponent, name))

    def _validated_state(self) -> tuple[float, float, float, int, float]:
        v = self._finite(self.v, "v")
        if not self._V_MIN <= v <= self._V_MAX:
            raise ValueError("v outside Chay-Keizer safety envelope")
        n = self._probability(self.n, "n")
        ca = self._nonnegative(self.ca, "ca")
        if ca > self._CA_MAX:
            raise ValueError("ca outside Chay-Keizer safety envelope")

        self._nonnegative(self.g_ca, "g_ca")
        self._nonnegative(self.g_k, "g_k")
        self._nonnegative(self.g_kca, "g_kca")
        self._nonnegative(self.g_l, "g_l")
        self._finite(self.e_ca, "e_ca")
        self._finite(self.e_k, "e_k")
        self._finite(self.e_l, "e_l")
        self._positive(self.k_d, "k_d")
        self._nonnegative(self.f_ca, "f_ca")
        self._nonnegative(self.k_ca, "k_ca")
        dt = self._positive(self.dt, "dt")
        self._finite(self.v_threshold, "v_threshold")

        substeps = max(1, math.ceil(dt / self._MAX_SUBSTEP))
        if substeps > 10000:
            raise ValueError("dt requires too many Chay-Keizer safety substeps")
        return v, n, ca, substeps, dt / substeps

    def _candidate(
        self, v: float, n: float, ca: float, h: float, current: float
    ) -> tuple[float, float, float]:
        m_inf = self._gate_inf(-(v + 25.0) / 8.0, "m_inf exponent")
        n_inf = self._gate_inf(-(v + 18.0) / 14.0, "n_inf exponent")
        tau_n_denominator = 1.0 + self._checked_exp((v + 18.0) / 14.0, "tau_n exponent")
        tau_n = 20.0 / tau_n_denominator
        ca_denominator = ca + self.k_d
        if ca_denominator <= 0.0:
            raise ValueError("calcium activation denominator must be positive")

        q_kca = ca / ca_denominator
        i_ca = self.g_ca * m_inf * (v - self.e_ca)
        i_k = self.g_k * n * (v - self.e_k)
        i_kca = self.g_kca * q_kca * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)

        v_next = v + (-i_ca - i_k - i_kca - i_l + current) * h
        n_next = n + (n_inf - n) / max(tau_n, 0.1) * h
        ca_next = ca + (-self.f_ca * i_ca - self.k_ca * ca) * h

        if not math.isfinite(v_next) or not self._V_MIN <= v_next <= self._V_MAX:
            raise ValueError("Chay-Keizer voltage candidate outside safety envelope")
        if not math.isfinite(n_next) or not 0.0 <= n_next <= 1.0:
            raise ValueError("Chay-Keizer n-gate candidate outside [0, 1]")
        if not math.isfinite(ca_next) or not 0.0 <= ca_next <= self._CA_MAX:
            raise ValueError("Chay-Keizer calcium candidate outside safety envelope")
        return v_next, n_next, ca_next

    def step(self, current: float) -> int:
        """Advance one timestep and return an upward-threshold spike flag."""

        current = self._finite(current, "current")
        v_initial = self.v
        v, n, ca, substeps, h = self._validated_state()
        crossed = False
        for _ in range(substeps):
            v_next, n_next, ca_next = self._candidate(v, n, ca, h, current)
            crossed = crossed or (v_next >= self.v_threshold and v < self.v_threshold)
            v, n, ca = v_next, n_next, ca_next

        self.v = v
        self.n = n
        self.ca = ca
        return 1 if crossed and v_initial < self.v_threshold else 0

    def reset(self) -> None:
        self.v, self.n, self.ca = -50.0, 0.01, 0.1
