# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chay 1985 — pancreatic beta cell burster

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ChayNeuron:
    """Chay 1985 pancreatic beta-cell burster with guarded stiff integration.

    Reference: Chay, T.R. (1985). Physica D 16:233-242.
    """

    v: float = -50.0
    n: float = 0.1
    ca: float = 0.1
    g_ca: float = 25.0
    g_k: float = 1400.0
    g_kca: float = 12.0
    g_l: float = 7.0
    e_ca: float = 100.0
    e_k: float = -75.0
    e_l: float = -40.0
    rho: float = 0.00015
    alpha_ca: float = 0.002
    k_ca: float = 0.04
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
            raise ValueError("v outside Chay safety envelope")
        n = self._probability(self.n, "n")
        ca = self._nonnegative(self.ca, "ca")
        if ca > self._CA_MAX:
            raise ValueError("ca outside Chay safety envelope")

        self._nonnegative(self.g_ca, "g_ca")
        self._nonnegative(self.g_k, "g_k")
        self._nonnegative(self.g_kca, "g_kca")
        self._nonnegative(self.g_l, "g_l")
        self._finite(self.e_ca, "e_ca")
        self._finite(self.e_k, "e_k")
        self._finite(self.e_l, "e_l")
        self._nonnegative(self.rho, "rho")
        self._nonnegative(self.alpha_ca, "alpha_ca")
        self._nonnegative(self.k_ca, "k_ca")
        dt = self._positive(self.dt, "dt")
        self._finite(self.v_threshold, "v_threshold")

        substeps = max(1, math.ceil(dt / self._MAX_SUBSTEP))
        if substeps > 10000:
            raise ValueError("dt requires too many Chay safety substeps")
        return v, n, ca, substeps, dt / substeps

    def _candidate(
        self, v: float, n: float, ca: float, h: float, current: float
    ) -> tuple[float, float, float]:
        m_inf = self._gate_inf(-(v + 25.0) / 8.0, "m_inf exponent")
        n_inf = self._gate_inf(-(v + 18.0) / 14.0, "n_inf exponent")
        tau_n = 1.0 / (0.01 * max(abs(v + 18.0), 0.01))
        ca_denominator = ca + 1.0
        if ca_denominator <= 0.0:
            raise ValueError("calcium activation denominator must be positive")

        i_ca = self.g_ca * m_inf * (v - self.e_ca)
        kca_act = ca / ca_denominator
        i_k = self.g_k * n * (v - self.e_k)
        i_kca = self.g_kca * kca_act * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)

        v_next = v + (-i_ca - i_k - i_kca - i_l + current) * h
        n_next = n + (n_inf - n) / max(tau_n, 0.01) * h
        ca_next = ca + self.rho * (-self.alpha_ca * i_ca - self.k_ca * ca) * h

        if not math.isfinite(v_next):
            raise ValueError("Chay voltage candidate must be finite")
        if not self._V_MIN <= v_next <= self._V_MAX:
            raise ValueError("Chay voltage candidate outside safety envelope")
        if not math.isfinite(n_next) or not 0.0 <= n_next <= 1.0:
            raise ValueError("Chay n-gate candidate outside [0, 1]")
        if not math.isfinite(ca_next) or not 0.0 <= ca_next <= self._CA_MAX:
            raise ValueError("Chay calcium candidate outside safety envelope")
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
        self.v, self.n, self.ca = -50.0, 0.1, 0.1
