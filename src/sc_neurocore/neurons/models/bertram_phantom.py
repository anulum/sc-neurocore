# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bertram et al. 2000 four-state phantom burster

from __future__ import annotations

import math
from dataclasses import dataclass

_VOLTAGE_MIN = -250.0
_VOLTAGE_MAX = 250.0
_GATE_TOL = 1e-9


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real value")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real value")
    return result


def _positive_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _gate_value(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass
class BertramPhantomBurster:
    """Four-state phantom burster of Bertram et al. (2000).

    The ionic equations and defaults follow equations 1–10 and the authors'
    ``BJ_00.ode`` implementation. ``n`` is a dynamic fast potassium gate;
    ``s1`` and ``s2`` are the 1 s and 120 s negative-feedback gates. The
    production integrator is simultaneous fixed-step RK4, rather than the
    authors' adaptive CVODE run. ``current`` is an additive external-current
    extension in fA. Events are sampled upward ``v_threshold`` crossings and
    do not reset any state.

    Reference: Bertram R, Previte J, Sherman A, Kinard TA, Satin LS (2000),
    Biophysical Journal 79(6):2880–2892,
    doi:10.1016/S0006-3495(00)76525-8.
    """

    v: float = -43.0
    n: float = 0.03
    s1: float = 0.1
    s2: float = 0.434
    lambda_n: float = 1.1
    g_ca: float = 280.0
    g_k: float = 1300.0
    g_s1: float = 20.0
    g_s2: float = 32.0
    g_l: float = 25.0
    e_ca: float = 100.0
    e_k: float = -80.0
    e_l: float = -40.0
    c_m: float = 4524.0
    v_m: float = -22.0
    s_m: float = 7.5
    v_n: float = -9.0
    s_n: float = 10.0
    v_s1: float = -40.0
    s_s1: float = 0.5
    v_s2: float = -42.0
    s_s2: float = 0.4
    tau_n_bar: float = 9.09
    tau_s1: float = 1000.0
    tau_s2: float = 120000.0
    dt: float = 0.5
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        self.v = _finite_float("v", self.v)
        self.n = _gate_value("n", self.n)
        self.s1 = _gate_value("s1", self.s1)
        self.s2 = _gate_value("s2", self.s2)
        self.lambda_n = _positive_float("lambda_n", self.lambda_n)
        self.g_ca = _non_negative_float("g_ca", self.g_ca)
        self.g_k = _non_negative_float("g_k", self.g_k)
        self.g_s1 = _non_negative_float("g_s1", self.g_s1)
        self.g_s2 = _non_negative_float("g_s2", self.g_s2)
        self.g_l = _non_negative_float("g_l", self.g_l)
        self.e_ca = _finite_float("e_ca", self.e_ca)
        self.e_k = _finite_float("e_k", self.e_k)
        self.e_l = _finite_float("e_l", self.e_l)
        self.c_m = _positive_float("c_m", self.c_m)
        self.v_m = _finite_float("v_m", self.v_m)
        self.s_m = _positive_float("s_m", self.s_m)
        self.v_n = _finite_float("v_n", self.v_n)
        self.s_n = _positive_float("s_n", self.s_n)
        self.v_s1 = _finite_float("v_s1", self.v_s1)
        self.s_s1 = _positive_float("s_s1", self.s_s1)
        self.v_s2 = _finite_float("v_s2", self.v_s2)
        self.s_s2 = _positive_float("s_s2", self.s_s2)
        self.tau_n_bar = _positive_float("tau_n_bar", self.tau_n_bar)
        self.tau_s1 = _positive_float("tau_s1", self.tau_s1)
        self.tau_s2 = _positive_float("tau_s2", self.tau_s2)
        self.dt = _positive_float("dt", self.dt)
        self.v_threshold = _finite_float("v_threshold", self.v_threshold)
        self._validate_state()

    def _validate_state(self) -> None:
        self.v = _finite_float("v", self.v)
        if self.v < _VOLTAGE_MIN or self.v > _VOLTAGE_MAX:
            raise ValueError("v outside Bertram phantom safety envelope")
        self.n = _gate_value("n", self.n)
        self.s1 = _gate_value("s1", self.s1)
        self.s2 = _gate_value("s2", self.s2)

    @staticmethod
    def _boltz(v: float, midpoint: float, slope: float) -> float:
        z = (midpoint - v) / slope
        if z >= 0.0:
            exp_neg = math.exp(-z)
            return exp_neg / (1.0 + exp_neg)
        exp_pos = math.exp(z)
        return 1.0 / (1.0 + exp_pos)

    def _derivatives(
        self,
        v: float,
        n: float,
        s1: float,
        s2: float,
        current: float,
    ) -> tuple[float, float, float, float]:
        m_inf = self._boltz(v, self.v_m, self.s_m)
        n_inf = self._boltz(v, self.v_n, self.s_n)
        s1_inf = self._boltz(v, self.v_s1, self.s_s1)
        s2_inf = self._boltz(v, self.v_s2, self.s_s2)
        tau_n = self.tau_n_bar / (1.0 + math.exp((v - self.v_n) / self.s_n))

        i_ca = self.g_ca * m_inf * (v - self.e_ca)
        i_k = self.g_k * n * (v - self.e_k)
        i_s1 = self.g_s1 * s1 * (v - self.e_k)
        i_s2 = self.g_s2 * s2 * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)

        dv = (-i_ca - i_k - i_s1 - i_s2 - i_l + current) / self.c_m
        dn = self.lambda_n * (n_inf - n) / tau_n
        ds1 = (s1_inf - s1) / self.tau_s1
        ds2 = (s2_inf - s2) / self.tau_s2
        return dv, dn, ds1, ds2

    def _rk4_candidate(self, current: float) -> tuple[float, float, float, float]:
        state = (self.v, self.n, self.s1, self.s2)
        dt = self.dt
        k1 = self._derivatives(*state, current)
        k2_state = tuple(state[index] + 0.5 * dt * k1[index] for index in range(4))
        k2 = self._derivatives(k2_state[0], k2_state[1], k2_state[2], k2_state[3], current)
        k3_state = tuple(state[index] + 0.5 * dt * k2[index] for index in range(4))
        k3 = self._derivatives(k3_state[0], k3_state[1], k3_state[2], k3_state[3], current)
        k4_state = tuple(state[index] + dt * k3[index] for index in range(4))
        k4 = self._derivatives(k4_state[0], k4_state[1], k4_state[2], k4_state[3], current)
        return (
            state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        )

    @staticmethod
    def _validate_candidate(
        v: float,
        n: float,
        s1: float,
        s2: float,
    ) -> tuple[float, float, float, float]:
        if not all(math.isfinite(value) for value in (v, n, s1, s2)):
            raise ValueError("Bertram phantom candidate must be finite")
        if v < _VOLTAGE_MIN or v > _VOLTAGE_MAX:
            raise ValueError("v candidate outside Bertram phantom safety envelope")
        for name, value in (("n", n), ("s1", s1), ("s2", s2)):
            if value < -_GATE_TOL or value > 1.0 + _GATE_TOL:
                raise ValueError(f"{name} candidate must remain in [0, 1]")
        return v, min(1.0, max(0.0, n)), min(1.0, max(0.0, s1)), min(1.0, max(0.0, s2))

    def step(self, current: float) -> int:
        current = _finite_float("current", current)
        self._validate_state()
        v_previous = self.v
        self.v, self.n, self.s1, self.s2 = self._validate_candidate(*self._rk4_candidate(current))
        return int(self.v >= self.v_threshold and v_previous < self.v_threshold)

    def reset(self) -> None:
        self.v = -43.0
        self.n = 0.03
        self.s1 = 0.1
        self.s2 = 0.434
