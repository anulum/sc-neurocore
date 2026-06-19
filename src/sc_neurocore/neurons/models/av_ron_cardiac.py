# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Av-Ron, Parnas & Segel 1993 — cardiac ganglion Type III

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AvRonCardiacNeuron:
    """Av-Ron, Parnas & Segel cardiac ganglion Type III burster.

    The four-state conductance model uses instantaneous sodium activation,
    voltage-dependent h/n/s gate relaxation, and candidate-first RK4 integration
    so invalid states cannot partially mutate the neuron.
    """

    v: float = -60.0
    h: float = 0.6
    n: float = 0.3
    s: float = 0.5
    g_na: float = 80.0
    g_k: float = 40.0
    g_s: float = 20.0
    g_l: float = 0.1
    e_na: float = 40.0
    e_k: float = -80.0
    e_s: float = -25.0
    e_l: float = -60.0
    dt: float = 0.02
    v_threshold: float = -20.0

    @staticmethod
    def _finite_values(values: tuple[float, ...]) -> bool:
        return all(math.isfinite(value) for value in values)

    @staticmethod
    def _gate_in_range(value: float) -> bool:
        return 0.0 <= value <= 1.0

    @staticmethod
    def _bounded_exp(value: float) -> float:
        return math.exp(max(min(value, 709.0), -745.0))

    @classmethod
    def _sigmoid_pos(cls, value: float) -> float:
        return 1.0 / (1.0 + cls._bounded_exp(-value))

    @classmethod
    def _sigmoid_neg(cls, value: float) -> float:
        return 1.0 / (1.0 + cls._bounded_exp(value))

    def _valid_runtime(self) -> bool:
        return (
            self._finite_values(
                (
                    self.v,
                    self.h,
                    self.n,
                    self.s,
                    self.g_na,
                    self.g_k,
                    self.g_s,
                    self.g_l,
                    self.e_na,
                    self.e_k,
                    self.e_s,
                    self.e_l,
                    self.dt,
                    self.v_threshold,
                )
            )
            and self.dt > 0.0
            and self.g_na >= 0.0
            and self.g_k >= 0.0
            and self.g_s >= 0.0
            and self.g_l >= 0.0
            and self._gate_in_range(self.h)
            and self._gate_in_range(self.n)
            and self._gate_in_range(self.s)
        )

    def _rates(self, voltage: float) -> tuple[float, float, float, float, float, float, float]:
        m_inf = self._sigmoid_pos((voltage + 40.0) / 7.0)
        h_inf = self._sigmoid_neg((voltage + 45.0) / 5.0)
        n_inf = self._sigmoid_pos((voltage + 40.0) / 15.0)
        s_inf = self._sigmoid_neg((voltage + 35.0) / 3.0)
        tau_h = 1.0 + 12.0 * self._sigmoid_neg((voltage + 50.0) / 8.0)
        tau_n = 1.0 + 8.0 * self._sigmoid_neg((voltage + 35.0) / 8.0)
        tau_s = 200.0 + 1000.0 * self._sigmoid_neg((voltage + 30.0) / 5.0)
        return m_inf, h_inf, n_inf, s_inf, tau_h, tau_n, tau_s

    def _derivatives(
        self, state: tuple[float, float, float, float], current: float
    ) -> tuple[float, float, float, float]:
        voltage, h_gate, n_gate, s_gate = state
        if not self._finite_values(state) or not (
            self._gate_in_range(h_gate)
            and self._gate_in_range(n_gate)
            and self._gate_in_range(s_gate)
        ):
            return (math.nan, math.nan, math.nan, math.nan)
        m_inf, h_inf, n_inf, s_inf, tau_h, tau_n, tau_s = self._rates(voltage)
        i_na = self.g_na * m_inf**3 * h_gate * (voltage - self.e_na)
        i_k = self.g_k * n_gate**4 * (voltage - self.e_k)
        i_s = self.g_s * s_gate * (voltage - self.e_s)
        i_l = self.g_l * (voltage - self.e_l)
        return (
            -i_na - i_k - i_s - i_l + current,
            (h_inf - h_gate) / tau_h,
            (n_inf - n_gate) / tau_n,
            (s_inf - s_gate) / tau_s,
        )

    @staticmethod
    def _add_scaled(
        state: tuple[float, float, float, float],
        slope: tuple[float, float, float, float],
        scale: float,
    ) -> tuple[float, float, float, float]:
        return (
            state[0] + scale * slope[0],
            state[1] + scale * slope[1],
            state[2] + scale * slope[2],
            state[3] + scale * slope[3],
        )

    def _rk4_candidate(self, current: float) -> tuple[float, float, float, float] | None:
        state = (self.v, self.h, self.n, self.s)
        half_dt = 0.5 * self.dt
        k1 = self._derivatives(state, current)
        k2 = self._derivatives(self._add_scaled(state, k1, half_dt), current)
        k3 = self._derivatives(self._add_scaled(state, k2, half_dt), current)
        k4 = self._derivatives(self._add_scaled(state, k3, self.dt), current)
        candidate = (
            state[0] + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + self.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + self.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        )
        if self._finite_values(candidate) and all(
            self._gate_in_range(value) for value in candidate[1:]
        ):
            return candidate
        return None

    def step(self, current: float) -> int:
        if not math.isfinite(current) or not self._valid_runtime():
            return 0
        v_prev = self.v
        candidate = self._rk4_candidate(current)
        if candidate is None:
            return 0
        self.v, self.h, self.n, self.s = candidate
        return int(self.v >= self.v_threshold and v_prev < self.v_threshold)

    def reset(self) -> None:
        self.v = -60.0
        self.h = 0.6
        self.n = 0.3
        self.s = 0.5
