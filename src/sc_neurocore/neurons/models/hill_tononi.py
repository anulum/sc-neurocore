# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hill and Tononi 2005 hybrid thalamocortical neuron

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

_State = tuple[float, float, float, float, float, float]

_STATE_NAMES = ("v", "theta", "d_k", "m_h", "m_t", "h_t", "spike_timer")
_PARAM_NAMES = (
    "g_na_l",
    "g_k_l",
    "g_na_p",
    "g_dk",
    "g_h",
    "g_t",
    "e_na",
    "e_k",
    "e_na_p",
    "e_dk",
    "e_h",
    "e_t",
    "n_na_p",
    "n_t",
    "tau_m",
    "theta_eq",
    "tau_theta",
    "g_spike",
    "t_spike",
    "tau_spike",
    "tau_d",
    "d_influx_peak",
    "d_threshold",
    "d_slope",
    "d_eq",
    "d_half",
    "dt",
)
_STRICTLY_POSITIVE_PARAMS = (
    "n_na_p",
    "n_t",
    "tau_m",
    "tau_theta",
    "t_spike",
    "tau_spike",
    "tau_d",
    "d_slope",
    "d_half",
    "dt",
)
_NON_NEGATIVE_PARAMS = (
    "g_na_l",
    "g_k_l",
    "g_na_p",
    "g_dk",
    "g_h",
    "g_t",
    "g_spike",
    "d_influx_peak",
    "d_eq",
)


def _safe_exp(value: float) -> float:
    """Return ``exp(value)`` and saturate overflow at positive infinity."""
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


@dataclass
class HillTononiNeuron:
    """Hill and Tononi's hybrid integrate-and-fire neuron.

    The continuous state is ``(V, theta, D, m_h, m_T, h_T)``. A spike sets
    both ``V`` and the dynamic threshold ``theta`` to ``E_Na`` and enables
    a brief potassium repolarisation pulse. ``D`` is the generic
    depolarisation measure used by ``I_DK``; the paper explicitly does not
    integrate intracellular sodium or calcium concentration.

    Defaults select the paper's cortical-excitatory waking profile. Sodium and
    potassium leaks, ``I_NaP``, and ``I_DK`` are active. ``I_h`` and ``I_T``
    remain available with zero default conductance because the source assigns
    them only to specific intrinsically bursting or thalamic cell types. The
    recurrence uses source step ``dt=0.25 ms`` and classical RK4. Synaptic
    conductance dynamics, minis, and the full network are outside this
    single-cell catalogue model.

    Primary source: Hill & Tononi, J Neurophysiol 93:1671–1698 (2005),
    doi:10.1152/jn.00915.2004. Maintained NEST ``ht_neuron`` equations
    disambiguate parentheses in the printed ``I_NaP``, ``D``, and ``I_T``
    formulae.
    """

    v: float = -70.0
    theta: float = -51.0
    d_k: float = 0.001
    m_h: float = 0.2871859013825026
    m_t: float = 0.1450215950687922
    h_t: float = 0.03732688734412946
    spike_timer: float = 0.0

    g_na_l: float = 0.2
    g_k_l: float = 1.0
    g_na_p: float = 0.5
    g_dk: float = 0.5
    g_h: float = 0.0
    g_t: float = 0.0
    e_na: float = 30.0
    e_k: float = -90.0
    e_na_p: float = 30.0
    e_dk: float = -90.0
    e_h: float = -40.0
    e_t: float = 0.0
    n_na_p: float = 3.0
    n_t: float = 2.0

    tau_m: float = 16.0
    theta_eq: float = -51.0
    tau_theta: float = 2.0
    g_spike: float = 1.0
    t_spike: float = 2.0
    tau_spike: float = 1.75
    tau_d: float = 1250.0
    d_influx_peak: float = 0.025
    d_threshold: float = -10.0
    d_slope: float = 5.0
    d_eq: float = 0.001
    d_half: float = 0.25
    dt: float = 0.25

    def __post_init__(self) -> None:
        self._validate_configuration(coerce=True)

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def _validate_configuration(self, *, coerce: bool = False) -> None:
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            value = self._finite_float(name, getattr(self, name))
            if coerce:
                setattr(self, name, value)
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE_PARAMS:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.d_k < 0.0:
            raise ValueError("d_k must be non-negative")
        if self.spike_timer < 0.0:
            raise ValueError("spike_timer must be non-negative")

    @staticmethod
    def m_h_inf(v: float) -> float:
        return 1.0 / (1.0 + _safe_exp((v + 75.0) / 5.5))

    @staticmethod
    def tau_m_h(v: float) -> float:
        return 1.0 / (_safe_exp(-14.59 - 0.086 * v) + _safe_exp(-1.87 + 0.0701 * v))

    @staticmethod
    def m_t_inf(v: float) -> float:
        return 1.0 / (1.0 + _safe_exp(-(v + 59.0) / 6.2))

    @staticmethod
    def tau_m_t(v: float) -> float:
        denominator = _safe_exp(-(v + 132.0) / 16.7) + _safe_exp((v + 16.8) / 18.2)
        return 0.22 / denominator + 0.13

    @staticmethod
    def h_t_inf(v: float) -> float:
        return 1.0 / (1.0 + _safe_exp((v + 83.0) / 4.0))

    @staticmethod
    def tau_h_t(v: float) -> float:
        return 8.2 + (56.6 + 0.27 * _safe_exp((v + 115.2) / 5.0)) / (
            1.0 + _safe_exp((v + 86.0) / 3.2)
        )

    def d_k_inf(self, v: float) -> float:
        influx = self.d_influx_peak / (1.0 + _safe_exp(-(v - self.d_threshold) / self.d_slope))
        return self.tau_d * influx + self.d_eq

    def _derivatives(
        self,
        state: _State,
        current: float,
        spike_active: bool,
    ) -> _State:
        v, theta, d_k, m_h, m_t, h_t = state
        if not all(math.isfinite(value) for value in (*state, current)):
            raise FloatingPointError("Hill-Tononi derivative input became non-finite")

        m_na_p = 1.0 / (1.0 + _safe_exp(-(v + 55.7) / 7.7))
        d_activation = 1.0 / (1.0 + (self.d_half / max(d_k, 1e-15)) ** 3.5)
        i_na_l = -self.g_na_l * (v - self.e_na)
        i_k_l = -self.g_k_l * (v - self.e_k)
        i_na_p = -self.g_na_p * m_na_p**self.n_na_p * (v - self.e_na_p)
        i_dk = -self.g_dk * d_activation * (v - self.e_dk)
        i_h = -self.g_h * m_h * (v - self.e_h)
        i_t = -self.g_t * m_t**self.n_t * h_t * (v - self.e_t)
        i_spike = -self.g_spike * (v - self.e_k) / self.tau_spike if spike_active else 0.0

        derivatives = (
            (i_na_l + i_k_l + i_na_p + i_dk + i_h + i_t + current) / self.tau_m + i_spike,
            -(theta - self.theta_eq) / self.tau_theta,
            (self.d_k_inf(v) - d_k) / self.tau_d,
            (self.m_h_inf(v) - m_h) / self.tau_m_h(v),
            (self.m_t_inf(v) - m_t) / self.tau_m_t(v),
            (self.h_t_inf(v) - h_t) / self.tau_h_t(v),
        )
        if not all(math.isfinite(value) for value in derivatives):
            raise FloatingPointError("Hill-Tononi derivative became non-finite")
        return derivatives

    def _rk4_candidate(
        self,
        state: _State,
        current: float,
        spike_active: bool,
    ) -> _State:
        dt = self.dt
        k1 = self._derivatives(state, current, spike_active)
        s2 = cast(_State, tuple(value + 0.5 * dt * slope for value, slope in zip(state, k1)))
        k2 = self._derivatives(s2, current, spike_active)
        s3 = cast(_State, tuple(value + 0.5 * dt * slope for value, slope in zip(state, k2)))
        k3 = self._derivatives(s3, current, spike_active)
        s4 = cast(_State, tuple(value + dt * slope for value, slope in zip(state, k3)))
        k4 = self._derivatives(s4, current, spike_active)
        return cast(
            _State,
            tuple(
                value + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
                for value, a, b, c, d in zip(state, k1, k2, k3, k4)
            ),
        )

    def step(self, current: float = 0.0) -> int:
        """Advance one source timestep and return ``1`` on spike emission."""
        current = self._finite_float("current", current)
        self._validate_configuration()
        refractory = self.spike_timer > 0.0
        state = (self.v, self.theta, self.d_k, self.m_h, self.m_t, self.h_t)
        candidate = self._rk4_candidate(state, current, refractory)
        if not all(math.isfinite(value) for value in candidate):
            raise FloatingPointError("Hill-Tononi candidate became non-finite")
        if candidate[2] < 0.0:
            raise FloatingPointError("Hill-Tononi D candidate became negative")

        timer = max(0.0, self.spike_timer - self.dt)
        spike = not refractory and candidate[0] >= candidate[1]
        if spike:
            candidate = (self.e_na, self.e_na, *candidate[2:])
            timer = self.t_spike

        self.v, self.theta, self.d_k, self.m_h, self.m_t, self.h_t = candidate
        self.spike_timer = timer
        return int(spike)

    def reset(self) -> None:
        """Restore the source cortical-excitatory waking initial state."""
        self.v = -70.0
        self.theta = -51.0
        self.d_k = 0.001
        self.m_h = self.m_h_inf(self.v)
        self.m_t = self.m_t_inf(self.v)
        self.h_t = self.h_t_inf(self.v)
        self.spike_timer = 0.0
