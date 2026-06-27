# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hay et al. 2011 — Layer 5 thick-tufted pyramidal cell

"""Reduced three-compartment Hay L5 pyramidal-cell model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_STATE_NAMES = ("v_s", "h_na", "n_k", "v_t", "m_ca", "h_ca", "m_ih", "v_a", "ca_a")
_PARAM_NAMES = (
    "g_na",
    "g_k",
    "g_l_s",
    "e_na",
    "e_k",
    "e_l",
    "g_ca_t",
    "g_ih",
    "g_l_t",
    "e_ca",
    "e_ih",
    "g_ca_a",
    "g_kca",
    "g_l_a",
    "g_st",
    "g_ta",
    "p_s",
    "p_t",
    "p_a",
    "ca_decay",
    "f_ca",
    "dt",
    "v_threshold",
    "c_m",
)
_NON_NEGATIVE_PARAMS = (
    "g_na",
    "g_k",
    "g_l_s",
    "g_ca_t",
    "g_ih",
    "g_l_t",
    "g_ca_a",
    "g_kca",
    "g_l_a",
    "g_st",
    "g_ta",
    "f_ca",
)
_STRICTLY_POSITIVE_PARAMS = ("p_s", "p_t", "p_a", "ca_decay", "dt", "c_m")
_N_SUBSTEPS = 4
_State = tuple[float, float, float, float, float, float, float, float, float]


def _safe_exp(value: float) -> float:
    """Return ``exp(value)`` and map overflow to ``math.inf``.

    Parameters
    ----------
    value:
        Exponent to evaluate.

    Returns
    -------
    float
        Exponential value in double precision, or positive infinity when the
        exponent overflows.
    """
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


@dataclass
class HayL5PyramidalNeuron:
    """Reduced Layer 5 thick-tufted pyramidal-cell model after Hay et al. 2011.

    The maintained production surface is a compact three-compartment reduction
    with soma, apical trunk, and apical tuft voltages plus six gates/calcium
    state variables. It preserves the public dual-input API
    ``step(current_soma, current_tuft=0.0)`` while moving the default numerical
    path to candidate-first RK4. The historical explicit Euler path remains
    available through ``integrator="baseline_euler"`` for regression and
    benchmark comparisons.

    Reference: Hay, E. et al. (2011). PLoS Comput. Biol. 7:e1002107.
    """

    v_s: float = -75.0
    h_na: float = 0.9
    n_k: float = 0.1
    v_t: float = -75.0
    m_ca: float = 0.0
    h_ca: float = 1.0
    m_ih: float = 0.0
    v_a: float = -75.0
    ca_a: float = 0.0001
    g_na: float = 300.0
    g_k: float = 40.0
    g_l_s: float = 0.03
    e_na: float = 50.0
    e_k: float = -85.0
    e_l: float = -75.0
    g_ca_t: float = 2.0
    g_ih: float = 0.02
    g_l_t: float = 0.03
    e_ca: float = 140.0
    e_ih: float = -45.0
    g_ca_a: float = 1.5
    g_kca: float = 2.5
    g_l_a: float = 0.03
    g_st: float = 1.5
    g_ta: float = 0.8
    p_s: float = 0.15
    p_t: float = 0.25
    p_a: float = 0.60
    ca_decay: float = 200.0
    f_ca: float = 0.0002
    dt: float = 0.025
    v_threshold: float = -30.0
    c_m: float = 1.0
    integrator: Literal["rk4", "baseline_euler"] = "rk4"

    def __post_init__(self) -> None:
        """Validate the selected integrator and numeric configuration."""
        if self.integrator not in {"rk4", "baseline_euler"}:
            raise ValueError(f"Unsupported integrator for HayL5PyramidalNeuron: {self.integrator}")
        self._validate_configuration()

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        """Return ``value`` as a finite float and reject booleans.

        Parameters
        ----------
        name:
            Field name used in error messages.
        value:
            Candidate numeric value.

        Returns
        -------
        float
            Finite floating-point value.
        """
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def _validate_configuration(self) -> None:
        """Coerce state and parameters to finite floats and enforce signs."""
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            setattr(self, name, self._finite_float(name, getattr(self, name)))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE_PARAMS:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.ca_a < 0.0:
            raise ValueError("ca_a must be non-negative")

    def _validate_runtime_configuration(self) -> None:
        """Re-check state and parameter validity before mutating state."""
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            self._finite_float(name, getattr(self, name))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE_PARAMS:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.ca_a < 0.0:
            raise ValueError("ca_a must be non-negative")

    def _derivatives(self, state: _State, current_soma: float, current_tuft: float) -> _State:
        """Return derivatives for all nine state variables.

        Parameters
        ----------
        state:
            Candidate state in ``(v_s, h_na, n_k, v_t, m_ca, h_ca, m_ih, v_a,
            ca_a)`` order.
        current_soma:
            External somatic current.
        current_tuft:
            External apical-tuft current.

        Returns
        -------
        tuple of float
            Nine derivatives in state order.
        """
        if not all(math.isfinite(value) for value in (*state, current_soma, current_tuft)):
            raise FloatingPointError("Hay L5 derivative input became non-finite")
        v_s, h_na, n_k, v_t, m_ca, h_ca, m_ih, v_a, ca_a = state
        ca_eff = max(ca_a, 0.0)

        m_na_inf = 1.0 / (1.0 + _safe_exp(-(v_s + 38.0) / 7.0))
        h_na_inf = 1.0 / (1.0 + _safe_exp((v_s + 65.0) / 6.0))
        n_k_inf = 1.0 / (1.0 + _safe_exp(-(v_s + 25.0) / 12.0))
        tau_h = 0.5 + 14.0 / (1.0 + _safe_exp((v_s + 35.0) / 10.0))
        tau_n = 1.0 + 5.0 / (1.0 + _safe_exp((v_s + 30.0) / 10.0))
        d_h_na = (h_na_inf - h_na) / tau_h
        d_n_k = (n_k_inf - n_k) / tau_n
        i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v_s - self.e_na)
        i_k = self.g_k * n_k * n_k * n_k * n_k * (v_s - self.e_k)
        i_l_s = self.g_l_s * (v_s - self.e_l)
        i_st = self.g_st * (v_s - v_t) / self.p_s

        m_ca_inf = 1.0 / (1.0 + _safe_exp(-(v_t + 27.0) / 7.0))
        h_ca_inf = 1.0 / (1.0 + _safe_exp((v_t + 52.0) / 5.0))
        m_ih_inf = 1.0 / (1.0 + _safe_exp((v_t + 75.0) / 5.5))
        d_m_ca = m_ca_inf - m_ca
        d_h_ca = (h_ca_inf - h_ca) / 20.0
        d_m_ih = (m_ih_inf - m_ih) / 50.0
        i_ca_t = self.g_ca_t * m_ca * m_ca * h_ca * (v_t - self.e_ca)
        i_ih = self.g_ih * m_ih * (v_t - self.e_ih)
        i_l_t = self.g_l_t * (v_t - self.e_l)
        i_ts = self.g_st * (v_t - v_s) / self.p_t
        i_ta = self.g_ta * (v_t - v_a) / self.p_t

        m_ca_a_inf = 1.0 / (1.0 + _safe_exp(-(v_a + 30.0) / 5.0))
        kca_act = ca_eff / (ca_eff + 0.001)
        i_ca_a = self.g_ca_a * m_ca_a_inf * m_ca_a_inf * (v_a - self.e_ca)
        i_kca = self.g_kca * kca_act * (v_a - self.e_k)
        i_l_a = self.g_l_a * (v_a - self.e_l)
        i_at = self.g_ta * (v_a - v_t) / self.p_a

        d_v_s = (-i_na - i_k - i_l_s - i_st + current_soma / self.p_s) / self.c_m
        d_v_t = (-i_ca_t - i_ih - i_l_t - i_ts - i_ta) / self.c_m
        d_v_a = (-i_ca_a - i_kca - i_l_a - i_at + current_tuft / self.p_a) / self.c_m
        d_ca_a = -self.f_ca * i_ca_a - ca_eff / self.ca_decay

        derivatives = (d_v_s, d_h_na, d_n_k, d_v_t, d_m_ca, d_h_ca, d_m_ih, d_v_a, d_ca_a)
        if not all(math.isfinite(value) for value in derivatives):
            raise FloatingPointError("Hay L5 derivative became non-finite")
        return derivatives

    def _rk4_substep(self, state: _State, current_soma: float, current_tuft: float) -> _State:
        """Return one RK4 candidate for the nine-state vector.

        Parameters
        ----------
        state:
            State at substep start.
        current_soma:
            Somatic current held constant across the RK4 stages.
        current_tuft:
            Tuft current held constant across the RK4 stages.

        Returns
        -------
        tuple of float
            Candidate state after one internal substep.
        """
        dt = self.dt
        k1 = self._derivatives(state, current_soma, current_tuft)
        s2 = (
            state[0] + 0.5 * dt * k1[0],
            state[1] + 0.5 * dt * k1[1],
            state[2] + 0.5 * dt * k1[2],
            state[3] + 0.5 * dt * k1[3],
            state[4] + 0.5 * dt * k1[4],
            state[5] + 0.5 * dt * k1[5],
            state[6] + 0.5 * dt * k1[6],
            state[7] + 0.5 * dt * k1[7],
            state[8] + 0.5 * dt * k1[8],
        )
        k2 = self._derivatives(s2, current_soma, current_tuft)
        s3 = (
            state[0] + 0.5 * dt * k2[0],
            state[1] + 0.5 * dt * k2[1],
            state[2] + 0.5 * dt * k2[2],
            state[3] + 0.5 * dt * k2[3],
            state[4] + 0.5 * dt * k2[4],
            state[5] + 0.5 * dt * k2[5],
            state[6] + 0.5 * dt * k2[6],
            state[7] + 0.5 * dt * k2[7],
            state[8] + 0.5 * dt * k2[8],
        )
        k3 = self._derivatives(s3, current_soma, current_tuft)
        s4 = (
            state[0] + dt * k3[0],
            state[1] + dt * k3[1],
            state[2] + dt * k3[2],
            state[3] + dt * k3[3],
            state[4] + dt * k3[4],
            state[5] + dt * k3[5],
            state[6] + dt * k3[6],
            state[7] + dt * k3[7],
            state[8] + dt * k3[8],
        )
        k4 = self._derivatives(s4, current_soma, current_tuft)
        return (
            state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
            state[4] + dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0,
            state[5] + dt * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]) / 6.0,
            state[6] + dt * (k1[6] + 2.0 * k2[6] + 2.0 * k3[6] + k4[6]) / 6.0,
            state[7] + dt * (k1[7] + 2.0 * k2[7] + 2.0 * k3[7] + k4[7]) / 6.0,
            state[8] + dt * (k1[8] + 2.0 * k2[8] + 2.0 * k3[8] + k4[8]) / 6.0,
        )

    def _euler_substep(self, state: _State, current_soma: float, current_tuft: float) -> _State:
        """Return one explicit Euler candidate for regression comparison.

        Parameters
        ----------
        state:
            State at substep start.
        current_soma:
            External somatic current.
        current_tuft:
            External apical-tuft current.

        Returns
        -------
        tuple of float
            Candidate state after one internal substep.
        """
        derivatives = self._derivatives(state, current_soma, current_tuft)
        return (
            state[0] + self.dt * derivatives[0],
            state[1] + self.dt * derivatives[1],
            state[2] + self.dt * derivatives[2],
            state[3] + self.dt * derivatives[3],
            state[4] + self.dt * derivatives[4],
            state[5] + self.dt * derivatives[5],
            state[6] + self.dt * derivatives[6],
            state[7] + self.dt * derivatives[7],
            state[8] + self.dt * derivatives[8],
        )

    @staticmethod
    def _validate_candidate(candidate: _State) -> _State:
        """Return a finite candidate with non-negative calcium or raise.

        Parameters
        ----------
        candidate:
            Candidate state in model state order.

        Returns
        -------
        tuple of float
            Candidate state with tuft calcium clipped to zero when needed.
        """
        if not all(math.isfinite(value) for value in candidate):
            raise FloatingPointError("Hay L5 candidate became non-finite")
        return (
            candidate[0],
            candidate[1],
            candidate[2],
            candidate[3],
            candidate[4],
            candidate[5],
            candidate[6],
            candidate[7],
            max(candidate[8], 0.0),
        )

    def step(self, current_soma: float, current_tuft: float = 0.0) -> int:
        """Advance the model by one public step and return a spike indicator.

        Parameters
        ----------
        current_soma:
            External current delivered to the somatic compartment.
        current_tuft:
            External current delivered to the apical-tuft compartment.

        Returns
        -------
        int
            ``1`` on an upward soma-threshold crossing, otherwise ``0``.
        """
        current_soma = self._finite_float("current_soma", current_soma)
        current_tuft = self._finite_float("current_tuft", current_tuft)
        self._validate_runtime_configuration()
        v_s_prev = self.v_s
        state = (
            self.v_s,
            self.h_na,
            self.n_k,
            self.v_t,
            self.m_ca,
            self.h_ca,
            self.m_ih,
            self.v_a,
            self.ca_a,
        )
        advance = self._euler_substep if self.integrator == "baseline_euler" else self._rk4_substep
        for _ in range(_N_SUBSTEPS):
            state = self._validate_candidate(advance(state, current_soma, current_tuft))

        (
            self.v_s,
            self.h_na,
            self.n_k,
            self.v_t,
            self.m_ca,
            self.h_ca,
            self.m_ih,
            self.v_a,
            self.ca_a,
        ) = state
        return int(self.v_s >= self.v_threshold and v_s_prev < self.v_threshold)

    def reset(self) -> None:
        """Restore voltage, gate, and calcium state to documented defaults."""
        self.v_s = -75.0
        self.h_na = 0.9
        self.n_k = 0.1
        self.v_t = -75.0
        self.m_ca = 0.0
        self.h_ca = 1.0
        self.m_ih = 0.0
        self.v_a = -75.0
        self.ca_a = 0.0001
