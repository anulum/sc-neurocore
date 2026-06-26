# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — De Schutter & Bower 1994 — cerebellar Purkinje cell

"""Compact conductance-based Purkinje-cell model after De Schutter & Bower."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_STATE_NAMES = ("v", "h_na", "n_k", "m_cap", "h_cap", "q_kca", "ca")
_PARAM_NAMES = (
    "g_na",
    "g_k",
    "g_cap",
    "g_kca",
    "g_l",
    "e_na",
    "e_k",
    "e_ca",
    "e_l",
    "ca_decay",
    "f_ca",
    "dt",
    "v_threshold",
)
_NON_NEGATIVE_PARAMS = ("g_na", "g_k", "g_cap", "g_kca", "g_l", "ca_decay", "f_ca")
_STRICTLY_POSITIVE_PARAMS = ("dt",)
_N_SUBSTEPS = 5


def _safe_exp(value: float) -> float:
    """Return ``exp(value)``, yielding ``+inf`` on overflow.

    Parameters
    ----------
    value:
        Exponent to evaluate.

    Returns
    -------
    float
        Exponential value, or ``math.inf`` if the double range overflows.
    """
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


@dataclass
class DeSchutterPurkinjeNeuron:
    """Single-compartment Purkinje-cell conductance model after De Schutter & Bower.

    The maintained Python model exposes the seven compact state variables
    ``(v, h_na, n_k, m_cap, h_cap, q_kca, ca)``. It is a compact point-neuron
    approximation; use the audit index before treating it as the full
    multi-compartment reconstruction from the original paper.

    The production update is candidate-first RK4 over all seven states. Five
    internal substeps are retained to preserve the existing model time base, but
    the public step commits only after every substep candidate is finite. The
    historical explicit Euler path remains available through
    ``integrator="baseline_euler"`` for regression comparisons.

    Reference: De Schutter, E. & Bower, J.M. (1994). J. Neurophysiol. 71:375–400.
    """

    v: float = -68.0
    h_na: float = 0.8
    n_k: float = 0.1
    m_cap: float = 0.0
    h_cap: float = 0.9
    q_kca: float = 0.0
    ca: float = 0.0001
    g_na: float = 125.0
    g_k: float = 10.0
    g_cap: float = 45.0
    g_kca: float = 35.0
    g_l: float = 0.5
    e_na: float = 45.0
    e_k: float = -85.0
    e_ca: float = 135.0
    e_l: float = -68.0
    ca_decay: float = 0.02
    f_ca: float = 0.00024
    dt: float = 0.01
    v_threshold: float = -20.0
    integrator: Literal["rk4", "baseline_euler"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in {"rk4", "baseline_euler"}:
            raise ValueError(f"Unsupported integrator for DeSchutterPurkinjeNeuron: {self.integrator}")
        self._validate_configuration()

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        """Return ``value`` as a finite float and reject booleans.

        Parameters
        ----------
        name:
            Field name used in errors.
        value:
            Candidate numeric value.

        Returns
        -------
        float
            Finite float value.
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
        """Coerce all state and parameter values to finite floats and enforce signs."""
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            setattr(self, name, self._finite_float(name, getattr(self, name)))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE_PARAMS:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.ca < 0.0:
            raise ValueError("ca must be non-negative")

    def _validate_runtime_configuration(self) -> None:
        """Re-check finite state and parameter signs before mutating state."""
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            self._finite_float(name, getattr(self, name))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE_PARAMS:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.ca < 0.0:
            raise ValueError("ca must be non-negative")

    def _derivatives(
        self,
        v: float,
        h_na: float,
        n_k: float,
        m_cap: float,
        h_cap: float,
        q_kca: float,
        ca: float,
        current: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Return seven-state derivatives from one consistent state.

        Parameters
        ----------
        v, h_na, n_k, m_cap, h_cap, q_kca, ca:
            Membrane voltage, five gates, and intracellular calcium.
        current:
            External current held constant across the RK4 stage.

        Returns
        -------
        tuple of float
            Derivatives in ``(v, h_na, n_k, m_cap, h_cap, q_kca, ca)`` order.
        """
        values = (v, h_na, n_k, m_cap, h_cap, q_kca, ca, current)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("De Schutter derivative input became non-finite")
        ca_eff = max(ca, 0.0)
        m_na_inf = 1.0 / (1.0 + _safe_exp(-(v + 35.0) / 7.5))
        h_na_inf = 1.0 / (1.0 + _safe_exp((v + 55.0) / 7.0))
        n_k_inf = 1.0 / (1.0 + _safe_exp(-(v + 30.0) / 15.0))
        m_cap_inf = 1.0 / (1.0 + _safe_exp(-(v + 19.0) / 5.5))
        h_cap_inf = 1.0 / (1.0 + _safe_exp((v + 48.0) / 7.0))
        q_kca_inf = ca_eff / (ca_eff + 0.0002)

        tau_h_na = 0.5 + 14.0 / (1.0 + _safe_exp((v + 40.0) / 12.0))
        tau_n_k = 1.0 + 11.0 / (1.0 + _safe_exp((v + 15.0) / 8.0))
        tau_m_cap = 0.3
        tau_h_cap = 45.0
        tau_q = 1.0

        d_h_na = (h_na_inf - h_na) / tau_h_na
        d_n_k = (n_k_inf - n_k) / tau_n_k
        d_m_cap = (m_cap_inf - m_cap) / tau_m_cap
        d_h_cap = (h_cap_inf - h_cap) / tau_h_cap
        d_q_kca = (q_kca_inf - q_kca) / tau_q

        i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v - self.e_na)
        i_k = self.g_k * n_k * n_k * n_k * n_k * (v - self.e_k)
        i_cap = self.g_cap * m_cap * m_cap * h_cap * (v - self.e_ca)
        i_kca = self.g_kca * q_kca * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)

        d_v = -i_na - i_k - i_cap - i_kca - i_l + current
        d_ca = -self.f_ca * i_cap - self.ca_decay * ca_eff
        derivatives = (d_v, d_h_na, d_n_k, d_m_cap, d_h_cap, d_q_kca, d_ca)
        if not all(math.isfinite(value) for value in derivatives):
            raise FloatingPointError("De Schutter derivative became non-finite")
        return derivatives

    def _rk4_substep(
        self,
        state: tuple[float, float, float, float, float, float, float],
        current: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Return one RK4 candidate for the seven-state vector.

        Parameters
        ----------
        state:
            State at substep start.
        current:
            External current held constant across all four stages.

        Returns
        -------
        tuple of float
            Candidate state after one internal substep.
        """
        dt = self.dt
        k1 = self._derivatives(*state, current)
        s2 = (
            state[0] + 0.5 * dt * k1[0],
            state[1] + 0.5 * dt * k1[1],
            state[2] + 0.5 * dt * k1[2],
            state[3] + 0.5 * dt * k1[3],
            state[4] + 0.5 * dt * k1[4],
            state[5] + 0.5 * dt * k1[5],
            state[6] + 0.5 * dt * k1[6],
        )
        k2 = self._derivatives(*s2, current)
        s3 = (
            state[0] + 0.5 * dt * k2[0],
            state[1] + 0.5 * dt * k2[1],
            state[2] + 0.5 * dt * k2[2],
            state[3] + 0.5 * dt * k2[3],
            state[4] + 0.5 * dt * k2[4],
            state[5] + 0.5 * dt * k2[5],
            state[6] + 0.5 * dt * k2[6],
        )
        k3 = self._derivatives(*s3, current)
        s4 = (
            state[0] + dt * k3[0],
            state[1] + dt * k3[1],
            state[2] + dt * k3[2],
            state[3] + dt * k3[3],
            state[4] + dt * k3[4],
            state[5] + dt * k3[5],
            state[6] + dt * k3[6],
        )
        k4 = self._derivatives(*s4, current)
        return (
            state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
            state[4] + dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0,
            state[5] + dt * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]) / 6.0,
            state[6] + dt * (k1[6] + 2.0 * k2[6] + 2.0 * k3[6] + k4[6]) / 6.0,
        )

    def _euler_substep(
        self,
        state: tuple[float, float, float, float, float, float, float],
        current: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Return one explicit Euler candidate for regression comparison.

        Parameters
        ----------
        state:
            State at substep start.
        current:
            External current.

        Returns
        -------
        tuple of float
            Candidate state after one internal substep.
        """
        derivatives = self._derivatives(*state, current)
        return (
            state[0] + self.dt * derivatives[0],
            state[1] + self.dt * derivatives[1],
            state[2] + self.dt * derivatives[2],
            state[3] + self.dt * derivatives[3],
            state[4] + self.dt * derivatives[4],
            state[5] + self.dt * derivatives[5],
            state[6] + self.dt * derivatives[6],
        )

    def _validate_candidate(
        self,
        candidate: tuple[float, float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float, float]:
        """Return a finite candidate with non-negative calcium or raise.

        Parameters
        ----------
        candidate:
            Candidate state.

        Returns
        -------
        tuple of float
            Candidate state with calcium clamped to zero if needed.
        """
        if not all(math.isfinite(value) for value in candidate):
            raise FloatingPointError("De Schutter candidate became non-finite")
        v, h_na, n_k, m_cap, h_cap, q_kca, ca = candidate
        return (v, h_na, n_k, m_cap, h_cap, q_kca, max(ca, 0.0))

    def step(self, current: float) -> int:
        """Advance the compact conductance model and return a spike indicator.

        Parameters
        ----------
        current:
            External current.

        Returns
        -------
        int
            ``1`` on threshold crossing, otherwise ``0``.
        """
        current = self._finite_float("current", current)
        self._validate_runtime_configuration()
        v_prev = self.v
        state = (self.v, self.h_na, self.n_k, self.m_cap, self.h_cap, self.q_kca, self.ca)
        advance = self._euler_substep if self.integrator == "baseline_euler" else self._rk4_substep
        for _ in range(_N_SUBSTEPS):
            state = self._validate_candidate(advance(state, current))

        self.v, self.h_na, self.n_k, self.m_cap, self.h_cap, self.q_kca, self.ca = state
        return int(self.v >= self.v_threshold and v_prev < self.v_threshold)

    def reset(self) -> None:
        """Restore voltage, gates, and calcium state to their defaults."""
        self.v = -68.0
        self.h_na = 0.8
        self.n_k = 0.1
        self.m_cap = 0.0
        self.h_cap = 0.9
        self.q_kca = 0.0
        self.ca = 0.0001
