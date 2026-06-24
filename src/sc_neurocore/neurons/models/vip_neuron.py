# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — VIP irregular-spiking interneuron (Porter 1998 + Kv4 A-current)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_STATE_NAMES = ("v", "h", "n", "a", "b")
_PARAM_NAMES = (
    "g_na",
    "g_k",
    "g_a",
    "g_l",
    "e_na",
    "e_k",
    "e_l",
    "c_m",
    "dt",
    "v_threshold",
)
_STRICTLY_POSITIVE_PARAMS = ("g_na", "g_k", "g_l", "c_m", "dt")
# ``g_a`` is finite-only so an A-current block experiment can set it to zero.
_NON_NEGATIVE_PARAMS = ("g_a",)
_N_SUBSTEPS = 4


@dataclass
class VIPNeuron:
    """VIP (vasoactive intestinal peptide) irregular-spiking interneuron.

    The membrane potential follows

    ``C dV/dt = -I_Na - I_K - I_A - I_L + I_ext``

    with transient sodium ``I_Na = g_Na m_inf^3 h (V - E_Na)`` (instantaneous
    activation), delayed-rectifier ``I_K = g_K n^4 (V - E_K)``, a transient A-type
    ``I_A = g_A a^3 b (V - E_K)`` (Kv4) whose slow inactivation produces the firing
    accommodation typical of the small high-input-resistance VIP soma, and an ohmic
    leak. Every gating variable relaxes through a sigmoidal steady state, so the
    right-hand side has no removable singularities.

    The production integrator is candidate-first RK4 over the five-state
    ``(V, h, n, a, b)`` system: each sub-step evaluates the full right-hand side
    from one consistent state, forms the RK4 candidate, and commits it only once
    finite. The historical hard-coded forward-Euler update — which staggered the
    gate and membrane increments against mismatched states — remains reachable only
    through the explicit ``integrator="baseline_euler"`` regression option.

    Reference: Porter, J.T., Cauli, B., Tsuzuki, K., Lambolez, B., Rossier, J. &
    Audinat, E. (1998). Selective excitation of subtypes of neocortical interneurons
    by nicotinic receptors. J. Neurosci. 19:5228–5235; Bhatt et al. (2019).
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    a: float = 0.0
    b: float = 0.9
    g_na: float = 35.0
    g_k: float = 6.0
    g_a: float = 8.0
    g_l: float = 0.01
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 0.5
    dt: float = 0.025
    v_threshold: float = -20.0
    integrator: Literal["rk4", "baseline_euler"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in {"rk4", "baseline_euler"}:
            raise ValueError(f"Unsupported integrator for VIPNeuron: {self.integrator}")
        self._validate_configuration()

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        """Return ``value`` as a finite float, raising ``ValueError`` otherwise.

        Parameters
        ----------
        name:
            Attribute name used in the error message.
        value:
            Candidate numeric value; booleans are rejected.

        Returns
        -------
        float
            The validated finite float.
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
        """Coerce every state and parameter to a finite float and enforce signs."""
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            setattr(self, name, self._finite_float(name, getattr(self, name)))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE_PARAMS:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def _validate_runtime_configuration(self) -> None:
        """Re-check finiteness and signs before a step mutates state."""
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            self._finite_float(name, getattr(self, name))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE_PARAMS:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def _derivatives(
        self, v: float, h: float, n: float, a: float, b: float, current: float
    ) -> tuple[float, float, float, float, float]:
        """Return ``(dV, dh, dn, da, db)`` of the five-state system at one state.

        Parameters
        ----------
        v, h, n, a, b:
            Membrane potential and the four gating variables.
        current:
            External stimulus current applied during the sub-step.

        Returns
        -------
        tuple of float
            The time derivatives in ``(V, h, n, a, b)`` order.
        """
        if not all(math.isfinite(value) for value in (v, h, n, a, b, current)):
            raise FloatingPointError("VIP derivative input became non-finite")
        m_inf = 1.0 / (1.0 + math.exp(-(v + 30.0) / 9.5))
        h_inf = 1.0 / (1.0 + math.exp((v + 53.0) / 7.0))
        tau_h = 0.37 + 2.78 / (1.0 + math.exp((v + 40.5) / 6.0))
        n_inf = 1.0 / (1.0 + math.exp(-(v + 30.0) / 10.0))
        tau_n = 0.37 + 1.85 / (1.0 + math.exp((v + 27.0) / 15.0))
        a_inf = 1.0 / (1.0 + math.exp(-(v + 50.0) / 20.0))
        b_inf = 1.0 / (1.0 + math.exp((v + 78.0) / 6.0))
        dh = (h_inf - h) / tau_h
        dn = (n_inf - n) / tau_n
        da = (a_inf - a) / 5.0
        db = (b_inf - b) / 50.0
        i_na = self.g_na * m_inf * m_inf * m_inf * h * (v - self.e_na)
        i_k = self.g_k * n * n * n * n * (v - self.e_k)
        i_a = self.g_a * a * a * a * b * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)
        dv = (-i_na - i_k - i_a - i_l + current) / self.c_m
        if not all(math.isfinite(value) for value in (dv, dh, dn, da, db)):
            raise FloatingPointError("VIP derivative became non-finite")
        return dv, dh, dn, da, db

    def _rk4_substep(
        self, state: tuple[float, float, float, float, float], current: float
    ) -> tuple[float, float, float, float, float]:
        """Return one classical RK4 increment of the five-state vector.

        Parameters
        ----------
        state:
            The ``(V, h, n, a, b)`` state at the start of the sub-step.
        current:
            External stimulus current held constant across the four RK4 stages.

        Returns
        -------
        tuple of float
            The advanced ``(V, h, n, a, b)`` candidate after one ``dt`` sub-step.
        """
        dt = self.dt
        v0, h0, n0, a0, b0 = state
        k1 = self._derivatives(v0, h0, n0, a0, b0, current)
        k2 = self._derivatives(
            v0 + 0.5 * dt * k1[0],
            h0 + 0.5 * dt * k1[1],
            n0 + 0.5 * dt * k1[2],
            a0 + 0.5 * dt * k1[3],
            b0 + 0.5 * dt * k1[4],
            current,
        )
        k3 = self._derivatives(
            v0 + 0.5 * dt * k2[0],
            h0 + 0.5 * dt * k2[1],
            n0 + 0.5 * dt * k2[2],
            a0 + 0.5 * dt * k2[3],
            b0 + 0.5 * dt * k2[4],
            current,
        )
        k4 = self._derivatives(
            v0 + dt * k3[0],
            h0 + dt * k3[1],
            n0 + dt * k3[2],
            a0 + dt * k3[3],
            b0 + dt * k3[4],
            current,
        )
        return (
            v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            h0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            n0 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            a0 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
            b0 + dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0,
        )

    def _euler_substep(
        self, state: tuple[float, float, float, float, float], current: float
    ) -> tuple[float, float, float, float, float]:
        """Return one forward-Euler increment of the five-state vector.

        Retained only for the explicit ``integrator="baseline_euler"`` regression
        path; unlike the historical implementation it still evaluates a single
        consistent right-hand side rather than staggering the gate and membrane
        updates.

        Parameters
        ----------
        state:
            The ``(V, h, n, a, b)`` state at the start of the sub-step.
        current:
            External stimulus current applied during the sub-step.

        Returns
        -------
        tuple of float
            The advanced ``(V, h, n, a, b)`` candidate after one ``dt`` sub-step.
        """
        dt = self.dt
        v0, h0, n0, a0, b0 = state
        dv, dh, dn, da, db = self._derivatives(v0, h0, n0, a0, b0, current)
        return (v0 + dt * dv, h0 + dt * dh, n0 + dt * dn, a0 + dt * da, b0 + dt * db)

    @staticmethod
    def _validate_candidate(
        candidate: tuple[float, float, float, float, float],
    ) -> tuple[float, float, float, float, float]:
        """Return the candidate state, raising if any component is non-finite.

        Parameters
        ----------
        candidate:
            The proposed ``(V, h, n, a, b)`` state from a sub-step.

        Returns
        -------
        tuple of float
            The validated candidate state.
        """
        for name, value in zip(_STATE_NAMES, candidate):
            if not math.isfinite(value):
                raise FloatingPointError(f"VIP {name} candidate became non-finite")
        return candidate

    def step(self, current: float = 0.0) -> int:
        """Advance the neuron by one ``4 * dt`` step and report a threshold crossing.

        Parameters
        ----------
        current:
            External stimulus current (µA/cm²) held constant across the step.

        Returns
        -------
        int
            ``1`` if ``V`` crossed ``v_threshold`` upward during the step, else ``0``.
        """
        current = self._finite_float("current", current)
        self._validate_runtime_configuration()
        v_prev = self.v
        state: tuple[float, float, float, float, float] = (
            self.v,
            self.h,
            self.n,
            self.a,
            self.b,
        )
        advance = self._euler_substep if self.integrator == "baseline_euler" else self._rk4_substep
        for _ in range(_N_SUBSTEPS):
            state = self._validate_candidate(advance(state, current))
        self.v, self.h, self.n, self.a, self.b = state
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        """Restore the resting membrane potential and gating defaults."""
        self.v = -65.0
        self.h, self.n, self.a, self.b = 0.8, 0.1, 0.0, 0.9
