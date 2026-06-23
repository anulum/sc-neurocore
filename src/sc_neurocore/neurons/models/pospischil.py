# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pospischil et al. 2008 — minimal HH for cortical/thalamic cells

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_STATE_NAMES = ("v", "m", "h", "n", "p")
_PARAM_NAMES = (
    "g_na",
    "g_kd",
    "g_m",
    "g_l",
    "e_na",
    "e_k",
    "e_l",
    "c_m",
    "vt",
    "dt",
    "v_threshold",
)
# ``g_m`` is legitimately zero for the fast-spiking variant, so it is finite-only.
_STRICTLY_POSITIVE_PARAMS = ("g_na", "g_kd", "g_l", "c_m", "dt")
_NON_NEGATIVE_PARAMS = ("g_m",)
_GATE_NAMES = ("m", "h", "n", "p")
_N_SUBSTEPS = 4


def _alpha_singular(numerator: float, slope: float, limit: float) -> float:
    """Return ``numerator / (exp(-numerator/slope) - 1)`` with the removable limit.

    The Traub-Miles activation rates have the Hodgkin-Huxley ``x/(exp(±x/k)-1)``
    form, which is finite at ``x = 0`` by L'Hôpital's rule but evaluates to ``0/0``
    numerically. Near the singularity the closed-form limit ``limit`` is returned;
    this matches the Rust/Julia/Go/Mojo kernels exactly rather than perturbing the
    denominator with an epsilon.

    Parameters
    ----------
    numerator:
        The shifted-voltage numerator ``dv - c`` of the rate expression.
    slope:
        The exponential slope ``k`` (with sign folded into ``numerator``).
    limit:
        The L'Hôpital value ``-slope`` of the ratio at ``numerator = 0``.

    Returns
    -------
    float
        The rate ratio, using ``limit`` within ``1e-6`` of the singularity.
    """
    if abs(numerator) < 1e-6:
        return limit
    return numerator / (math.exp(numerator / slope) - 1.0)


@dataclass
class PospischilNeuron:
    """Pospischil et al. 2008 — minimal Hodgkin-Huxley cortical/thalamic neuron.

    The membrane potential follows

    ``C dV/dt = -I_Na - I_Kd - I_M - I_L + I_ext``

    with transient sodium ``I_Na = g_Na m^3 h (V - E_Na)``, delayed-rectifier
    potassium ``I_Kd = g_Kd n^4 (V - E_K)``, the slow voltage-gated potassium
    adaptation current ``I_M = g_M p (V - E_K)``, and an ohmic leak. The Traub-Miles
    activation kinetics are written against the shifted potential ``V - V_T`` and the
    ``M`` current uses the Yamada ``p_inf``/``tau_p`` relaxation. The slow ``I_M``
    conductance selects the firing class: ``g_M = 0.07`` regular spiking (default),
    ``g_M = 0`` fast spiking, ``g_M = 0.03`` intrinsically bursting.

    The production integrator is candidate-first RK4 over the five-state
    ``(V, m, h, n, p)`` system: every sub-step evaluates the full right-hand side
    from one consistent state, forms the RK4 candidate, and commits it only once it
    is finite. The historical hard-coded forward-Euler update — which evaluated the
    gates and the membrane against mismatched states — remains reachable only through
    the explicit ``integrator="baseline_euler"`` option for regression comparison.

    Reference: Pospischil, M. et al. (2008). Minimal Hodgkin-Huxley type models for
    different classes of cortical and thalamic neurons. Biol. Cybern. 99:427–441.
    """

    v: float = -70.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.3
    p: float = 0.0
    g_na: float = 50.0
    g_kd: float = 5.0
    g_m: float = 0.07
    g_l: float = 0.1
    e_na: float = 50.0
    e_k: float = -90.0
    e_l: float = -70.0
    c_m: float = 1.0
    vt: float = -56.2
    dt: float = 0.025
    v_threshold: float = -20.0
    integrator: Literal["rk4", "baseline_euler"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in {"rk4", "baseline_euler"}:
            raise ValueError(f"Unsupported integrator for PospischilNeuron: {self.integrator}")
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
        self, v: float, m: float, h: float, n: float, p: float, current: float
    ) -> tuple[float, float, float, float, float]:
        """Return ``(dV, dm, dh, dn, dp)`` of the five-state system at one state.

        Parameters
        ----------
        v, m, h, n, p:
            Membrane potential and the four gating variables.
        current:
            External stimulus current applied during the sub-step.

        Returns
        -------
        tuple of float
            The time derivatives in ``(V, m, h, n, p)`` order.
        """
        if not all(math.isfinite(value) for value in (v, m, h, n, p, current)):
            raise FloatingPointError("Pospischil derivative input became non-finite")
        dv_vt = v - self.vt
        am = -0.32 * _alpha_singular(dv_vt - 13.0, -4.0, -4.0)
        bm = 0.28 * _alpha_singular(dv_vt - 40.0, 5.0, 5.0)
        ah = 0.128 * math.exp(-(dv_vt - 17.0) / 18.0)
        bh = 4.0 / (1.0 + math.exp(-(dv_vt - 40.0) / 5.0))
        an = -0.032 * _alpha_singular(dv_vt - 15.0, -5.0, -5.0)
        bn = 0.5 * math.exp(-(dv_vt - 10.0) / 40.0)
        p_inf = 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0))
        tau_p = 608.0 / (3.3 * math.exp((v + 35.0) / 20.0) + math.exp(-(v + 35.0) / 20.0))
        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n) - bn * n
        dp = (p_inf - p) / tau_p
        i_na = self.g_na * m * m * m * h * (v - self.e_na)
        i_kd = self.g_kd * n * n * n * n * (v - self.e_k)
        i_m = self.g_m * p * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)
        dv = (-i_na - i_kd - i_m - i_l + current) / self.c_m
        if not all(math.isfinite(value) for value in (dv, dm, dh, dn, dp)):
            raise FloatingPointError("Pospischil derivative became non-finite")
        return dv, dm, dh, dn, dp

    def _rk4_substep(
        self, state: tuple[float, float, float, float, float], current: float
    ) -> tuple[float, float, float, float, float]:
        """Return one classical RK4 increment of the five-state vector.

        Parameters
        ----------
        state:
            The ``(V, m, h, n, p)`` state at the start of the sub-step.
        current:
            External stimulus current held constant across the four RK4 stages.

        Returns
        -------
        tuple of float
            The advanced ``(V, m, h, n, p)`` candidate after one ``dt`` sub-step.
        """
        dt = self.dt
        v0, m0, h0, n0, p0 = state
        k1 = self._derivatives(v0, m0, h0, n0, p0, current)
        k2 = self._derivatives(
            v0 + 0.5 * dt * k1[0],
            m0 + 0.5 * dt * k1[1],
            h0 + 0.5 * dt * k1[2],
            n0 + 0.5 * dt * k1[3],
            p0 + 0.5 * dt * k1[4],
            current,
        )
        k3 = self._derivatives(
            v0 + 0.5 * dt * k2[0],
            m0 + 0.5 * dt * k2[1],
            h0 + 0.5 * dt * k2[2],
            n0 + 0.5 * dt * k2[3],
            p0 + 0.5 * dt * k2[4],
            current,
        )
        k4 = self._derivatives(
            v0 + dt * k3[0],
            m0 + dt * k3[1],
            h0 + dt * k3[2],
            n0 + dt * k3[3],
            p0 + dt * k3[4],
            current,
        )
        return (
            v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            m0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            h0 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            n0 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
            p0 + dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0,
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
            The ``(V, m, h, n, p)`` state at the start of the sub-step.
        current:
            External stimulus current applied during the sub-step.

        Returns
        -------
        tuple of float
            The advanced ``(V, m, h, n, p)`` candidate after one ``dt`` sub-step.
        """
        dt = self.dt
        v0, m0, h0, n0, p0 = state
        dv, dm, dh, dn, dp = self._derivatives(v0, m0, h0, n0, p0, current)
        return (
            v0 + dt * dv,
            m0 + dt * dm,
            h0 + dt * dh,
            n0 + dt * dn,
            p0 + dt * dp,
        )

    @staticmethod
    def _validate_candidate(
        candidate: tuple[float, float, float, float, float],
    ) -> tuple[float, float, float, float, float]:
        """Return the candidate state, raising if any component is non-finite.

        Parameters
        ----------
        candidate:
            The proposed ``(V, m, h, n, p)`` state from a sub-step.

        Returns
        -------
        tuple of float
            The validated candidate state.
        """
        for name, value in zip(_STATE_NAMES, candidate):
            if not math.isfinite(value):
                raise FloatingPointError(f"Pospischil {name} candidate became non-finite")
        return candidate

    def step(self, current: float) -> int:
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
            self.m,
            self.h,
            self.n,
            self.p,
        )
        advance = self._euler_substep if self.integrator == "baseline_euler" else self._rk4_substep
        for _ in range(_N_SUBSTEPS):
            state = self._validate_candidate(advance(state, current))
        self.v, self.m, self.h, self.n, self.p = state
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        """Restore the resting membrane potential and gating defaults."""
        self.v = -70.0
        self.m, self.h, self.n, self.p = 0.05, 0.6, 0.3, 0.0
