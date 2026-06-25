# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Martinotti Cell

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_VT = -56.2
_STATE_NAMES = ("v", "m", "h", "n", "p", "s")
_PARAM_NAMES = (
    "g_na",
    "g_k",
    "g_m",
    "g_t",
    "g_l",
    "e_na",
    "e_k",
    "e_ca",
    "e_l",
    "c_m",
    "dt",
    "v_threshold",
)
_STRICTLY_POSITIVE_PARAMS = ("c_m", "dt")
# Conductances may be set to zero for current-block experiments.
_NON_NEGATIVE_PARAMS = ("g_na", "g_k", "g_m", "g_t", "g_l")
_N_SUBSTEPS = 4


def _alpha_singular(numerator: float, slope: float, limit: float) -> float:
    """Return ``numerator / (exp(numerator/slope) - 1)`` with the removable limit.

    The Traub-Miles activation rates have the Hodgkin-Huxley ``x/(exp(±x/k)-1)``
    form, finite at ``x = 0`` by L'Hôpital's rule but numerically ``0/0``. Within
    ``1e-6`` of the singularity the closed-form limit (equal to ``slope``) is
    returned, matching the Rust/Julia/Go/Mojo kernels exactly rather than
    perturbing the denominator with an epsilon.

    Parameters
    ----------
    numerator:
        The shifted-voltage numerator ``dv - c`` of the rate expression.
    slope:
        The exponential slope (sign folded into ``numerator``).
    limit:
        The L'Hôpital value of the ratio at ``numerator = 0`` (equal to ``slope``).

    Returns
    -------
    float
        The rate ratio, using ``limit`` within ``1e-6`` of the singularity.
    """
    if abs(numerator) < 1e-6:
        return limit
    return numerator / (math.exp(numerator / slope) - 1.0)


@dataclass
class MartinottiNeuron:
    """Martinotti cell — adapting interneuron targeting layer 1 apical dendrites.

    Pospischil 2008 gating with fast Na⁺, delayed-rectifier K⁺, a very strong
    M-current (Kv7) that gives the pronounced spike-frequency adaptation
    characteristic of the type, a low-threshold T-type Ca²⁺ current for rebound,
    and an ohmic leak — the six-state ``(V, m, h, n, p, s)`` system (T-type
    activation ``m_T`` is instantaneous; unlike the SST cell there is no ``Ih``).

    The sodium and potassium activation rates follow the Traub-Miles
    ``x/(exp(±x/k)-1)`` form and use the L'Hôpital limit at the removable
    singularity. The β_m numerator is the published ``V - V_T - 40`` offset; an
    earlier revision shared the ``-17`` offset of α_h, which lowered β_m enough to
    drive the cell into depolarisation block (it fired two or three spikes then
    stuck near threshold for any stimulus). The corrected kinetics restore a
    monotone frequency-current relation.

    The production integrator is candidate-first RK4 over the six-state system:
    each sub-step evaluates the full right-hand side from one consistent state,
    forms the RK4 candidate, and commits it only once finite. The historical
    hard-coded forward-Euler update — which staggered the gate and membrane
    increments against mismatched states — remains reachable only through the
    explicit ``integrator="baseline_euler"`` regression option.

    Reference: Silberberg, G. & Markram, H. (2007). Disynaptic inhibition between
    neocortical pyramidal cells mediated by Martinotti cells. Neuron 53:735–746;
    Toledo-Rodriguez et al. (2005); Pospischil et al. (2008) Biol. Cybern. 99:427.
    """

    v: float = -65.0
    m: float = 0.02
    h: float = 0.8
    n: float = 0.2
    p: float = 0.0
    s: float = 0.9
    g_na: float = 40.0
    g_k: float = 5.0
    g_m: float = 0.25
    g_t: float = 0.01
    g_l: float = 0.05
    e_na: float = 50.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_l: float = -65.0
    c_m: float = 0.8
    dt: float = 0.025
    v_threshold: float = -20.0
    integrator: Literal["rk4", "baseline_euler"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in {"rk4", "baseline_euler"}:
            raise ValueError(f"Unsupported integrator for MartinottiNeuron: {self.integrator}")
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
        self,
        v: float,
        m: float,
        h: float,
        n: float,
        p: float,
        s: float,
        current: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Return ``(dV, dm, dh, dn, dp, ds)`` of the system at one state.

        Parameters
        ----------
        v, m, h, n, p, s:
            Membrane potential and the five gating variables.
        current:
            External stimulus current applied during the sub-step.

        Returns
        -------
        tuple of float
            The time derivatives in ``(V, m, h, n, p, s)`` order.
        """
        if not all(math.isfinite(value) for value in (v, m, h, n, p, s, current)):
            raise FloatingPointError("Martinotti derivative input became non-finite")
        dv_vt = v - _VT
        alpha_m = -0.32 * _alpha_singular(dv_vt - 13.0, -4.0, -4.0)
        beta_m = 0.28 * _alpha_singular(dv_vt - 40.0, 5.0, 5.0)
        alpha_h = 0.128 * math.exp(-(dv_vt - 17.0) / 18.0)
        beta_h = 4.0 / (1.0 + math.exp(-(dv_vt - 40.0) / 5.0))
        alpha_n = -0.032 * _alpha_singular(dv_vt - 15.0, -5.0, -5.0)
        beta_n = 0.5 * math.exp(-(dv_vt - 10.0) / 40.0)
        dm = alpha_m * (1.0 - m) - beta_m * m
        dh = alpha_h * (1.0 - h) - beta_h * h
        dn = alpha_n * (1.0 - n) - beta_n * n
        p_inf = 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0))
        tau_p = 400.0 / (3.3 * math.exp((v + 35.0) / 20.0) + math.exp(-(v + 35.0) / 20.0))
        dp = (p_inf - p) / tau_p
        m_t_inf = 1.0 / (1.0 + math.exp(-(v + 57.0) / 6.2))
        s_inf = 1.0 / (1.0 + math.exp((v + 81.0) / 4.0))
        tau_s = 30.0 + 200.0 / (1.0 + math.exp((v + 70.0) / 5.0))
        ds = (s_inf - s) / tau_s
        i_na = self.g_na * m * m * m * h * (v - self.e_na)
        i_k = self.g_k * n * n * n * n * (v - self.e_k)
        i_m = self.g_m * p * (v - self.e_k)
        i_t = self.g_t * m_t_inf * m_t_inf * s * (v - self.e_ca)
        i_l = self.g_l * (v - self.e_l)
        dv = (-i_na - i_k - i_m - i_t - i_l + current) / self.c_m
        if not all(math.isfinite(value) for value in (dv, dm, dh, dn, dp, ds)):
            raise FloatingPointError("Martinotti derivative became non-finite")
        return dv, dm, dh, dn, dp, ds

    def _rk4_substep(
        self,
        state: tuple[float, float, float, float, float, float],
        current: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Return one classical RK4 increment of the six-state vector.

        Parameters
        ----------
        state:
            The ``(V, m, h, n, p, s)`` state at the start of the sub-step.
        current:
            External stimulus current held constant across the four RK4 stages.

        Returns
        -------
        tuple of float
            The advanced ``(V, m, h, n, p, s)`` candidate after one ``dt`` sub-step.
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
        )
        k2 = self._derivatives(*s2, current)
        s3 = (
            state[0] + 0.5 * dt * k2[0],
            state[1] + 0.5 * dt * k2[1],
            state[2] + 0.5 * dt * k2[2],
            state[3] + 0.5 * dt * k2[3],
            state[4] + 0.5 * dt * k2[4],
            state[5] + 0.5 * dt * k2[5],
        )
        k3 = self._derivatives(*s3, current)
        s4 = (
            state[0] + dt * k3[0],
            state[1] + dt * k3[1],
            state[2] + dt * k3[2],
            state[3] + dt * k3[3],
            state[4] + dt * k3[4],
            state[5] + dt * k3[5],
        )
        k4 = self._derivatives(*s4, current)
        return (
            state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
            state[4] + dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0,
            state[5] + dt * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]) / 6.0,
        )

    def _euler_substep(
        self,
        state: tuple[float, float, float, float, float, float],
        current: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Return one forward-Euler increment of the six-state vector.

        Retained only for the explicit ``integrator="baseline_euler"`` regression
        path; unlike the historical implementation it evaluates a single consistent
        right-hand side rather than staggering the gate and membrane updates.

        Parameters
        ----------
        state:
            The ``(V, m, h, n, p, s)`` state at the start of the sub-step.
        current:
            External stimulus current applied during the sub-step.

        Returns
        -------
        tuple of float
            The advanced ``(V, m, h, n, p, s)`` candidate after one ``dt`` sub-step.
        """
        dt = self.dt
        k = self._derivatives(*state, current)
        return (
            state[0] + dt * k[0],
            state[1] + dt * k[1],
            state[2] + dt * k[2],
            state[3] + dt * k[3],
            state[4] + dt * k[4],
            state[5] + dt * k[5],
        )

    @staticmethod
    def _validate_candidate(
        candidate: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        """Return the candidate state, raising if any component is non-finite.

        Parameters
        ----------
        candidate:
            The proposed ``(V, m, h, n, p, s)`` state from a sub-step.

        Returns
        -------
        tuple of float
            The validated candidate state.
        """
        for name, value in zip(_STATE_NAMES, candidate):
            if not math.isfinite(value):
                raise FloatingPointError(f"Martinotti {name} candidate became non-finite")
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
        state: tuple[float, float, float, float, float, float] = (
            self.v,
            self.m,
            self.h,
            self.n,
            self.p,
            self.s,
        )
        advance = self._euler_substep if self.integrator == "baseline_euler" else self._rk4_substep
        for _ in range(_N_SUBSTEPS):
            state = self._validate_candidate(advance(state, current))
        self.v, self.m, self.h, self.n, self.p, self.s = state
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        """Restore the resting potential and gating defaults."""
        self.v = -65.0
        self.m = 0.02
        self.h = 0.8
        self.n = 0.2
        self.p = 0.0
        self.s = 0.9
