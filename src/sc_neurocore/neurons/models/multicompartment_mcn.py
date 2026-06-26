# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multicompartment MCN Neuron (Spiking-WM, PNAS 2025)

"""Multi-compartment neuron matching the Spiking-WM architecture.

Dual-dendrite model with basal and apical compartments. The apical dendrite
gates how strongly basal information influences the soma, enabling
nonlinear integration for long-term temporal memory in RL tasks.

Exact equations from arXiv:2503.00713 (Spiking-WM, PNAS 2025):

    tau_b * dV_b/dt = -V_b + x_b                                  (basal)
    tau_a * dV_a/dt = -V_a + x_a                                  (apical)
    tau   * dU/dt   = -U + sigma(V_a) * [g_B/g_L * (V_b - U) + I]  (soma)
    S[t] = Theta(U[t] - V_th)                                     (spike)
    U[t] <- U[t] * (1 - S[t])                                     (soft reset)

Default parameters from Table II: tau = tau_a = tau_b = 2.0,
g_B/g_L = 1.0, beta = 1.0 (sigmoid steepness), V_th = 1.0.

Reference: Brain-Cog-Lab, arXiv:2503.00713, PNAS 2025.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_STATE_NAMES = ("u", "v_basal", "v_apical")
_PARAM_NAMES = ("tau", "tau_b", "tau_a", "g_ratio", "beta", "v_th", "dt")
_STRICTLY_POSITIVE_PARAMS = ("tau", "tau_b", "tau_a", "beta", "v_th", "dt")
_NON_NEGATIVE_PARAMS = ("g_ratio",)


def _safe_exp(value: float) -> float:
    """Return ``exp(value)``, yielding ``+inf`` on overflow instead of raising.

    Parameters
    ----------
    value:
        The exponent.

    Returns
    -------
    float
        ``exp(value)``, or ``math.inf`` if the exponent overflows binary64.
    """
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


@dataclass
class MulticompartmentMCNNeuron:
    """Multi-compartment neuron (Spiking-WM, PNAS 2025).

    The production update is candidate-first RK4 over the three coupled state
    variables ``(u, v_basal, v_apical)``. Basal, apical, and somatic drives are
    held constant across each RK4 stage; the stage apical voltage gates the
    stage basal-to-soma coupling, so all derivatives are evaluated from one
    consistent state. The historical explicit Euler path remains available only
    through ``integrator="baseline_euler"`` for regression comparisons.

    Parameters
    ----------
    tau : float
        Soma time constant. Default: 2.0.
    tau_b : float
        Basal dendrite time constant. Default: 2.0.
    tau_a : float
        Apical dendrite time constant. Default: 2.0.
    g_ratio : float
        Basal-to-soma conductance ratio (g_B/g_L). Default: 1.0.
    beta : float
        Sigmoid steepness for apical gating. Default: 1.0.
    v_th : float
        Spike threshold. Default: 1.0.
    dt : float
        Integration timestep. Default: 1.0.
    integrator : {"rk4", "baseline_euler"}
        Numerical integration path. Default: "rk4".
    """

    tau: float = 2.0
    tau_b: float = 2.0
    tau_a: float = 2.0
    g_ratio: float = 1.0
    beta: float = 1.0
    v_th: float = 1.0
    dt: float = 1.0
    integrator: Literal["rk4", "baseline_euler"] = "rk4"

    u: float = 0.0
    v_basal: float = 0.0
    v_apical: float = 0.0

    def __post_init__(self) -> None:
        if self.integrator not in {"rk4", "baseline_euler"}:
            raise ValueError(f"Unsupported integrator for MulticompartmentMCNNeuron: {self.integrator}")
        self._validate_configuration()

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        """Return ``value`` as a finite float, rejecting booleans.

        Parameters
        ----------
        name:
            Attribute name used in the error message.
        value:
            Candidate numeric value.

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
        """Coerce every state and parameter to finite floats and enforce signs."""
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            setattr(self, name, self._finite_float(name, getattr(self, name)))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE_PARAMS:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

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

    def _sigma(self, x: float) -> float:
        """Return the apical sigmoid gate ``1 / (1 + exp(-beta*x))``.

        Parameters
        ----------
        x:
            Apical membrane potential used by the gate.

        Returns
        -------
        float
            Gate value in ``[0, 1]`` for finite inputs.
        """
        x = self._finite_float("x", x)
        return 1.0 / (1.0 + _safe_exp(-self.beta * x))

    def _derivatives(
        self,
        u: float,
        v_basal: float,
        v_apical: float,
        x_basal: float,
        x_apical: float,
        i_soma: float,
    ) -> tuple[float, float, float]:
        """Return ``(du, dv_basal, dv_apical)`` from one consistent state.

        Parameters
        ----------
        u, v_basal, v_apical:
            Soma, basal dendrite, and apical dendrite state.
        x_basal, x_apical, i_soma:
            Basal, apical, and direct somatic drives held constant across the
            RK4 stage.

        Returns
        -------
        tuple of float
            Derivatives in ``(u, v_basal, v_apical)`` order.
        """
        values = (u, v_basal, v_apical, x_basal, x_apical, i_soma)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Multicompartment MCN derivative input became non-finite")
        gate = self._sigma(v_apical)
        du = (-u + gate * (self.g_ratio * (v_basal - u) + i_soma)) / self.tau
        dv_basal = (-v_basal + x_basal) / self.tau_b
        dv_apical = (-v_apical + x_apical) / self.tau_a
        derivatives = (du, dv_basal, dv_apical)
        if not all(math.isfinite(value) for value in derivatives):
            raise FloatingPointError("Multicompartment MCN derivative became non-finite")
        return derivatives

    def _rk4_substep(
        self,
        state: tuple[float, float, float],
        x_basal: float,
        x_apical: float,
        i_soma: float,
    ) -> tuple[float, float, float]:
        """Return one classical RK4 increment of ``(u, v_basal, v_apical)``.

        Parameters
        ----------
        state:
            State tuple at sub-step start.
        x_basal, x_apical, i_soma:
            Drives held constant across the four RK4 stages.

        Returns
        -------
        tuple of float
            Candidate next state.
        """
        dt = self.dt
        k1 = self._derivatives(*state, x_basal, x_apical, i_soma)
        k2_state = (
            state[0] + 0.5 * dt * k1[0],
            state[1] + 0.5 * dt * k1[1],
            state[2] + 0.5 * dt * k1[2],
        )
        k2 = self._derivatives(*k2_state, x_basal, x_apical, i_soma)
        k3_state = (
            state[0] + 0.5 * dt * k2[0],
            state[1] + 0.5 * dt * k2[1],
            state[2] + 0.5 * dt * k2[2],
        )
        k3 = self._derivatives(*k3_state, x_basal, x_apical, i_soma)
        k4_state = (
            state[0] + dt * k3[0],
            state[1] + dt * k3[1],
            state[2] + dt * k3[2],
        )
        k4 = self._derivatives(*k4_state, x_basal, x_apical, i_soma)
        return (
            state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        )

    def _euler_substep(
        self,
        state: tuple[float, float, float],
        x_basal: float,
        x_apical: float,
        i_soma: float,
    ) -> tuple[float, float, float]:
        """Return one explicit Euler increment for baseline comparisons.

        Parameters
        ----------
        state:
            State tuple at sub-step start.
        x_basal, x_apical, i_soma:
            Drives held constant for the step.

        Returns
        -------
        tuple of float
            Candidate next state.
        """
        derivatives = self._derivatives(*state, x_basal, x_apical, i_soma)
        return (
            state[0] + self.dt * derivatives[0],
            state[1] + self.dt * derivatives[1],
            state[2] + self.dt * derivatives[2],
        )

    def _validate_candidate(self, candidate: tuple[float, float, float]) -> tuple[float, float, float]:
        """Return a finite candidate state or raise before commit.

        Parameters
        ----------
        candidate:
            Candidate ``(u, v_basal, v_apical)`` state.

        Returns
        -------
        tuple of float
            The validated candidate.
        """
        if not all(math.isfinite(value) for value in candidate):
            raise FloatingPointError("Multicompartment MCN candidate became non-finite")
        return candidate

    def step_compartments(
        self,
        x_basal: float,
        x_apical: float,
        i_soma: float,
    ) -> int:
        """Advance one step with basal, apical, and direct somatic drives.

        Parameters
        ----------
        x_basal:
            Basal dendrite drive.
        x_apical:
            Apical dendrite drive.
        i_soma:
            Direct somatic drive.

        Returns
        -------
        int
            ``1`` if the candidate soma crosses threshold, otherwise ``0``.
        """
        x_basal = self._finite_float("x_basal", x_basal)
        x_apical = self._finite_float("x_apical", x_apical)
        i_soma = self._finite_float("i_soma", i_soma)
        self._validate_runtime_configuration()
        state = (self.u, self.v_basal, self.v_apical)
        advance = self._euler_substep if self.integrator == "baseline_euler" else self._rk4_substep
        next_u, next_v_basal, next_v_apical = self._validate_candidate(
            advance(state, x_basal, x_apical, i_soma)
        )

        spike = int(next_u >= self.v_th)
        self.u = 0.0 if spike else next_u
        self.v_basal = next_v_basal
        self.v_apical = next_v_apical
        if spike:
            self.u = 0.0
        return spike

    def step(self, current: float) -> int:
        """Advance one step with ``current`` delivered to the basal dendrite.

        Parameters
        ----------
        current:
            Basal input current.

        Returns
        -------
        int
            ``1`` if a spike occurs, otherwise ``0``.
        """
        current = self._finite_float("current", current)
        return self.step_compartments(current, 0.0, 0.0)

    def reset(self) -> None:
        """Reset soma, basal, and apical state to initial conditions."""
        self.u = 0.0
        self.v_basal = 0.0
        self.v_apical = 0.0
