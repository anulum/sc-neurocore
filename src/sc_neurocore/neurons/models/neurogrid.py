# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Boahen 2014 — Neurogrid subthreshold analog 2-compartment

"""Reduced Neurogrid two-compartment EIF neuron."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_STATE_NAMES = ("v_s", "v_d")
_PARAM_NAMES = (
    "tau_s",
    "tau_d",
    "g_c",
    "delta_t",
    "v_rest",
    "v_threshold",
    "v_peak",
    "v_reset",
    "dt",
)
_STRICTLY_POSITIVE_PARAMS = ("tau_s", "tau_d", "delta_t", "dt")
_NON_NEGATIVE_PARAMS = ("g_c",)
_State = tuple[float, float]


@dataclass
class NeuroGridNeuron:
    """Boahen-style reduced Neurogrid two-compartment analog neuron.

    The model couples a passive dendritic integrator to an EIF-like soma. The
    production path uses candidate-first RK4 for the continuous two-state flow
    and applies the discrete Neurogrid spike/reset rule once to the accepted
    public-step candidate. The historical dendrite-first explicit Euler update
    remains available through ``integrator="baseline_euler"`` for regression
    comparisons.

    Reference: Benjamin, B.V. et al. (2014). Proc. IEEE 102:699-716.
    """

    v_s: float = -65.0
    v_d: float = -65.0
    tau_s: float = 20.0
    tau_d: float = 50.0
    g_c: float = 0.5
    delta_t: float = 2.0
    v_rest: float = -65.0
    v_threshold: float = -50.0
    v_peak: float = 20.0
    v_reset: float = -65.0
    dt: float = 0.1
    integrator: Literal["rk4", "baseline_euler"] = "rk4"

    def __post_init__(self) -> None:
        """Validate the integrator choice and initial numeric configuration."""
        if self.integrator not in {"rk4", "baseline_euler"}:
            raise ValueError(f"Unsupported integrator for NeuroGridNeuron: {self.integrator}")
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
        """Coerce state/parameters to finite floats and enforce signs."""
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            setattr(self, name, self._finite_float(name, getattr(self, name)))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE_PARAMS:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def _validate_runtime_configuration(self) -> None:
        """Re-check state/parameter validity before mutating state."""
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            self._finite_float(name, getattr(self, name))
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE_PARAMS:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def _derivatives(self, state: _State, current: float) -> _State:
        """Return continuous two-state derivatives for the Neurogrid flow.

        Parameters
        ----------
        state:
            Candidate ``(v_s, v_d)`` state.
        current:
            Dendritic input current.

        Returns
        -------
        tuple of float
            ``(dv_s, dv_d)`` derivatives.
        """
        v_s, v_d = state
        if not math.isfinite(v_s) or not math.isfinite(v_d) or not math.isfinite(current):
            raise FloatingPointError("Neurogrid derivative input became non-finite")
        v_s_eff = min(v_s, self.v_peak)
        dv_d = (-(v_d - self.v_rest) + current - self.g_c * (v_d - v_s_eff)) / self.tau_d
        exp_arg = min((v_s_eff - self.v_threshold) / self.delta_t, 20.0)
        exp_term = self.delta_t * math.exp(exp_arg)
        dv_s = (-(v_s_eff - self.v_rest) + exp_term + self.g_c * (v_d - v_s_eff)) / self.tau_s
        derivatives = (dv_s, dv_d)
        if not all(math.isfinite(value) for value in derivatives):
            raise FloatingPointError("Neurogrid derivative became non-finite")
        return derivatives

    def _rk4_substep(self, state: _State, current: float) -> _State:
        """Return one RK4 candidate for the two-state continuous flow."""
        dt = self.dt
        k1 = self._derivatives(state, current)
        s2 = (state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1])
        k2 = self._derivatives(s2, current)
        s3 = (state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1])
        k3 = self._derivatives(s3, current)
        s4 = (state[0] + dt * k3[0], state[1] + dt * k3[1])
        k4 = self._derivatives(s4, current)
        return (
            state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        )

    def _euler_substep(self, state: _State, current: float) -> _State:
        """Return the historical dendrite-first Euler candidate."""
        v_s, v_d = state
        dv_d = (-(v_d - self.v_rest) + current - self.g_c * (v_d - v_s)) / self.tau_d
        v_d_next = v_d + dv_d * self.dt
        exp_arg = min((v_s - self.v_threshold) / self.delta_t, 20.0)
        exp_term = self.delta_t * math.exp(exp_arg)
        dv_s = (-(v_s - self.v_rest) + exp_term + self.g_c * (v_d_next - v_s)) / self.tau_s
        return (v_s + dv_s * self.dt, v_d_next)

    @staticmethod
    def _validate_candidate(candidate: _State) -> _State:
        """Return a finite candidate or raise."""
        if not all(math.isfinite(value) for value in candidate):
            raise FloatingPointError("Neurogrid candidate became non-finite")
        return candidate

    def step(self, current: float) -> int:
        """Advance the neuron by one public step and return a spike indicator.

        Parameters
        ----------
        current:
            Dendritic input current.

        Returns
        -------
        int
            ``1`` when the accepted soma candidate reaches `v_peak`, otherwise
            ``0``.
        """
        current = self._finite_float("current", current)
        self._validate_runtime_configuration()
        state = (self.v_s, self.v_d)
        advance = self._euler_substep if self.integrator == "baseline_euler" else self._rk4_substep
        next_v_s, next_v_d = self._validate_candidate(advance(state, current))
        spike = next_v_s >= self.v_peak
        self.v_s = self.v_reset if spike else next_v_s
        self.v_d = next_v_d
        return int(spike)

    def reset(self) -> None:
        """Restore soma and dendrite voltages to rest."""
        self.v_s = -65.0
        self.v_d = -65.0
