# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dendritic NMDA Neuron (Jahr & Stevens 1990)

"""Two-compartment neuron with voltage-dependent NMDA Mg2+ block.

Implements a soma-dendrite pair where the dendrite has NMDA receptors
with the classical Jahr & Stevens (1990) magnesium block:

    B(V) = 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))

The NMDA current is:

    I_NMDA = g_NMDA * glutamate * B(V_dend) * (V_dend - E_NMDA)

This enables coincidence detection: the dendrite only passes current
when both presynaptic glutamate AND postsynaptic depolarisation are present.

Dendrite dynamics:

    tau_d * dV_d/dt = -(V_d - E_L) + I_NMDA + g_c * (V_s - V_d)

Soma dynamics:

    tau_s * dV_s/dt = -(V_s - E_L) + I_ext + g_c * (V_d - V_s)

Reference: Jahr & Stevens (1990), Schiller et al. (2000).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_REST_POTENTIAL = -65.0
_State = tuple[float, float]
_STRICTLY_POSITIVE_PARAMS = ("tau_soma", "tau_dend", "dt")
_NON_NEGATIVE_PARAMS = ("g_nmda", "mg_conc", "g_coupling")
_FINITE_PARAMS = ("e_nmda", "theta", "v_soma", "v_dend")


@dataclass
class DendriticNMDANeuron:
    """Two-compartment neuron with NMDA Mg2+ block (Jahr & Stevens 1990).

    Parameters
    ----------
    g_nmda : float
        NMDA conductance. Default: 1.5.
    e_nmda : float
        NMDA reversal potential (mV). Default: 0.0.
    mg_conc : float
        Extracellular Mg2+ concentration (mM). Default: 1.0.
    g_coupling : float
        Soma-dendrite coupling conductance. Default: 0.5.
    tau_soma : float
        Soma time constant (ms). Default: 20.0.
    tau_dend : float
        Dendrite time constant (ms). Default: 50.0.
    theta : float
        Spike threshold (mV). Default: -50.0.
    dt : float
        Integration timestep (ms). Default: 0.1.
    integrator : {"rk4", "baseline_euler"}
        Numerical integration path. ``"rk4"`` is the production default;
        ``"baseline_euler"`` preserves the historical dendrite-first Euler
        update as an explicit comparison path.
    """

    g_nmda: float = 1.5
    e_nmda: float = 0.0
    mg_conc: float = 1.0
    g_coupling: float = 0.5
    tau_soma: float = 20.0
    tau_dend: float = 50.0
    theta: float = -50.0
    dt: float = 0.1

    v_soma: float = -65.0
    v_dend: float = -65.0
    integrator: Literal["rk4", "baseline_euler"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in ("rk4", "baseline_euler"):
            raise ValueError("integrator must be 'rk4' or 'baseline_euler'")
        self._validate_configuration()

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        """Return ``value`` as a finite float or raise ``ValueError``."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _validate_configuration(self) -> None:
        """Validate static parameters and mutable state before integration."""
        for name in _STRICTLY_POSITIVE_PARAMS:
            value = self._finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        for name in _NON_NEGATIVE_PARAMS:
            value = self._finite_float(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        for name in _FINITE_PARAMS:
            self._finite_float(name, getattr(self, name))

    def mg_block(self, v: float) -> float:
        """Mg2+ block factor: B(V) = 1/(1 + [Mg]/3.57 * exp(-0.062*V))."""
        voltage = self._finite_float("voltage", v)

        return self._mg_block_value(voltage)

    def _mg_block_value(self, voltage: float) -> float:
        exponent = -0.062 * voltage
        try:
            exp_term = math.exp(exponent)
        except OverflowError:
            exp_term = math.inf
        return 1.0 / (1.0 + (self.mg_conc / 3.57) * exp_term)

    def _derivatives(self, state: _State, i_soma: float, glutamate: float) -> _State:
        v_soma, v_dend = state
        b = self._mg_block_value(v_dend)
        i_nmda = self.g_nmda * glutamate * b * (v_dend - self.e_nmda)
        dv_soma = (-v_soma - 65.0 + i_soma + self.g_coupling * (v_dend - v_soma)) / self.tau_soma
        dv_dend = (-v_dend - 65.0 + i_nmda + self.g_coupling * (v_soma - v_dend)) / self.tau_dend
        return dv_soma, dv_dend

    def _rk4_substep(self, state: _State, i_soma: float, glutamate: float) -> _State:
        dt = self.dt
        k1_s, k1_d = self._derivatives(state, i_soma, glutamate)
        k2_s, k2_d = self._derivatives(
            (state[0] + 0.5 * dt * k1_s, state[1] + 0.5 * dt * k1_d),
            i_soma,
            glutamate,
        )
        k3_s, k3_d = self._derivatives(
            (state[0] + 0.5 * dt * k2_s, state[1] + 0.5 * dt * k2_d),
            i_soma,
            glutamate,
        )
        k4_s, k4_d = self._derivatives(
            (state[0] + dt * k3_s, state[1] + dt * k3_d),
            i_soma,
            glutamate,
        )
        return (
            state[0] + dt * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s) / 6.0,
            state[1] + dt * (k1_d + 2.0 * k2_d + 2.0 * k3_d + k4_d) / 6.0,
        )

    def _baseline_euler_substep(self, i_soma: float, glutamate: float) -> _State:
        b = self._mg_block_value(self.v_dend)
        i_nmda = self.g_nmda * glutamate * b * (self.v_dend - self.e_nmda)
        dv_dend = (
            -self.v_dend - 65.0 + i_nmda + self.g_coupling * (self.v_soma - self.v_dend)
        ) / self.tau_dend
        next_v_dend = self.v_dend + dv_dend * self.dt
        i_dend_to_soma = self.g_coupling * (next_v_dend - self.v_soma)
        dv_soma = (-self.v_soma - 65.0 + i_soma + i_dend_to_soma) / self.tau_soma
        next_v_soma = self.v_soma + dv_soma * self.dt
        return next_v_soma, next_v_dend

    def step(self, i_soma: float, glutamate: float) -> int:
        """Step with somatic input current and dendritic glutamate.

        Returns 1 if spike, 0 otherwise.
        """
        i_soma = self._finite_float("i_soma", i_soma)
        glutamate = self._finite_float("glutamate", glutamate)
        if glutamate < 0.0:
            raise ValueError("glutamate must be finite and non-negative")
        self._validate_configuration()

        if self.integrator == "baseline_euler":
            next_v_soma, next_v_dend = self._baseline_euler_substep(i_soma, glutamate)
        else:
            next_v_soma, next_v_dend = self._rk4_substep(
                (self.v_soma, self.v_dend), i_soma, glutamate
            )
        if not math.isfinite(next_v_soma) or not math.isfinite(next_v_dend):
            raise ValueError("candidate state must be finite")

        if next_v_soma >= self.theta:
            self.v_soma = _REST_POTENTIAL
            self.v_dend = next_v_dend
            return 1
        self.v_soma = next_v_soma
        self.v_dend = next_v_dend
        return 0

    def reset(self) -> None:
        """Reset state to resting potential."""
        self.v_soma = _REST_POTENTIAL
        self.v_dend = _REST_POTENTIAL
