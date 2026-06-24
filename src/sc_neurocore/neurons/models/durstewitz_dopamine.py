# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Durstewitz, Seamans & Sejnowski 2000 — PFC neuron with D1 modulation

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_STATE_NAMES = ("v", "h_na", "n_k")
_PARAM_NAMES = (
    "g_na",
    "g_k",
    "g_nmda",
    "g_l",
    "e_na",
    "e_k",
    "e_nmda",
    "e_l",
    "mg",
    "d1_level",
    "g_nmda_scale",
    "g_k_scale",
    "v_shift_na",
    "dt",
    "v_threshold",
)
_STRICTLY_POSITIVE_PARAMS = ("dt",)
# Conductances and the Mg²⁺ concentration may be zero for channel-block experiments.
_NON_NEGATIVE_PARAMS = ("g_na", "g_k", "g_nmda", "g_l", "mg")
_N_SUBSTEPS = 1


@dataclass
class DurstewitzDopamineNeuron:
    """Durstewitz, Seamans & Sejnowski 2000 — PFC neuron with D1 modulation.

    A minimal Hodgkin-Huxley prefrontal pyramidal cell with fast Na⁺
    (instantaneous activation ``m_∞``, dynamic inactivation ``h_na``), a
    delayed-rectifier K⁺ (``n_k``), an NMDA current with the Jahr & Stevens
    (1990) Mg²⁺ block, and an ohmic leak — the three-state ``(V, h_na, n_k)``
    system. D1 agonism (``d1_level`` in ``[0, 1]``) enhances NMDA
    (``g_nmda_scale``), shifts the Na⁺ half-activation (``v_shift_na``), and
    enhances K⁺ (``g_k_scale``), stabilising persistent up-states.

    The production integrator is candidate-first RK4 over the three-state
    system: each sub-step evaluates the full right-hand side from one consistent
    state, forms the RK4 candidate, and commits it only once finite. The
    historical hard-coded forward-Euler update — which advanced the gates from
    the old voltage and then the voltage from the freshly updated gates, mixing
    two inconsistent states — remains reachable only through the explicit
    ``integrator="baseline_euler"`` regression option, which now evaluates a
    single consistent right-hand side.

    Reference: Durstewitz, D., Seamans, J. K. & Sejnowski, T. J. (2000).
    Dopamine-mediated stabilization of delay-period activity in a network model
    of prefrontal cortex. J. Neurophysiol. 83:1733–1750.
    """

    v: float = -65.0
    h_na: float = 0.7
    n_k: float = 0.2
    g_na: float = 45.0
    g_k: float = 18.0
    g_nmda: float = 0.5
    g_l: float = 0.02
    e_na: float = 55.0
    e_k: float = -80.0
    e_nmda: float = 0.0
    e_l: float = -65.0
    mg: float = 1.0  # mM extracellular Mg²⁺
    d1_level: float = 0.0  # 0..1 D1 agonism
    g_nmda_scale: float = 2.5  # max NMDA boost at d1=1
    g_k_scale: float = 1.5
    v_shift_na: float = -5.0  # mV, Na half-activation shift at d1=1
    dt: float = 0.05
    v_threshold: float = -20.0
    integrator: Literal["rk4", "baseline_euler"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in {"rk4", "baseline_euler"}:
            raise ValueError(
                f"Unsupported integrator for DurstewitzDopamineNeuron: {self.integrator}"
            )
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

    def mg_block(self, v: float) -> float:
        """Jahr & Stevens (1990) Mg²⁺ block ``B(V) = 1/(1 + [Mg]/3.57 · exp(-0.062 V))``.

        Parameters
        ----------
        v:
            Membrane potential (mV).

        Returns
        -------
        float
            The unblocked NMDA fraction in ``(0, 1]``.
        """
        v = self._finite_float("v", v)
        return 1.0 / (1.0 + self.mg / 3.57 * math.exp(-0.062 * v))

    def _derivatives(
        self, v: float, h_na: float, n_k: float, current: float
    ) -> tuple[float, float, float]:
        """Return ``(dV, dh_na, dn_k)`` of the system at one consistent state.

        The sodium activation ``m_∞`` is instantaneous, so it is recomputed from
        ``v`` at every stage. The conductance powers use explicit multiplication
        (``m·m·m``, ``n·n·n·n``) and the Mg²⁺ block keeps the
        ``[Mg]/3.57 · exp`` operand order so the Rust, Julia, Go, and Mojo
        kernels reproduce the trajectory bit-for-bit.

        Parameters
        ----------
        v, h_na, n_k:
            Membrane potential, Na⁺ inactivation, and K⁺ activation.
        current:
            External stimulus current held constant across the RK4 stage.

        Returns
        -------
        tuple of float
            The time derivatives in ``(V, h_na, n_k)`` order.
        """
        if not all(math.isfinite(value) for value in (v, h_na, n_k, current)):
            raise FloatingPointError("Durstewitz derivative input became non-finite")
        v_sh = self.d1_level * self.v_shift_na
        m_na_inf = 1.0 / (1.0 + math.exp(-(v + 30.0 + v_sh) / 9.5))
        h_na_inf = 1.0 / (1.0 + math.exp((v + 53.0) / 7.0))
        n_k_inf = 1.0 / (1.0 + math.exp(-(v + 30.0) / 10.0))
        tau_h = 0.5 + 14.0 / (1.0 + math.exp((v + 50.0) / 12.0))
        tau_n = 1.0 + 11.0 / (1.0 + math.exp((v + 40.0) / 10.0))
        d_h_na = (h_na_inf - h_na) / tau_h
        d_n_k = (n_k_inf - n_k) / tau_n
        mg_block = 1.0 / (1.0 + self.mg / 3.57 * math.exp(-0.062 * v))
        nmda_g = self.g_nmda * (1.0 + self.d1_level * (self.g_nmda_scale - 1.0))
        k_g = self.g_k * (1.0 + self.d1_level * (self.g_k_scale - 1.0))
        i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v - self.e_na)
        i_k = k_g * n_k * n_k * n_k * n_k * (v - self.e_k)
        i_nmda = nmda_g * mg_block * (v - self.e_nmda)
        i_l = self.g_l * (v - self.e_l)
        d_v = -i_na - i_k - i_nmda - i_l + current
        if not all(math.isfinite(value) for value in (d_v, d_h_na, d_n_k)):
            raise FloatingPointError("Durstewitz derivative became non-finite")
        return d_v, d_h_na, d_n_k

    def _rk4_substep(
        self, state: tuple[float, float, float], current: float
    ) -> tuple[float, float, float]:
        """Return one classical RK4 increment of the ``(V, h_na, n_k)`` vector.

        Parameters
        ----------
        state:
            The ``(V, h_na, n_k)`` state at the start of the sub-step.
        current:
            External stimulus current held constant across the four RK4 stages.

        Returns
        -------
        tuple of float
            The advanced ``(V, h_na, n_k)`` candidate after one ``dt`` sub-step.
        """
        dt = self.dt
        k1 = self._derivatives(*state, current)
        s2 = (
            state[0] + 0.5 * dt * k1[0],
            state[1] + 0.5 * dt * k1[1],
            state[2] + 0.5 * dt * k1[2],
        )
        k2 = self._derivatives(*s2, current)
        s3 = (
            state[0] + 0.5 * dt * k2[0],
            state[1] + 0.5 * dt * k2[1],
            state[2] + 0.5 * dt * k2[2],
        )
        k3 = self._derivatives(*s3, current)
        s4 = (
            state[0] + dt * k3[0],
            state[1] + dt * k3[1],
            state[2] + dt * k3[2],
        )
        k4 = self._derivatives(*s4, current)
        return (
            state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        )

    def _euler_substep(
        self, state: tuple[float, float, float], current: float
    ) -> tuple[float, float, float]:
        """Return one forward-Euler increment of the ``(V, h_na, n_k)`` vector.

        Retained only for the explicit ``integrator="baseline_euler"`` regression
        path; unlike the historical implementation it evaluates a single
        consistent right-hand side rather than advancing the gates and the
        membrane from mismatched states.

        Parameters
        ----------
        state:
            The ``(V, h_na, n_k)`` state at the start of the sub-step.
        current:
            External stimulus current applied during the sub-step.

        Returns
        -------
        tuple of float
            The advanced ``(V, h_na, n_k)`` candidate after one ``dt`` sub-step.
        """
        dt = self.dt
        k = self._derivatives(*state, current)
        return (
            state[0] + dt * k[0],
            state[1] + dt * k[1],
            state[2] + dt * k[2],
        )

    @staticmethod
    def _validate_candidate(
        candidate: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Return the candidate state, raising if any component is non-finite.

        Parameters
        ----------
        candidate:
            The proposed ``(V, h_na, n_k)`` state from a sub-step.

        Returns
        -------
        tuple of float
            The validated candidate state.
        """
        for name, value in zip(_STATE_NAMES, candidate):
            if not math.isfinite(value):
                raise FloatingPointError(f"Durstewitz {name} candidate became non-finite")
        return candidate

    def step(self, current: float = 0.0) -> int:
        """Advance the neuron by one ``dt`` step and report a threshold crossing.

        Parameters
        ----------
        current:
            External stimulus current held constant across the step.

        Returns
        -------
        int
            ``1`` if ``V`` crossed ``v_threshold`` upward during the step, else ``0``.
        """
        current = self._finite_float("current", current)
        self._validate_runtime_configuration()
        v_prev = self.v
        state: tuple[float, float, float] = (self.v, self.h_na, self.n_k)
        advance = self._euler_substep if self.integrator == "baseline_euler" else self._rk4_substep
        for _ in range(_N_SUBSTEPS):
            state = self._validate_candidate(advance(state, current))
        self.v, self.h_na, self.n_k = state
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        """Restore the resting potential and gating defaults."""
        self.v, self.h_na, self.n_k = -65.0, 0.7, 0.2
