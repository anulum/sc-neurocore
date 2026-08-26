# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — preserved six-state thalamocortical oscillator

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_STATE_NAMES = ("v", "h_na", "n_k", "m_h", "h_t", "na_i")
_PARAM_NAMES = (
    "g_na",
    "g_k",
    "g_h",
    "g_t",
    "g_kna",
    "g_l",
    "e_na",
    "e_k",
    "e_h",
    "e_ca",
    "e_l",
    "na_pump_max",
    "na_eq",
    "dt",
    "v_threshold",
)
# dt is the timestep; na_eq is a Michaelis constant that sits in a denominator
# with the non-negative na_i, so it must be strictly positive.
_STRICTLY_POSITIVE_PARAMS = ("dt", "na_eq")
# Conductances and the pump rate may be zeroed for channel-block experiments.
_NON_NEGATIVE_PARAMS = ("g_na", "g_k", "g_h", "g_t", "g_kna", "g_l", "na_pump_max")
_N_SUBSTEPS = 1


def _safe_exp(value: float) -> float:
    """Return ``exp(value)``, yielding ``+inf`` on overflow instead of raising.

    Python's ``math.exp`` raises ``OverflowError`` once the argument exceeds
    roughly ``709``, whereas the libm ``exp`` of the Rust, Julia, Go, and Mojo
    backends (and NumPy) return ``+inf``. Matching that ``+inf`` behaviour keeps
    the candidate-first finite check — rather than a Python-only exception type —
    the single fail-closed gate when an out-of-range stimulus blows the
    trajectory up.

    Parameters
    ----------
    value:
        The exponent.

    Returns
    -------
    float
        ``exp(value)``, or ``math.inf`` if that overflows the double range.
    """
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


@dataclass
class SCSixStateThalamocorticalNeuron:
    """Preserved SC six-state thalamocortical oscillator.

    A conductance-based cell with fast Na⁺ (instantaneous ``m_∞``, dynamic
    inactivation ``h_na``), delayed-rectifier K⁺ (``n_k``), a hyperpolarisation-
    activated ``Ih`` (``m_h``), a low-threshold T-type Ca²⁺ current (instantaneous
    ``m_T``, dynamic inactivation ``h_t``), a sodium-dependent K⁺ current
    (``I_KNa`` gated by intracellular ``na_i`` through a Hill function), and an
    ohmic leak — the six-state ``(V, h_na, n_k, m_h, h_t, na_i)`` system. The
    intracellular sodium accumulates from the sodium current and is removed by a
    saturating Na/K pump.

    The production integrator is candidate-first RK4 over the six-state system:
    each sub-step evaluates the full right-hand side from one consistent state,
    forms the RK4 candidate, and commits it only once finite. The historical
    hard-coded forward-Euler update — which advanced the gates from the old
    voltage and then the voltage and sodium from the freshly updated gates,
    mixing inconsistent states — remains reachable only through the explicit
    ``integrator="baseline_euler"`` regression option.

    The sodium concentration is clamped to be non-negative once per committed
    step, matching the preserved SC recurrence. The conductance powers use explicit
    multiplication and the ``I_KNa`` Hill exponent ``3.5`` is evaluated as
    ``b·b·b·sqrt(b)`` (an IEEE-754 exact decomposition of ``b**3.5``) so the
    Rust, Julia, Go, and Mojo kernels reproduce the trajectory bit-for-bit
    rather than depending on a per-platform ``pow`` implementation.

    This recurrence is retained for compatibility with earlier SC-NeuroCore
    releases. It is deliberately not attributed to Hill and Tononi (2005), whose
    source model is a hybrid integrate-and-fire system with a dynamic threshold.
    """

    v: float = -65.0
    h_na: float = 0.6
    n_k: float = 0.3
    m_h: float = 0.0
    h_t: float = 0.9
    na_i: float = 5.0  # mM intracellular Na⁺
    g_na: float = 50.0
    g_k: float = 5.0
    g_h: float = 1.0
    g_t: float = 3.0
    g_kna: float = 1.33
    g_l: float = 0.02
    e_na: float = 50.0
    e_k: float = -90.0
    e_h: float = -43.0
    e_ca: float = 120.0
    e_l: float = -70.0
    na_pump_max: float = 20.0  # mM/s, Na/K pump rate
    na_eq: float = 9.5  # mM, Na pump half-saturation
    dt: float = 0.05
    v_threshold: float = -20.0
    integrator: Literal["rk4", "baseline_euler"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in {"rk4", "baseline_euler"}:
            raise ValueError(
                f"Unsupported integrator for SCSixStateThalamocorticalNeuron: {self.integrator}"
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

    def _derivatives(
        self,
        v: float,
        h_na: float,
        n_k: float,
        m_h: float,
        h_t: float,
        na_i: float,
        current: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Return ``(dV, dh_na, dn_k, dm_h, dh_t, dna_i)`` at one consistent state.

        The sodium and T-type activations are instantaneous, so they are
        recomputed from ``v`` at every stage. The conductance powers use explicit
        multiplication and the ``I_KNa`` Hill term uses the ``b·b·b·sqrt(b)``
        decomposition of ``b**3.5``.

        Parameters
        ----------
        v, h_na, n_k, m_h, h_t, na_i:
            Membrane potential, the four dynamic gates, and intracellular sodium.
        current:
            External stimulus current held constant across the RK4 stage.

        Returns
        -------
        tuple of float
            The time derivatives in ``(V, h_na, n_k, m_h, h_t, na_i)`` order.
        """
        if not all(math.isfinite(value) for value in (v, h_na, n_k, m_h, h_t, na_i, current)):
            raise FloatingPointError(
                "SC six-state thalamocortical derivative input became non-finite"
            )
        m_na_inf = 1.0 / (1.0 + _safe_exp(-(v + 38.0) / 10.0))
        h_na_inf = 1.0 / (1.0 + _safe_exp((v + 43.0) / 6.0))
        n_k_inf = 1.0 / (1.0 + _safe_exp(-(v + 27.0) / 11.5))
        m_h_inf = 1.0 / (1.0 + _safe_exp((v + 75.0) / 5.5))
        m_t_inf = 1.0 / (1.0 + _safe_exp(-(v + 59.0) / 6.2))
        h_t_inf = 1.0 / (1.0 + _safe_exp((v + 83.0) / 4.0))
        hill_base = 38.7 / max(na_i, 0.01)
        w_kna = 0.37 / (1.0 + hill_base * hill_base * hill_base * math.sqrt(hill_base))
        tau_h_na = max(1.0 + 10.0 / (1.0 + _safe_exp((v + 40.0) / 10.0)), 0.1)
        z_n = (v + 50.0) / 25.0
        tau_n_k = max(5.0 + 47.0 * _safe_exp(-(z_n * z_n)), 0.1)
        tau_m_h = max(
            20.0 + 1000.0 / (_safe_exp((v + 71.5) / 14.2) + _safe_exp(-(v + 89.0) / 11.6)), 1.0
        )
        if v < -81.0:
            tau_h_t = max(
                30.8 + 211.4 * _safe_exp((v + 115.2) / 5.0) / (1.0 + _safe_exp((v + 86.0) / 3.2)),
                0.1,
            )
        else:
            tau_h_t = 10.0
        d_h_na = (h_na_inf - h_na) / tau_h_na
        d_n_k = (n_k_inf - n_k) / tau_n_k
        d_m_h = (m_h_inf - m_h) / tau_m_h
        d_h_t = (h_t_inf - h_t) / tau_h_t
        i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v - self.e_na)
        i_k = self.g_k * n_k * n_k * n_k * n_k * (v - self.e_k)
        i_h = self.g_h * m_h * (v - self.e_h)
        i_t = self.g_t * m_t_inf * m_t_inf * h_t * (v - self.e_ca)
        i_kna = self.g_kna * w_kna * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)
        d_v = -i_na - i_k - i_h - i_t - i_kna - i_l + current
        d_na_i = -0.001 * i_na - self.na_pump_max * (na_i / (na_i + self.na_eq))
        derivatives = (d_v, d_h_na, d_n_k, d_m_h, d_h_t, d_na_i)
        if not all(math.isfinite(value) for value in derivatives):
            raise FloatingPointError("SC six-state thalamocortical derivative became non-finite")
        return derivatives

    def _rk4_substep(
        self,
        state: tuple[float, float, float, float, float, float],
        current: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Return one classical RK4 increment of the six-state vector.

        Parameters
        ----------
        state:
            The ``(V, h_na, n_k, m_h, h_t, na_i)`` state at the sub-step start.
        current:
            External stimulus current held constant across the four RK4 stages.

        Returns
        -------
        tuple of float
            The advanced ``(V, h_na, n_k, m_h, h_t, na_i)`` candidate.
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
        path; unlike the historical implementation it evaluates a single
        consistent right-hand side rather than advancing the gates, the membrane,
        and the sodium concentration from mismatched states.

        Parameters
        ----------
        state:
            The ``(V, h_na, n_k, m_h, h_t, na_i)`` state at the sub-step start.
        current:
            External stimulus current applied during the sub-step.

        Returns
        -------
        tuple of float
            The advanced ``(V, h_na, n_k, m_h, h_t, na_i)`` candidate.
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
        """Return the candidate with ``na_i`` clamped to be non-negative.

        Parameters
        ----------
        candidate:
            The proposed ``(V, h_na, n_k, m_h, h_t, na_i)`` state from a sub-step.

        Returns
        -------
        tuple of float
            The validated candidate; raises if any component is non-finite.
        """
        for name, value in zip(_STATE_NAMES, candidate):
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"SC six-state thalamocortical {name} candidate became non-finite"
                )
        return (
            candidate[0],
            candidate[1],
            candidate[2],
            candidate[3],
            candidate[4],
            max(candidate[5], 0.0),
        )

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
        state: tuple[float, float, float, float, float, float] = (
            self.v,
            self.h_na,
            self.n_k,
            self.m_h,
            self.h_t,
            self.na_i,
        )
        advance = self._euler_substep if self.integrator == "baseline_euler" else self._rk4_substep
        for _ in range(_N_SUBSTEPS):
            state = self._validate_candidate(advance(state, current))
        self.v, self.h_na, self.n_k, self.m_h, self.h_t, self.na_i = state
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        """Restore the resting potential, gating defaults, and sodium baseline."""
        self.v, self.h_na, self.n_k, self.m_h, self.h_t = -65.0, 0.6, 0.3, 0.0, 0.9
        self.na_i = 5.0
