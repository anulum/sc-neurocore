# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — PV+ Fast-Spiking Interneuron (Wang-Buzsáki 1996 + Kv3.1)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_STATE_NAMES = ("v", "h", "n", "p")
_PARAM_NAMES = (
    "g_na",
    "g_k",
    "g_kv3",
    "g_l",
    "e_na",
    "e_k",
    "e_l",
    "c_m",
    "phi",
    "dt",
    "v_threshold",
)
_STRICTLY_POSITIVE_PARAMS = ("g_na", "g_k", "g_l", "c_m", "phi", "dt")
# ``g_kv3`` is finite-only so a Kv3-block experiment can set it to zero.
_NON_NEGATIVE_PARAMS = ("g_kv3",)


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    """Return ``a (v+vhalf) / (1 - exp(-(v+vhalf)/k))`` with the removable limit.

    The Wang-Buzsáki activation rates share the Hodgkin-Huxley ``x/(1-exp(-x/k))``
    form, which is finite at ``x = 0`` by L'Hôpital's rule but evaluates to ``0/0``
    numerically. The closed-form limit ``a * k`` (passed as ``fallback``) is returned
    within ``1e-7`` of the singularity, matching every backend kernel.

    Parameters
    ----------
    a:
        Rate prefactor.
    vhalf:
        Half-activation voltage offset.
    v:
        Membrane potential.
    k:
        Exponential slope.
    fallback:
        The L'Hôpital value ``a * k`` of the ratio at ``v + vhalf = 0``.

    Returns
    -------
    float
        The activation rate, using ``fallback`` within ``1e-7`` of the singularity.
    """
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


@dataclass
class PVFastSpikingNeuron:
    """PV+ (parvalbumin) fast-spiking interneuron (Wang-Buzsáki 1996 + Kv3.1).

    The membrane potential follows

    ``C dV/dt = -I_Na - I_K - I_Kv3 - I_L + I_ext``

    with transient sodium ``I_Na = g_Na m_inf^3 h (V - E_Na)`` (instantaneous
    activation), delayed-rectifier ``I_K = g_K n^4 (V - E_K)``, the fast Kv3.1
    current ``I_Kv3 = g_Kv3 p (V - E_K)`` that narrows the action potential and
    sustains >200 Hz firing without adaptation, and an ohmic leak. The ``h`` and
    ``n`` gates carry the Wang-Buzsáki temperature factor ``phi``.

    The production integrator is candidate-first RK4 over the four-state
    ``(V, h, n, p)`` system: each sub-step evaluates the full right-hand side from
    one consistent state, forms the RK4 candidate, and commits it only once finite.
    The historical hard-coded forward-Euler update — which staggered the gate and
    membrane increments against mismatched states — remains reachable only through
    the explicit ``integrator="baseline_euler"`` regression option.

    Reference: Wang, X.-J. & Buzsáki, G. (1996). Gamma oscillation by synaptic
    inhibition in a hippocampal interneuronal network model. J. Neurosci.
    16:6402–6413.
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    p: float = 0.0
    g_na: float = 35.0
    g_k: float = 9.0
    g_kv3: float = 5.0
    g_l: float = 0.1
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    dt: float = 0.01
    v_threshold: float = -20.0
    integrator: Literal["rk4", "baseline_euler"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in {"rk4", "baseline_euler"}:
            raise ValueError(f"Unsupported integrator for PVFastSpikingNeuron: {self.integrator}")
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

    def _n_substeps(self) -> int:
        """Return the number of fixed sub-steps that tile a 0.5 ms macro-step."""
        return max(1, int(0.5 / max(self.dt, 0.001)))

    def _derivatives(
        self, v: float, h: float, n: float, p: float, current: float
    ) -> tuple[float, float, float, float]:
        """Return ``(dV, dh, dn, dp)`` of the four-state system at one state.

        Parameters
        ----------
        v, h, n, p:
            Membrane potential and the three gating variables.
        current:
            External stimulus current applied during the sub-step.

        Returns
        -------
        tuple of float
            The time derivatives in ``(V, h, n, p)`` order.
        """
        if not all(math.isfinite(value) for value in (v, h, n, p, current)):
            raise FloatingPointError("PV-FS derivative input became non-finite")
        am = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
        bm = 4.0 * math.exp(-(v + 60.0) / 18.0)
        m_inf = am / (am + bm)
        ah = 0.07 * math.exp(-(v + 58.0) / 20.0)
        bh = 1.0 / (1.0 + math.exp(-(v + 28.0) / 10.0))
        an = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
        bn = 0.125 * math.exp(-(v + 44.0) / 80.0)
        p_inf = 1.0 / (1.0 + math.exp(-(v + 10.0) / 10.0))
        dh = self.phi * (ah * (1.0 - h) - bh * h)
        dn = self.phi * (an * (1.0 - n) - bn * n)
        dp = self.phi * (p_inf - p)
        i_na = self.g_na * m_inf * m_inf * m_inf * h * (v - self.e_na)
        i_k = self.g_k * n * n * n * n * (v - self.e_k)
        i_kv3 = self.g_kv3 * p * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)
        dv = (-i_na - i_k - i_kv3 - i_l + current) / self.c_m
        if not all(math.isfinite(value) for value in (dv, dh, dn, dp)):
            raise FloatingPointError("PV-FS derivative became non-finite")
        return dv, dh, dn, dp

    def _rk4_substep(
        self, state: tuple[float, float, float, float], current: float
    ) -> tuple[float, float, float, float]:
        """Return one classical RK4 increment of the four-state vector.

        Parameters
        ----------
        state:
            The ``(V, h, n, p)`` state at the start of the sub-step.
        current:
            External stimulus current held constant across the four RK4 stages.

        Returns
        -------
        tuple of float
            The advanced ``(V, h, n, p)`` candidate after one ``dt`` sub-step.
        """
        dt = self.dt
        v0, h0, n0, p0 = state
        k1 = self._derivatives(v0, h0, n0, p0, current)
        k2 = self._derivatives(
            v0 + 0.5 * dt * k1[0],
            h0 + 0.5 * dt * k1[1],
            n0 + 0.5 * dt * k1[2],
            p0 + 0.5 * dt * k1[3],
            current,
        )
        k3 = self._derivatives(
            v0 + 0.5 * dt * k2[0],
            h0 + 0.5 * dt * k2[1],
            n0 + 0.5 * dt * k2[2],
            p0 + 0.5 * dt * k2[3],
            current,
        )
        k4 = self._derivatives(
            v0 + dt * k3[0],
            h0 + dt * k3[1],
            n0 + dt * k3[2],
            p0 + dt * k3[3],
            current,
        )
        return (
            v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            h0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            n0 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            p0 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        )

    def _euler_substep(
        self, state: tuple[float, float, float, float], current: float
    ) -> tuple[float, float, float, float]:
        """Return one forward-Euler increment of the four-state vector.

        Retained only for the explicit ``integrator="baseline_euler"`` regression
        path; unlike the historical implementation it still evaluates a single
        consistent right-hand side rather than staggering the gate and membrane
        updates.

        Parameters
        ----------
        state:
            The ``(V, h, n, p)`` state at the start of the sub-step.
        current:
            External stimulus current applied during the sub-step.

        Returns
        -------
        tuple of float
            The advanced ``(V, h, n, p)`` candidate after one ``dt`` sub-step.
        """
        dt = self.dt
        v0, h0, n0, p0 = state
        dv, dh, dn, dp = self._derivatives(v0, h0, n0, p0, current)
        return (v0 + dt * dv, h0 + dt * dh, n0 + dt * dn, p0 + dt * dp)

    @staticmethod
    def _validate_candidate(
        candidate: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Return the candidate state, raising if any component is non-finite.

        Parameters
        ----------
        candidate:
            The proposed ``(V, h, n, p)`` state from a sub-step.

        Returns
        -------
        tuple of float
            The validated candidate state.
        """
        for name, value in zip(_STATE_NAMES, candidate):
            if not math.isfinite(value):
                raise FloatingPointError(f"PV-FS {name} candidate became non-finite")
        return candidate

    def step(self, current: float = 0.0) -> int:
        """Advance the neuron by one 0.5 ms step and report a threshold crossing.

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
        state: tuple[float, float, float, float] = (self.v, self.h, self.n, self.p)
        advance = self._euler_substep if self.integrator == "baseline_euler" else self._rk4_substep
        for _ in range(self._n_substeps()):
            state = self._validate_candidate(advance(state, current))
        self.v, self.h, self.n, self.p = state
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        """Restore the resting membrane potential and gating defaults."""
        self.v = -65.0
        self.h, self.n, self.p = 0.8, 0.1, 0.0
