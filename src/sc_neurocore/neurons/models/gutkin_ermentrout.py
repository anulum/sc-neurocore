# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gutkin & Ermentrout 1998 — persistent Na + K minimal

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _sigmoid(argument: float) -> float:
    if argument >= 0.0:
        scale = math.exp(-argument)
        return 1.0 / (1.0 + scale)
    scale = math.exp(argument)
    return scale / (1.0 + scale)


@dataclass
class GutkinErmentroutNeuron:
    """Gutkin-Ermentrout persistent-sodium conductance neuron.

    The model keeps voltage ``v`` and delayed-rectifier activation ``n`` as
    dynamic states. Persistent sodium activation is instantaneous through
    ``m_inf(v)``. The implementation advances the coupled ODE with a
    candidate-first fourth-order Runge-Kutta step and commits the candidate
    only when the complete numeric contract remains finite and biologically
    bounded.

    Parameters
    ----------
    v:
        Membrane voltage state in the inherited normalized millivolt scale.
    n:
        Delayed-rectifier potassium activation gate. Must remain in ``[0, 1]``.
    g_na:
        Persistent sodium conductance. Must be non-negative.
    g_k:
        Delayed-rectifier potassium conductance. Must be non-negative.
    g_l:
        Leak conductance. Must be non-negative.
    e_na:
        Sodium reversal potential.
    e_k:
        Potassium reversal potential.
    e_l:
        Leak reversal potential.
    dt:
        Integration step. Must be positive and finite.
    v_threshold:
        Upward voltage-crossing spike threshold.

    Raises
    ------
    ValueError
        If the initial parameters violate the finite-state, conductance, gate,
        or timestep contract.

    Notes
    -----
    The historical SC-NeuroCore surface uses an implicit unit membrane
    capacitance for this reduced model. Spike output is an event marker from
    the threshold crossing; voltage is not reset by this model.

    References
    ----------
    Gutkin, B. S., & Ermentrout, G. B. (1998). Dynamics of membrane
    excitability determine interspike interval variability: A link between
    spike generation mechanisms and cortical spike train statistics. Neural
    Computation, 10(5), 1047-1065.
    """

    v: float = -65.0
    n: float = 0.1
    g_na: float = 20.0
    g_k: float = 10.0
    g_l: float = 8.0
    e_na: float = 60.0
    e_k: float = -90.0
    e_l: float = -80.0
    dt: float = 0.05
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        """Validate the initial state and parameters."""

        if not self._valid_static_contract():
            raise ValueError("invalid Gutkin-Ermentrout initial state or parameters")

    def step(self, current: float) -> int:
        """Advance one RK4 step under constant external current.

        Parameters
        ----------
        current:
            External input current held constant during the integration step.

        Returns
        -------
        int
            ``1`` for an upward threshold crossing during the committed step,
            otherwise ``0``.

        Raises
        ------
        ValueError
            If the current, existing state, intermediate RK4 stage, or final
            candidate violates the numeric contract. The neuron state is left
            unchanged on failure.
        """

        if not _finite(float(current)) or not self._valid_static_contract():
            raise ValueError("invalid Gutkin-Ermentrout runtime state or current")
        candidate = self._rk4_candidate(float(current))
        if candidate is None:
            raise ValueError("invalid Gutkin-Ermentrout RK4 candidate")
        v_prev = self.v
        next_v, next_n = candidate
        self.v = next_v
        self.n = next_n
        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        """Restore the documented default voltage and potassium gate."""

        self.v = -65.0
        self.n = 0.1

    def _valid_static_contract(self) -> bool:
        return (
            _finite(self.v)
            and _finite(self.n)
            and 0.0 <= self.n <= 1.0
            and _finite(self.g_na)
            and self.g_na >= 0.0
            and _finite(self.g_k)
            and self.g_k >= 0.0
            and _finite(self.g_l)
            and self.g_l >= 0.0
            and _finite(self.e_na)
            and _finite(self.e_k)
            and _finite(self.e_l)
            and _finite(self.dt)
            and self.dt > 0.0
            and _finite(self.v_threshold)
        )

    def _m_inf(self, v: float) -> float:
        return _sigmoid((v + 20.0) / 15.0)

    def _n_inf(self, v: float) -> float:
        return _sigmoid((v + 25.0) / 5.0)

    def _rhs(self, v: float, n_gate: float, current: float) -> tuple[float, float] | None:
        if not (_finite(v) and _finite(n_gate) and 0.0 <= n_gate <= 1.0 and _finite(current)):
            return None
        m_inf = self._m_inf(v)
        n_inf = self._n_inf(v)
        i_na = self.g_na * m_inf * (v - self.e_na)
        i_k = self.g_k * n_gate * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)
        dv = -i_na - i_k - i_l + current
        dn = n_inf - n_gate
        if _finite(dv) and _finite(dn):
            return dv, dn
        return None

    def _rk4_candidate(self, current: float) -> tuple[float, float] | None:
        k1 = self._rhs(self.v, self.n, current)
        if k1 is None:
            return None
        k2 = self._rhs(self.v + 0.5 * self.dt * k1[0], self.n + 0.5 * self.dt * k1[1], current)
        if k2 is None:
            return None
        k3 = self._rhs(self.v + 0.5 * self.dt * k2[0], self.n + 0.5 * self.dt * k2[1], current)
        if k3 is None:
            return None
        k4 = self._rhs(self.v + self.dt * k3[0], self.n + self.dt * k3[1], current)
        if k4 is None:
            return None
        next_v = self.v + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        next_n = self.n + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        if _finite(next_v) and _finite(next_n) and 0.0 <= next_n <= 1.0:
            return next_v, next_n
        return None


# ── BURSTING MODELS ────────────────────────────────────────────────
