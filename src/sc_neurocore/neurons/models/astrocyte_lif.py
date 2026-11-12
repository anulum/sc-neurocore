# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Astrocyte-LIF Hybrid Neuron (Perea et al. 2009)

"""Astrocyte-LIF hybrid unit with calcium wave feedback.

Models the tripartite synapse: a glial astrocyte monitors extracellular
glutamate from a paired LIF neuron and provides slow homeostatic feedback
via calcium-dependent gliotransmitter release.

Equations:

    dCa/dt = -Ca/tau_ca + delta * S_pre(t)        (calcium rise on presynaptic spike)
    I_glio = g_glio * H(Ca - Ca_thresh)            (gliotransmitter release)
    dV/dt  = (-(V - E_L) + I_ext + I_glio)/tau_m  (LIF with glial feedback)

H is the Heaviside step function. Gliotransmitter release occurs when
calcium exceeds threshold, providing slow excitatory feedback.

Reference: Perea, Navarrete & Araque, "Tripartite synapses" (2009).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _positive(name: str, value: float) -> None:
    _finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _non_negative(name: str, value: float) -> None:
    _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class AstrocyteLIFNeuron:
    """Astrocyte-LIF hybrid with tripartite synapse feedback.

    Runtime state is revalidated before every step. Calcium and membrane
    candidates are computed locally and committed only after both are finite
    and physiologically admissible, preventing corrupted glial state from
    leaking into downstream membrane dynamics.
    """

    tau_m: float = 20.0
    tau_ca: float = 500.0
    e_l: float = -65.0
    theta: float = -50.0
    v_reset: float = -65.0
    ca_delta: float = 0.1
    ca_thresh: float = 0.5
    g_glio: float = 2.0
    dt: float = 0.1

    v: float = -65.0
    ca: float = 0.0

    def __post_init__(self) -> None:
        """Validate the configuration and the initial state.

        Raises
        ------
        ValueError
            If a parameter or state value is non-finite or outside its range, or
            if ``theta`` is not above ``v_reset``.
        """
        self._validate()

    def _validate(self) -> None:
        for name in ("tau_m", "tau_ca", "dt"):
            _positive(name, getattr(self, name))
        for name in ("e_l", "theta", "v_reset", "v"):
            _finite(name, getattr(self, name))
        if self.theta <= self.v_reset:
            raise ValueError("theta must be greater than v_reset")
        for name in ("ca_delta", "ca_thresh", "g_glio", "ca"):
            _non_negative(name, getattr(self, name))

    def step_with_pre(self, i_ext: float, pre_spike: bool) -> int:
        """Advance one Euler step with external current and a presynaptic spike flag.

        Calcium decays with ``tau_ca`` and gains ``ca_delta`` on a presynaptic
        spike. The gliotransmitter current ``g_glio`` flows while the new calcium
        exceeds ``ca_thresh``. The membrane follows
        ``tau_m dV/dt = -(V - E_L) + I_ext + I_glio`` and resets to ``v_reset``
        when it reaches ``theta``. The state changes only after every candidate
        value has passed its check.

        Parameters
        ----------
        i_ext : float
            External current, in the voltage units of the membrane equation.
        pre_spike : bool
            Whether a presynaptic spike arrives in this step.

        Returns
        -------
        int
            1 on a spike, else 0.

        Raises
        ------
        ValueError
            If the state, a parameter, ``i_ext`` or a candidate value is invalid.
        TypeError
            If ``pre_spike`` is not a ``bool``.
        """
        self._validate()
        _finite("i_ext", i_ext)
        if type(pre_spike) is not bool:
            raise TypeError("pre_spike must be bool")

        dca = -self.ca / self.tau_ca
        if pre_spike:
            dca += self.ca_delta / self.dt
        ca_next = max(self.ca + dca * self.dt, 0.0)
        _non_negative("ca candidate", ca_next)

        i_glio = self.g_glio if ca_next > self.ca_thresh else 0.0
        _finite("gliotransmitter current", i_glio)
        dv = (-(self.v - self.e_l) + i_ext + i_glio) / self.tau_m
        v_next = self.v + dv * self.dt
        _finite("membrane candidate", v_next)

        spike = v_next >= self.theta
        self.ca = ca_next
        self.v = self.v_reset if spike else v_next
        return 1 if spike else 0

    def step(self, current: float) -> int:
        """Advance one step without a presynaptic spike (no calcium increment).

        Parameters
        ----------
        current : float
            External current, in the voltage units of the membrane equation.

        Returns
        -------
        int
            1 on a spike, else 0.

        Raises
        ------
        ValueError
            If the state, a parameter or ``current`` is invalid.
        """
        return self.step_with_pre(current, pre_spike=False)

    def reset(self) -> None:
        """Return ``v`` to ``e_l`` and clear the calcium; the parameters are kept."""
        self.v = self.e_l
        self.ca = 0.0
