# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tsodyks & Markram 1997 — LIF with short-term synaptic

"""Tsodyks-Markram LIF neuron with short-term synaptic depression and facilitation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

_isfinite = math.isfinite


def _require_finite_float(value: object, name: str) -> float:
    """Validate that ``value`` is a finite real number and return it as ``float``."""
    if type(value) is float:
        result = value
    elif isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    else:
        result = float(value)
    if not _isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _require_positive_float(value: object, name: str) -> float:
    """Validate that ``value`` is a strictly positive finite real number."""
    result = _require_finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return result


def _require_non_negative_float(value: object, name: str) -> float:
    """Validate that ``value`` is a non-negative finite real number."""
    result = _require_finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return result


def _require_unit_interval(value: object, name: str) -> float:
    """Validate that ``value`` is a finite real number in ``[0, 1]``."""
    result = _require_finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}.")
    return result


def _require_bool(value: object, name: str) -> bool:
    """Validate that ``value`` is a Python or NumPy boolean."""
    if type(value) is bool:
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"{name} must be a boolean, got {type(value).__name__}.")


@dataclass
class TsodyksMarkramNeuron:
    """Tsodyks & Markram 1997 — LIF with short-term synaptic plasticity.

    LIF membrane: tau_m dV/dt = -(V - V_rest) + R*I_syn + R*I_ext
    dx/dt = (1 - x)/tau_d - u*x*delta(spike_in)
    du/dt = (U - u)/tau_f + U*(1-u)*delta(spike_in)
    I_syn = A * u * x on presynaptic spike

    Reference: Tsodyks, M. et al. (1998). Neural Comput. 10:821–835.

    Parameters
    ----------
    v : float, default=-65.0
        Initial membrane potential in millivolts.
    x : float, default=1.0
        Initial available synaptic resource fraction.
    u : float, default=0.2
        Initial utilisation (release probability) variable.
    v_rest : float, default=-65.0
        Resting potential in millivolts.
    v_reset : float, default=-65.0
        Post-spike reset potential in millivolts.
    v_threshold : float, default=-50.0
        Firing threshold in millivolts.
    tau_m : float, default=20.0
        Membrane time constant in milliseconds; must be strictly positive.
    tau_d : float, default=200.0
        Depression recovery time constant in milliseconds; must be strictly positive.
    tau_f : float, default=600.0
        Facilitation decay time constant in milliseconds; must be strictly positive.
    u_se : float, default=0.2
        Baseline synaptic utilisation in ``[0, 1]``.
    a_se : float, default=50.0
        Absolute synaptic efficacy.
    r_m : float, default=1.0
        Membrane resistance; must be non-negative.
    dt : float, default=0.1
        Discrete integration timestep in milliseconds; must be strictly positive.

    Raises
    ------
    TypeError
        If any parameter is a boolean or not a real number.
    ValueError
        If any parameter is non-finite or violates its domain bounds.
    """

    v: float = -65.0
    x: float = 1.0
    u: float = 0.2
    v_rest: float = -65.0
    v_reset: float = -65.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    tau_d: float = 200.0
    tau_f: float = 600.0
    u_se: float = 0.2
    a_se: float = 50.0
    r_m: float = 1.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        """Validate and normalise all membrane and short-term plasticity fields."""
        self.v = _require_finite_float(self.v, "v")
        self.x = _require_finite_float(self.x, "x")
        self.u = _require_finite_float(self.u, "u")
        self.v_rest = _require_finite_float(self.v_rest, "v_rest")
        self.v_reset = _require_finite_float(self.v_reset, "v_reset")
        self.v_threshold = _require_finite_float(self.v_threshold, "v_threshold")
        self.tau_m = _require_positive_float(self.tau_m, "tau_m")
        self.tau_d = _require_positive_float(self.tau_d, "tau_d")
        self.tau_f = _require_positive_float(self.tau_f, "tau_f")
        self.u_se = _require_unit_interval(self.u_se, "u_se")
        self.a_se = _require_finite_float(self.a_se, "a_se")
        self.r_m = _require_non_negative_float(self.r_m, "r_m")
        self.dt = _require_positive_float(self.dt, "dt")

    def step(self, current: float, presynaptic_spike: bool = False) -> int:
        """Advance short-term synaptic variables and LIF membrane by one timestep.

        Parameters
        ----------
        current : float
            External injected current.
        presynaptic_spike : bool, default=False
            Whether a presynaptic spike arrives during this step.

        Returns
        -------
        int
            ``1`` if the updated membrane potential reaches ``v_threshold``, else ``0``.

        Raises
        ------
        TypeError
            If ``current`` or any numeric field is a boolean or not a real number,
            or if ``presynaptic_spike`` is not a boolean.
        ValueError
            If ``current`` or any field is non-finite or out of domain, or if
            candidate state variables overflow; state is unchanged on error.
        """
        v = self.v
        x = self.x
        u = self.u
        v_rest = self.v_rest
        v_reset = self.v_reset
        v_threshold = self.v_threshold
        tau_m = self.tau_m
        tau_d = self.tau_d
        tau_f = self.tau_f
        u_se = self.u_se
        a_se = self.a_se
        r_m = self.r_m
        dt = self.dt
        if not (
            current.__class__ is float
            and presynaptic_spike.__class__ is bool
            and v.__class__ is float
            and x.__class__ is float
            and u.__class__ is float
            and v_rest.__class__ is float
            and v_reset.__class__ is float
            and v_threshold.__class__ is float
            and tau_m.__class__ is float
            and tau_d.__class__ is float
            and tau_f.__class__ is float
            and u_se.__class__ is float
            and a_se.__class__ is float
            and r_m.__class__ is float
            and dt.__class__ is float
            and (
                current
                + v
                + x
                + u
                + v_rest
                + v_reset
                + v_threshold
                + tau_m
                + tau_d
                + tau_f
                + u_se
                + a_se
                + r_m
                + dt
            )
            * 0.0
            == 0.0
            and tau_m > 0.0
            and tau_d > 0.0
            and tau_f > 0.0
            and 0.0 <= u_se <= 1.0
            and r_m >= 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            presynaptic_spike = _require_bool(presynaptic_spike, "presynaptic_spike")
            v = _require_finite_float(v, "v")
            x = _require_finite_float(x, "x")
            u = _require_finite_float(u, "u")
            v_rest = _require_finite_float(v_rest, "v_rest")
            v_reset = _require_finite_float(v_reset, "v_reset")
            v_threshold = _require_finite_float(v_threshold, "v_threshold")
            tau_m = _require_positive_float(tau_m, "tau_m")
            tau_d = _require_positive_float(tau_d, "tau_d")
            tau_f = _require_positive_float(tau_f, "tau_f")
            u_se = _require_unit_interval(u_se, "u_se")
            a_se = _require_finite_float(a_se, "a_se")
            r_m = _require_non_negative_float(r_m, "r_m")
            dt = _require_positive_float(dt, "dt")

        next_x = x + (1.0 - x) / tau_d * dt
        next_u = u + (u_se - u) / tau_f * dt

        i_syn = 0.0
        if presynaptic_spike:
            next_u += u_se * (1.0 - next_u)
            i_syn = a_se * next_u * next_x
            next_x -= next_u * next_x

        dv = (-(v - v_rest) + r_m * (i_syn + current)) / tau_m * dt
        next_v = v + dv
        if not (next_x - next_x == 0.0 and next_u - next_u == 0.0 and next_v - next_v == 0.0):
            raise ValueError("TsodyksMarkram step produced a non-finite state.")

        self.x = next_x
        self.u = next_u
        if next_v >= v_threshold:
            self.v = v_reset
            return 1
        self.v = next_v
        return 0

    def reset(self) -> None:
        """Reset membrane potential and short-term plasticity variables."""
        self.v = self.v_rest
        self.x = 1.0
        self.u = self.u_se
