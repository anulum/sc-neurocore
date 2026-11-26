# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pagkalos, Chavlis & Poirazi 2023 — Dendrify active-dendrite model

"""Dendrify two-compartment active-dendrite leaky integrate-and-fire neuron."""

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


def _require_bool(value: object, name: str) -> bool:
    """Validate that ``value`` is a Python or NumPy boolean."""
    if type(value) is bool:
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"{name} must be a boolean, got {type(value).__name__}.")


@dataclass
class DendrifyNeuron:
    """Pagkalos, Chavlis & Poirazi 2023 — Dendrify active-dendrite 2-compartment model.

    Soma: standard LIF with reset.
    Dendrite: has a dendritic spike mechanism (NMDA-like) that produces
    a supralinear calcium plateau when input exceeds d_threshold.

    Reference: Pagkalos, M. et al. (2023). Nat. Commun. 14:3234.

    Parameters
    ----------
    v_s : float, default=-65.0
        Initial somatic membrane potential in millivolts.
    v_d : float, default=-65.0
        Initial dendritic membrane potential in millivolts.
    d_active : bool, default=False
        Whether the dendritic plateau is initially active.
    tau_s : float, default=10.0
        Somatic membrane time constant in milliseconds; must be strictly positive.
    tau_d : float, default=20.0
        Dendritic membrane time constant in milliseconds; must be strictly positive.
    g_c : float, default=0.8
        Soma-dendrite coupling conductance; must be non-negative.
    d_threshold : float, default=-35.0
        Dendritic plateau initiation threshold in millivolts.
    d_amplitude : float, default=30.0
        Dendritic plateau current injection amplitude.
    d_duration : float, default=10.0
        Dendritic plateau duration in milliseconds; must be non-negative.
    d_timer : float, default=0.0
        Remaining plateau timer in milliseconds.
    v_rest : float, default=-65.0
        Resting potential in millivolts.
    v_threshold : float, default=-50.0
        Somatic firing threshold in millivolts.
    v_reset : float, default=-65.0
        Somatic post-spike reset potential in millivolts.
    dt : float, default=0.1
        Discrete integration timestep in milliseconds; must be strictly positive.

    Raises
    ------
    TypeError
        If any numeric parameter is a boolean or not a real number, or if
        ``d_active`` is not a boolean.
    ValueError
        If any parameter is non-finite or violates its positivity or
        non-negativity domain.
    """

    v_s: float = -65.0
    v_d: float = -65.0
    d_active: bool = False  # dendritic plateau state
    tau_s: float = 10.0
    tau_d: float = 20.0
    g_c: float = 0.8  # soma-dendrite coupling
    d_threshold: float = -35.0  # mV, dendritic spike threshold
    d_amplitude: float = 30.0  # mV, plateau amplitude
    d_duration: float = 10.0  # ms
    d_timer: float = 0.0
    v_rest: float = -65.0
    v_threshold: float = -50.0
    v_reset: float = -65.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        """Validate and normalise all compartment states and parameters."""
        self.v_s = _require_finite_float(self.v_s, "v_s")
        self.v_d = _require_finite_float(self.v_d, "v_d")
        self.d_active = _require_bool(self.d_active, "d_active")
        self.tau_s = _require_positive_float(self.tau_s, "tau_s")
        self.tau_d = _require_positive_float(self.tau_d, "tau_d")
        self.g_c = _require_non_negative_float(self.g_c, "g_c")
        self.d_threshold = _require_finite_float(self.d_threshold, "d_threshold")
        self.d_amplitude = _require_finite_float(self.d_amplitude, "d_amplitude")
        self.d_duration = _require_non_negative_float(self.d_duration, "d_duration")
        self.d_timer = _require_finite_float(self.d_timer, "d_timer")
        self.v_rest = _require_finite_float(self.v_rest, "v_rest")
        self.v_threshold = _require_finite_float(self.v_threshold, "v_threshold")
        self.v_reset = _require_finite_float(self.v_reset, "v_reset")
        self.dt = _require_positive_float(self.dt, "dt")

    def step(self, current: float) -> int:
        """Advance the dendritic and somatic compartments by one Euler timestep.

        Parameters
        ----------
        current : float
            External dendritic input drive.

        Returns
        -------
        int
            ``1`` on an upward somatic crossing of ``v_threshold``, else ``0``.

        Raises
        ------
        TypeError
            If ``current`` or any numeric field is a boolean or not a real number,
            or if ``d_active`` is not a boolean.
        ValueError
            If ``current`` or any field is non-finite or out of domain, or if
            candidate compartment states overflow; state is unchanged on error.
        """
        v_s = self.v_s
        v_d = self.v_d
        d_active = self.d_active
        tau_s = self.tau_s
        tau_d = self.tau_d
        g_c = self.g_c
        d_threshold = self.d_threshold
        d_amplitude = self.d_amplitude
        d_duration = self.d_duration
        d_timer = self.d_timer
        v_rest = self.v_rest
        v_threshold = self.v_threshold
        v_reset = self.v_reset
        dt = self.dt
        if not (
            current.__class__ is float
            and v_s.__class__ is float
            and v_d.__class__ is float
            and d_active.__class__ is bool
            and tau_s.__class__ is float
            and tau_d.__class__ is float
            and g_c.__class__ is float
            and d_threshold.__class__ is float
            and d_amplitude.__class__ is float
            and d_duration.__class__ is float
            and d_timer.__class__ is float
            and v_rest.__class__ is float
            and v_threshold.__class__ is float
            and v_reset.__class__ is float
            and dt.__class__ is float
            and (
                current
                + v_s
                + v_d
                + tau_s
                + tau_d
                + g_c
                + d_threshold
                + d_amplitude
                + d_duration
                + d_timer
                + v_rest
                + v_threshold
                + v_reset
                + dt
            )
            * 0.0
            == 0.0
            and tau_s > 0.0
            and tau_d > 0.0
            and g_c >= 0.0
            and d_duration >= 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            v_s = _require_finite_float(v_s, "v_s")
            v_d = _require_finite_float(v_d, "v_d")
            d_active = _require_bool(d_active, "d_active")
            tau_s = _require_positive_float(tau_s, "tau_s")
            tau_d = _require_positive_float(tau_d, "tau_d")
            g_c = _require_non_negative_float(g_c, "g_c")
            d_threshold = _require_finite_float(d_threshold, "d_threshold")
            d_amplitude = _require_finite_float(d_amplitude, "d_amplitude")
            d_duration = _require_non_negative_float(d_duration, "d_duration")
            d_timer = _require_finite_float(d_timer, "d_timer")
            v_rest = _require_finite_float(v_rest, "v_rest")
            v_threshold = _require_finite_float(v_threshold, "v_threshold")
            v_reset = _require_finite_float(v_reset, "v_reset")
            dt = _require_positive_float(dt, "dt")

        dv_d = (-(v_d - v_rest) + current - g_c * (v_d - v_s)) / tau_d
        next_v_d = v_d + dv_d * dt
        next_d_active = d_active
        next_d_timer = d_timer
        if not next_d_active and next_v_d >= d_threshold:
            next_d_active = True
            next_d_timer = d_duration
        if next_d_active:
            next_d_timer -= dt
            d_inject = d_amplitude
            if next_d_timer <= 0.0:
                next_d_active = False
        else:
            d_inject = 0.0

        dv_s = (-(v_s - v_rest) + g_c * (next_v_d - v_s) + d_inject) / tau_s
        next_v_s = v_s + dv_s * dt
        if not (
            next_v_d - next_v_d == 0.0
            and next_d_timer - next_d_timer == 0.0
            and next_v_s - next_v_s == 0.0
        ):
            raise ValueError("Dendrify step produced a non-finite compartment state.")

        self.v_d = next_v_d
        self.d_active = next_d_active
        self.d_timer = next_d_timer
        if next_v_s >= v_threshold and v_s < v_threshold:
            self.v_s = v_reset
            return 1
        self.v_s = next_v_s
        return 0

    def reset(self) -> None:
        """Reset somatic, dendritic, and plateau states to rest."""
        self.v_s, self.v_d = -65.0, -65.0
        self.d_active, self.d_timer = False, 0.0
