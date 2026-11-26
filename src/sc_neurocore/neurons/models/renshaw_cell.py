# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Renshaw Cell (Spinal Inhibitory Interneuron)

"""Spinal inhibitory Renshaw cell with Wang-Buzsaki gating and adaptation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

_exp = math.exp
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


def _require_bounded_float(value: object, name: str, low: float, high: float) -> float:
    """Validate that ``value`` is finite and within ``[low, high]``."""
    result = _require_finite_float(value, name)
    if result < low or result > high:
        raise ValueError(f"{name} must be within [{low}, {high}], got {value!r}.")
    return result


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - _exp(-d / k))


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _exact_gate(previous: float, alpha: float, beta: float, phi: float, dt: float) -> float:
    total = phi * (alpha + beta)
    if not _isfinite(total):
        raise ValueError(f"gate relaxation rate must be finite, got {total!r}.")
    steady = alpha / (alpha + beta)
    return _clamp01(steady + (previous - steady) * _exp(-total * dt))


def _exact_relax(previous: float, steady: float, tau: float, dt: float) -> float:
    return _clamp01(steady + (previous - steady) * _exp(-dt / tau))


def _physiological_voltage(value: float) -> bool:
    return _isfinite(value) and -150.0 <= value <= 100.0


@dataclass
class RenshawCell:
    """Renshaw cell — spinal inhibitory interneuron for recurrent inhibition.

    WB gating core with strong adaptation to produce burst-then-decay
    response to motor axon collateral input.

    Reference: Renshaw (1941); Windhorst (1996) Prog Neurobiol 46(5).

    Parameters
    ----------
    v : float, default=-65.0
        Initial membrane potential in millivolts within ``[-150, 100]``.
    h : float, default=0.8
        Initial sodium inactivation gate within ``[0, 1]``.
    n : float, default=0.1
        Initial potassium activation gate within ``[0, 1]``.
    adapt : float, default=0.0
        Initial adaptation gate within ``[0, 1]``.
    g_na : float, default=35.0
        Maximal sodium conductance; must be non-negative.
    g_k : float, default=9.0
        Maximal potassium conductance; must be non-negative.
    g_adapt : float, default=5.0
        Maximal adaptation conductance; must be non-negative.
    g_l : float, default=0.12
        Leak conductance; must be non-negative.
    e_na : float, default=55.0
        Sodium reversal potential in millivolts.
    e_k : float, default=-90.0
        Potassium reversal potential in millivolts.
    e_l : float, default=-65.0
        Leak reversal potential in millivolts.
    c_m : float, default=1.0
        Membrane capacitance; must be strictly positive.
    phi : float, default=5.0
        Gating temperature/rate scaling factor; must be strictly positive.
    tau_adapt : float, default=50.0
        Adaptation time constant in milliseconds; must be strictly positive.
    dt : float, default=0.01
        Substep integration interval in milliseconds; must be strictly positive.
    v_threshold : float, default=-20.0
        Upward spike-detection threshold in millivolts.

    Raises
    ------
    TypeError
        If any parameter is a boolean or not a real number.
    ValueError
        If any parameter is non-finite or violates its physiological,
        positivity, non-negativity, or ``[0, 1]`` gating domain.
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    adapt: float = 0.0
    g_na: float = 35.0
    g_k: float = 9.0
    g_adapt: float = 5.0
    g_l: float = 0.12
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    tau_adapt: float = 50.0
    dt: float = 0.01
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        """Validate and normalise all initial state variables and parameters."""
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        """Validate and normalise every state variable and parameter field."""
        self.v = _require_bounded_float(self.v, "v", -150.0, 100.0)
        self.h = _require_bounded_float(self.h, "h", 0.0, 1.0)
        self.n = _require_bounded_float(self.n, "n", 0.0, 1.0)
        self.adapt = _require_bounded_float(self.adapt, "adapt", 0.0, 1.0)
        self.g_na = _require_non_negative_float(self.g_na, "g_na")
        self.g_k = _require_non_negative_float(self.g_k, "g_k")
        self.g_adapt = _require_non_negative_float(self.g_adapt, "g_adapt")
        self.g_l = _require_non_negative_float(self.g_l, "g_l")
        self.e_na = _require_finite_float(self.e_na, "e_na")
        self.e_k = _require_finite_float(self.e_k, "e_k")
        self.e_l = _require_finite_float(self.e_l, "e_l")
        self.c_m = _require_positive_float(self.c_m, "c_m")
        self.phi = _require_positive_float(self.phi, "phi")
        self.tau_adapt = _require_positive_float(self.tau_adapt, "tau_adapt")
        self.dt = _require_positive_float(self.dt, "dt")
        self.v_threshold = _require_finite_float(self.v_threshold, "v_threshold")

    def step(self, current: float = 0.0) -> int:
        """Advance gating variables and membrane potential over one 0.5 ms window.

        Parameters
        ----------
        current : float, default=0.0
            External injected current.

        Returns
        -------
        int
            ``1`` on an upward crossing of ``v_threshold``, else ``0``.

        Raises
        ------
        TypeError
            If ``current`` or any neuron field is a boolean or not a real number.
        ValueError
            If ``current`` or any field is non-finite or out of domain, if a
            gate relaxation rate overflows, if total conductance is non-positive
            or non-finite, or if candidate voltage leaves ``[-150, 100]`` mV;
            state is unchanged on error.
        """
        current = _require_finite_float(current, "current")
        self._validate_runtime()

        v_prev = self.v
        v = self.v
        h = self.h
        n = self.n
        adapt = self.adapt
        n_sub = max(1, int(0.5 / max(self.dt, 0.001)))
        for _ in range(n_sub):
            am = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            bm = 4.0 * _exp(-(v + 60.0) / 18.0)
            m_inf = am / (am + bm)
            ah = 0.07 * _exp(-(v + 58.0) / 20.0)
            bh = 1.0 / (1.0 + _exp(-(v + 28.0) / 10.0))
            an = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            bn = 0.125 * _exp(-(v + 44.0) / 80.0)

            h_next = _exact_gate(h, ah, bh, self.phi, self.dt)
            n_next = _exact_gate(n, an, bn, self.phi, self.dt)

            adapt_inf = 1.0 / (1.0 + _exp(-(v + 30.0) / 5.0))
            adapt_next = _exact_relax(adapt, adapt_inf, self.tau_adapt, self.dt)

            g_na = self.g_na * m_inf**3 * h_next
            g_k = self.g_k * n_next**4
            g_adapt = self.g_adapt * adapt_next
            g_total = g_na + g_k + g_adapt + self.g_l
            if not _isfinite(g_total) or g_total <= 0.0:
                raise ValueError(
                    f"total membrane conductance must be finite and positive, got {g_total!r}."
                )

            steady_v = (
                current
                + g_na * self.e_na
                + g_k * self.e_k
                + g_adapt * self.e_k
                + self.g_l * self.e_l
            ) / g_total
            v_next = steady_v + (v - steady_v) * _exp(-(g_total / self.c_m) * self.dt)
            if not _physiological_voltage(v_next):
                raise ValueError(
                    f"candidate membrane voltage must be finite and within [-150.0, 100.0], "
                    f"got {v_next!r}."
                )

            v = v_next
            h = h_next
            n = n_next
            adapt = adapt_next

        self.v = v
        self.h = h
        self.n = n
        self.adapt = adapt

        return 1 if self.v >= self.v_threshold and v_prev < self.v_threshold else 0

    def reset(self) -> None:
        """Restore membrane potential, ionic gates, and adaptation to resting defaults."""
        self.v = -65.0
        self.h = 0.8
        self.n = 0.1
        self.adapt = 0.0
