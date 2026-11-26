# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Destexhe et al. 1996 — thalamocortical relay with T-current and I_h

"""Destexhe thalamocortical relay neuron with low-threshold T-type calcium current."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

_isfinite = math.isfinite
_np_exp = np.exp


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


def _safe_sigmoid(value: float) -> float:
    """Evaluate the logistic sigmoid via ``np.exp`` without overflow warnings."""
    if value >= -709.0:
        return 1.0 / (1.0 + float(_np_exp(-value)))
    exp_v = float(_np_exp(value))
    return exp_v / (1.0 + exp_v)


def _safe_exp_pos(value: float) -> float:
    """Evaluate ``np.exp(value)``, returning ``+inf`` above the IEEE 754 overflow bound."""
    if value > 709.0:
        return float("inf")
    return float(_np_exp(value))


@dataclass
class DestexheThalamicNeuron:
    """Destexhe et al. 1996 — thalamocortical relay with T-current and I_h.

    6 ODEs: V, m_Na, h_Na, n_K, m_T, h_T (+ optional h-current).

    Reference: Destexhe, Bal, McCormick & Sejnowski (1996). Ionic mechanisms
    underlying synchronized oscillations and propagating waves in a model of
    ferret thalamic slices. J Neurophysiol 76:2049-2070.

    Parameters
    ----------
    v : float, default=-65.0
        Initial membrane potential in millivolts.
    h_na : float, default=0.6
        Initial sodium inactivation gate.
    n_k : float, default=0.3
        Initial potassium activation gate.
    m_t : float, default=0.0
        Initial T-current activation gate.
    h_t : float, default=1.0
        Initial T-current inactivation gate.
    g_na : float, default=100.0
        Maximal sodium conductance; must be non-negative.
    g_k : float, default=10.0
        Maximal potassium conductance; must be non-negative.
    g_t : float, default=2.0
        Low-threshold T-type calcium conductance; must be non-negative.
    g_l : float, default=0.05
        Leak conductance; must be non-negative.
    e_na : float, default=50.0
        Sodium reversal potential in millivolts.
    e_k : float, default=-90.0
        Potassium reversal potential in millivolts.
    e_ca : float, default=120.0
        Calcium reversal potential in millivolts.
    e_l : float, default=-70.0
        Leak reversal potential in millivolts.
    dt : float, default=0.02
        Substep integration interval in milliseconds; must be strictly positive.
    v_threshold : float, default=-20.0
        Upward spike-detection threshold in millivolts.

    Raises
    ------
    TypeError
        If any parameter is a boolean or not a real number.
    ValueError
        If any parameter is non-finite or violates its positivity or
        non-negativity domain.
    """

    v: float = -65.0
    h_na: float = 0.6
    n_k: float = 0.3
    m_t: float = 0.0
    h_t: float = 1.0
    g_na: float = 100.0
    g_k: float = 10.0
    g_t: float = 2.0
    g_l: float = 0.05
    e_na: float = 50.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_l: float = -70.0
    dt: float = 0.02
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        """Validate and normalise all initial state variables and parameters."""
        self.v = _require_finite_float(self.v, "v")
        self.h_na = _require_finite_float(self.h_na, "h_na")
        self.n_k = _require_finite_float(self.n_k, "n_k")
        self.m_t = _require_finite_float(self.m_t, "m_t")
        self.h_t = _require_finite_float(self.h_t, "h_t")
        self.g_na = _require_non_negative_float(self.g_na, "g_na")
        self.g_k = _require_non_negative_float(self.g_k, "g_k")
        self.g_t = _require_non_negative_float(self.g_t, "g_t")
        self.g_l = _require_non_negative_float(self.g_l, "g_l")
        self.e_na = _require_finite_float(self.e_na, "e_na")
        self.e_k = _require_finite_float(self.e_k, "e_k")
        self.e_ca = _require_finite_float(self.e_ca, "e_ca")
        self.e_l = _require_finite_float(self.e_l, "e_l")
        self.dt = _require_positive_float(self.dt, "dt")
        self.v_threshold = _require_finite_float(self.v_threshold, "v_threshold")

    def step(self, current: float) -> int:
        """Advance the thalamocortical relay neuron by five Euler substeps.

        Parameters
        ----------
        current : float
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
            If ``current`` or any field is non-finite or out of domain, or if
            any substep produces a non-finite state; state is unchanged on error.
        """
        v = self.v
        h_na = self.h_na
        n_k = self.n_k
        m_t = self.m_t
        h_t = self.h_t
        g_na = self.g_na
        g_k = self.g_k
        g_t = self.g_t
        g_l = self.g_l
        e_na = self.e_na
        e_k = self.e_k
        e_ca = self.e_ca
        e_l = self.e_l
        dt = self.dt
        v_threshold = self.v_threshold
        if not (
            current.__class__ is float
            and v.__class__ is float
            and h_na.__class__ is float
            and n_k.__class__ is float
            and m_t.__class__ is float
            and h_t.__class__ is float
            and g_na.__class__ is float
            and g_k.__class__ is float
            and g_t.__class__ is float
            and g_l.__class__ is float
            and e_na.__class__ is float
            and e_k.__class__ is float
            and e_ca.__class__ is float
            and e_l.__class__ is float
            and dt.__class__ is float
            and v_threshold.__class__ is float
            and (
                current
                + v
                + h_na
                + n_k
                + m_t
                + h_t
                + g_na
                + g_k
                + g_t
                + g_l
                + e_na
                + e_k
                + e_ca
                + e_l
                + dt
                + v_threshold
            )
            * 0.0
            == 0.0
            and g_na >= 0.0
            and g_k >= 0.0
            and g_t >= 0.0
            and g_l >= 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            v = _require_finite_float(v, "v")
            h_na = _require_finite_float(h_na, "h_na")
            n_k = _require_finite_float(n_k, "n_k")
            m_t = _require_finite_float(m_t, "m_t")
            h_t = _require_finite_float(h_t, "h_t")
            g_na = _require_non_negative_float(g_na, "g_na")
            g_k = _require_non_negative_float(g_k, "g_k")
            g_t = _require_non_negative_float(g_t, "g_t")
            g_l = _require_non_negative_float(g_l, "g_l")
            e_na = _require_finite_float(e_na, "e_na")
            e_k = _require_finite_float(e_k, "e_k")
            e_ca = _require_finite_float(e_ca, "e_ca")
            e_l = _require_finite_float(e_l, "e_l")
            dt = _require_positive_float(dt, "dt")
            v_threshold = _require_finite_float(v_threshold, "v_threshold")

        v_prev = v
        for _ in range(5):
            m_na_inf = _safe_sigmoid((v + 37.0) / 7.0)
            h_na_inf = _safe_sigmoid(-(v + 41.0) / 4.0)
            n_k_inf = _safe_sigmoid((v + 25.0) / 12.0)
            m_t_inf = _safe_sigmoid((v + 57.0) / 6.5)
            h_t_inf = _safe_sigmoid(-(v + 81.0) / 4.0)

            tau_h_na = 1.0 / (
                0.128 * _safe_exp_pos(-(v + 46.0) / 18.0) + 4.0 * _safe_sigmoid((v + 23.0) / 5.0)
            )
            tau_n_k = 1.0 / (0.032 * 5.0 + 0.5 * _safe_exp_pos(-(v + 40.0) / 40.0))
            tau_h_t = (
                30.8
                + 211.4
                * float(_np_exp((v + 115.2) / 5.0))
                / (1.0 + float(_np_exp((v + 86.0) / 3.2)))
                if v < -81.0
                else 10.0
            )

            h_na += (h_na_inf - h_na) / max(tau_h_na, 0.1) * dt
            n_k += (n_k_inf - n_k) / max(tau_n_k, 0.1) * dt
            m_t = m_t_inf
            h_t += (h_t_inf - h_t) / max(tau_h_t, 0.1) * dt

            i_na = g_na * m_na_inf**3 * h_na * (v - e_na)
            i_k = g_k * n_k**4 * (v - e_k)
            i_t = g_t * m_t**2 * h_t * (v - e_ca)
            i_l = g_l * (v - e_l)
            v += (-i_na - i_k - i_t - i_l + current) * dt
            if not (
                v - v == 0.0
                and h_na - h_na == 0.0
                and n_k - n_k == 0.0
                and m_t - m_t == 0.0
                and h_t - h_t == 0.0
            ):
                raise ValueError("DestexheThalamic step produced a non-finite state.")

        self.v = v
        self.h_na = h_na
        self.n_k = n_k
        self.m_t = m_t
        self.h_t = h_t
        return 1 if (v >= v_threshold and v_prev < v_threshold) else 0

    def reset(self) -> None:
        """Reset membrane potential and ionic gating variables to defaults."""
        self.v = -65.0
        self.h_na, self.n_k, self.m_t, self.h_t = 0.6, 0.3, 0.0, 1.0
