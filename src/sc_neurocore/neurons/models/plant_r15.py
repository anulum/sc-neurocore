# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Plant 1981 — Aplysia R15 parabolic burster, 5 ODEs

"""Plant R15 five-ODE parabolic bursting conductance neuron."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

_isfinite = math.isfinite
_np_exp = np.exp


def _checked_np_exp(value: float) -> float:
    """Evaluate ``np.exp(value)`` as Python ``float``, returning ``+inf`` above overflow."""
    if value > 709.782712893384:
        return float("inf")
    return float(_np_exp(value))


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


@dataclass
class PlantR15Neuron:
    """Plant 1981 — Aplysia R15 parabolic burster, 5 ODEs.

    dV/dt  = (-I_Na - I_K - I_Ca - I_L - I_KCa + I_ext) / C
    dm/dt  = (m_inf(V) - m) / tau_m(V)
    dh/dt  = (h_inf(V) - h) / tau_h(V)
    dn/dt  = (n_inf(V) - n) / tau_n(V)
    dCa/dt = -k_Ca * I_Ca - Ca / tau_Ca

    Reference: Plant, R.E. & Kim, M. (1976). Biophys. J. 16:227–244.

    Parameters
    ----------
    v : float, default=-50.0
        Initial membrane potential in millivolts.
    m : float, default=0.05
        Initial sodium activation gate.
    h : float, default=0.6
        Initial sodium inactivation gate.
    n : float, default=0.3
        Initial potassium activation gate.
    ca : float, default=0.1
        Initial intracellular calcium concentration; must be non-negative.
    g_na : float, default=4.0
        Maximal sodium conductance; must be non-negative.
    g_k : float, default=0.3
        Maximal potassium conductance; must be non-negative.
    g_ca : float, default=0.004
        Maximal calcium conductance; must be non-negative.
    g_l : float, default=0.003
        Leak conductance; must be non-negative.
    g_kca : float, default=0.03
        Maximal calcium-activated potassium conductance; must be non-negative.
    e_na : float, default=30.0
        Sodium reversal potential in millivolts.
    e_k : float, default=-75.0
        Potassium reversal potential in millivolts.
    e_ca : float, default=140.0
        Calcium reversal potential in millivolts.
    e_l : float, default=-40.0
        Leak reversal potential in millivolts.
    c_m : float, default=1.0
        Membrane capacitance; must be strictly positive.
    k_ca : float, default=0.0085
        Calcium influx scaling factor; must be non-negative.
    tau_ca : float, default=500.0
        Calcium decay time constant in milliseconds; must be strictly positive.
    dt : float, default=0.05
        Substep integration interval in milliseconds; must be strictly positive.
    v_threshold : float, default=-10.0
        Upward spike-detection threshold in millivolts.

    Raises
    ------
    TypeError
        If any parameter is a boolean or not a real number.
    ValueError
        If any parameter is non-finite or violates its positivity or
        non-negativity domain.
    """

    v: float = -50.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.3
    ca: float = 0.1
    g_na: float = 4.0
    g_k: float = 0.3
    g_ca: float = 0.004
    g_l: float = 0.003
    g_kca: float = 0.03
    e_na: float = 30.0
    e_k: float = -75.0
    e_ca: float = 140.0
    e_l: float = -40.0
    c_m: float = 1.0
    k_ca: float = 0.0085
    tau_ca: float = 500.0
    dt: float = 0.05
    v_threshold: float = -10.0

    def __post_init__(self) -> None:
        """Validate and normalise all state variables and parameters."""
        self.v = _require_finite_float(self.v, "v")
        self.m = _require_finite_float(self.m, "m")
        self.h = _require_finite_float(self.h, "h")
        self.n = _require_finite_float(self.n, "n")
        self.ca = _require_non_negative_float(self.ca, "ca")
        self.g_na = _require_non_negative_float(self.g_na, "g_na")
        self.g_k = _require_non_negative_float(self.g_k, "g_k")
        self.g_ca = _require_non_negative_float(self.g_ca, "g_ca")
        self.g_l = _require_non_negative_float(self.g_l, "g_l")
        self.g_kca = _require_non_negative_float(self.g_kca, "g_kca")
        self.e_na = _require_finite_float(self.e_na, "e_na")
        self.e_k = _require_finite_float(self.e_k, "e_k")
        self.e_ca = _require_finite_float(self.e_ca, "e_ca")
        self.e_l = _require_finite_float(self.e_l, "e_l")
        self.c_m = _require_positive_float(self.c_m, "c_m")
        self.k_ca = _require_non_negative_float(self.k_ca, "k_ca")
        self.tau_ca = _require_positive_float(self.tau_ca, "tau_ca")
        self.dt = _require_positive_float(self.dt, "dt")
        self.v_threshold = _require_finite_float(self.v_threshold, "v_threshold")

    def step(self, current: float) -> int:
        """Advance the five-ODE Plant R15 model by five Euler substeps.

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
            any substep overflows or produces a non-finite state; state is
            unchanged on error.
        """
        v = self.v
        m = self.m
        h = self.h
        n = self.n
        ca = self.ca
        g_na = self.g_na
        g_k = self.g_k
        g_ca = self.g_ca
        g_l = self.g_l
        g_kca = self.g_kca
        e_na = self.e_na
        e_k = self.e_k
        e_ca = self.e_ca
        e_l = self.e_l
        c_m = self.c_m
        k_ca = self.k_ca
        tau_ca = self.tau_ca
        dt = self.dt
        v_threshold = self.v_threshold
        if not (
            current.__class__ is float
            and v.__class__ is float
            and m.__class__ is float
            and h.__class__ is float
            and n.__class__ is float
            and ca.__class__ is float
            and g_na.__class__ is float
            and g_k.__class__ is float
            and g_ca.__class__ is float
            and g_l.__class__ is float
            and g_kca.__class__ is float
            and e_na.__class__ is float
            and e_k.__class__ is float
            and e_ca.__class__ is float
            and e_l.__class__ is float
            and c_m.__class__ is float
            and k_ca.__class__ is float
            and tau_ca.__class__ is float
            and dt.__class__ is float
            and v_threshold.__class__ is float
            and (
                current
                + v
                + m
                + h
                + n
                + ca
                + g_na
                + g_k
                + g_ca
                + g_l
                + g_kca
                + e_na
                + e_k
                + e_ca
                + e_l
                + c_m
                + k_ca
                + tau_ca
                + dt
                + v_threshold
            )
            * 0.0
            == 0.0
            and ca >= 0.0
            and g_na >= 0.0
            and g_k >= 0.0
            and g_ca >= 0.0
            and g_l >= 0.0
            and g_kca >= 0.0
            and c_m > 0.0
            and k_ca >= 0.0
            and tau_ca > 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            v = _require_finite_float(v, "v")
            m = _require_finite_float(m, "m")
            h = _require_finite_float(h, "h")
            n = _require_finite_float(n, "n")
            ca = _require_non_negative_float(ca, "ca")
            g_na = _require_non_negative_float(g_na, "g_na")
            g_k = _require_non_negative_float(g_k, "g_k")
            g_ca = _require_non_negative_float(g_ca, "g_ca")
            g_l = _require_non_negative_float(g_l, "g_l")
            g_kca = _require_non_negative_float(g_kca, "g_kca")
            e_na = _require_finite_float(e_na, "e_na")
            e_k = _require_finite_float(e_k, "e_k")
            e_ca = _require_finite_float(e_ca, "e_ca")
            e_l = _require_finite_float(e_l, "e_l")
            c_m = _require_positive_float(c_m, "c_m")
            k_ca = _require_non_negative_float(k_ca, "k_ca")
            tau_ca = _require_positive_float(tau_ca, "tau_ca")
            dt = _require_positive_float(dt, "dt")
            v_threshold = _require_finite_float(v_threshold, "v_threshold")

        v_prev = v
        try:
            for _ in range(5):
                am = 0.1 * (50.0 + v) / (1.0 - _checked_np_exp(-(50.0 + v) / 10.0) + 1e-12)
                bm = 4.0 * _checked_np_exp(-(75.0 + v) / 18.0)
                ah = 0.07 * _checked_np_exp(-(v + 50.0) / 20.0)
                bh = 1.0 / (1.0 + _checked_np_exp(-(20.0 + v) / 10.0))
                an = 0.01 * (55.0 + v) / (1.0 - _checked_np_exp(-(55.0 + v) / 10.0) + 1e-12)
                bn = 0.125 * _checked_np_exp(-(65.0 + v) / 80.0)

                m += (am * (1.0 - m) - bm * m) * dt
                h += (ah * (1.0 - h) - bh * h) * dt
                n += (an * (1.0 - n) - bn * n) * dt

                m_ca_inf = 1.0 / (1.0 + _checked_np_exp(-(v + 25.0) / 5.0))
                i_na = g_na * m**3 * h * (v - e_na)
                i_k = g_k * n**4 * (v - e_k)
                i_ca = g_ca * m_ca_inf**2 * (v - e_ca)
                i_kca = g_kca * ca / (0.5 + ca) * (v - e_k)
                i_l = g_l * (v - e_l)

                v += (-i_na - i_k - i_ca - i_kca - i_l + current) / c_m * dt
                ca += (-k_ca * i_ca - ca / tau_ca) * dt
                if ca < 0.0:
                    ca = 0.0
                if not (
                    v - v == 0.0
                    and m - m == 0.0
                    and h - h == 0.0
                    and n - n == 0.0
                    and ca - ca == 0.0
                ):
                    raise ValueError("PlantR15 step produced a non-finite state.")
        except OverflowError as exc:
            raise ValueError("PlantR15 step overflowed to a non-finite state.") from exc

        self.v = v
        self.m = m
        self.h = h
        self.n = n
        self.ca = ca
        return 1 if (v >= v_threshold and v_prev < v_threshold) else 0

    def reset(self) -> None:
        """Reset membrane, gating, and calcium states to rest."""
        self.v = -50.0
        self.m, self.h, self.n = 0.05, 0.6, 0.3
        self.ca = 0.1
