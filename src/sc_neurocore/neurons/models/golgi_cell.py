# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cerebellar Golgi Cell (Solinas 2007)

"""Cerebellar Golgi cell conductance model with 11 ionic currents (Solinas 2007)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

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


def _require_positive_int(value: object, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return result


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - _exp(-d / k))


def _boltz(v: float, vh: float, k: float) -> float:
    x = (v - vh) / k
    if x >= 0.0:
        return 1.0 / (1.0 + _exp(-x))
    ex = _exp(x)
    return ex / (1.0 + ex)


def _voltage(value: float) -> bool:
    return _isfinite(value) and -100.0 <= value <= 60.0


def _gate_alpha_beta(previous: float, alpha: float, beta: float, phi: float, dt: float) -> float:
    total = phi * (alpha + beta)
    steady = alpha / (alpha + beta)
    return min(1.0, max(0.0, steady + (previous - steady) * _exp(-total * dt)))


def _gate_inf(previous: float, steady: float, tau: float, dt: float) -> float:
    return min(1.0, max(0.0, steady + (previous - steady) * _exp(-dt / tau)))


def _calcium_exact(previous: float, entry: float, tau: float, dt: float) -> float:
    steady = entry * tau
    value = steady + (previous - steady) * _exp(-dt / tau)
    if not _isfinite(value):
        raise ValueError(f"calcium state became non-finite: {value!r}.")
    return max(0.0, value)


@dataclass
class GolgiCell:
    """Cerebellar Golgi cell — Solinas et al. 2007 full model.

    11 ionic currents: INa_t (m³h), INa_p, IKdr (n⁴), IKA (a³b),
    IKM (w), ICaT (mT²s), ICaN (c²), IBK (V+Ca²⁺), ISK (Ca²⁺),
    Ih (r), IL. Spontaneously active 3–10 Hz.

    Reference: Solinas et al. (2007) Front Cell Neurosci 1:2.

    Parameters
    ----------
    v : float, default=-60.0
        Initial membrane potential in millivolts within ``[-100, 60]``.
    m, h, p_na, n, a, b, w, m_t, s, c_n, r : float
        Initial gating probabilities, each within ``[0, 1]``.
    ca : float, default=0.05
        Initial intracellular calcium concentration; must be non-negative.
    g_na_t, g_na_p, g_kdr, g_ka, g_km, g_cat, g_can, g_bk, g_sk, g_h, g_l : float
        Maximal ionic conductances; each must be non-negative.
    e_na, e_k, e_ca, e_h, e_l : float
        Ionic reversal potentials in millivolts.
    c_m : float, default=1.0
        Membrane capacitance; must be strictly positive.
    tau_ca : float, default=200.0
        Calcium decay time constant in milliseconds; must be strictly positive.
    kd_bk : float, default=1.0
        BK calcium half-activation constant; must be strictly positive.
    kd_sk : float, default=0.5
        SK calcium half-activation constant; must be strictly positive.
    dt : float, default=0.5
        Outer integration timestep in milliseconds; must be strictly positive.
    sub_steps : int, default=10
        Number of exponential-integrator substeps per ``step``; must be a positive integer.
    gain : float, default=1.0
        External input current scaling factor; must be non-negative.

    Raises
    ------
    TypeError
        If any floating-point field is a boolean or not a real number, or if
        ``sub_steps`` is a boolean or not an integer.
    ValueError
        If any field is non-finite or violates its physiological, positivity,
        non-negativity, or ``[0, 1]`` gating domain.
    """

    v: float = -60.0
    m: float = 0.02
    h: float = 0.85
    p_na: float = 0.01
    n: float = 0.05
    a: float = 0.1
    b: float = 0.8
    w: float = 0.01
    m_t: float = 0.01
    s: float = 0.9
    c_n: float = 0.01
    r: float = 0.1
    ca: float = 0.05
    g_na_t: float = 48.0
    g_na_p: float = 0.2
    g_kdr: float = 16.0
    g_ka: float = 8.0
    g_km: float = 1.0
    g_cat: float = 0.5
    g_can: float = 1.0
    g_bk: float = 3.0
    g_sk: float = 1.0
    g_h: float = 0.1
    g_l: float = 0.05
    e_na: float = 55.0
    e_k: float = -90.0
    e_ca: float = 120.0
    e_h: float = -40.0
    e_l: float = -55.0
    c_m: float = 1.0
    tau_ca: float = 200.0
    kd_bk: float = 1.0
    kd_sk: float = 0.5
    dt: float = 0.5
    sub_steps: int = 10
    gain: float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalise all initial state variables and parameters."""
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        """Validate and normalise every state variable and parameter field."""
        self.v = _require_bounded_float(self.v, "v", -100.0, 60.0)
        self.m = _require_bounded_float(self.m, "m", 0.0, 1.0)
        self.h = _require_bounded_float(self.h, "h", 0.0, 1.0)
        self.p_na = _require_bounded_float(self.p_na, "p_na", 0.0, 1.0)
        self.n = _require_bounded_float(self.n, "n", 0.0, 1.0)
        self.a = _require_bounded_float(self.a, "a", 0.0, 1.0)
        self.b = _require_bounded_float(self.b, "b", 0.0, 1.0)
        self.w = _require_bounded_float(self.w, "w", 0.0, 1.0)
        self.m_t = _require_bounded_float(self.m_t, "m_t", 0.0, 1.0)
        self.s = _require_bounded_float(self.s, "s", 0.0, 1.0)
        self.c_n = _require_bounded_float(self.c_n, "c_n", 0.0, 1.0)
        self.r = _require_bounded_float(self.r, "r", 0.0, 1.0)
        self.ca = _require_non_negative_float(self.ca, "ca")
        self.g_na_t = _require_non_negative_float(self.g_na_t, "g_na_t")
        self.g_na_p = _require_non_negative_float(self.g_na_p, "g_na_p")
        self.g_kdr = _require_non_negative_float(self.g_kdr, "g_kdr")
        self.g_ka = _require_non_negative_float(self.g_ka, "g_ka")
        self.g_km = _require_non_negative_float(self.g_km, "g_km")
        self.g_cat = _require_non_negative_float(self.g_cat, "g_cat")
        self.g_can = _require_non_negative_float(self.g_can, "g_can")
        self.g_bk = _require_non_negative_float(self.g_bk, "g_bk")
        self.g_sk = _require_non_negative_float(self.g_sk, "g_sk")
        self.g_h = _require_non_negative_float(self.g_h, "g_h")
        self.g_l = _require_non_negative_float(self.g_l, "g_l")
        self.e_na = _require_finite_float(self.e_na, "e_na")
        self.e_k = _require_finite_float(self.e_k, "e_k")
        self.e_ca = _require_finite_float(self.e_ca, "e_ca")
        self.e_h = _require_finite_float(self.e_h, "e_h")
        self.e_l = _require_finite_float(self.e_l, "e_l")
        self.c_m = _require_positive_float(self.c_m, "c_m")
        self.tau_ca = _require_positive_float(self.tau_ca, "tau_ca")
        self.kd_bk = _require_positive_float(self.kd_bk, "kd_bk")
        self.kd_sk = _require_positive_float(self.kd_sk, "kd_sk")
        self.dt = _require_positive_float(self.dt, "dt")
        self.sub_steps = _require_positive_int(self.sub_steps, "sub_steps")
        self.gain = _require_non_negative_float(self.gain, "gain")

    def step(self, current: float = 0.0) -> int:
        """Advance all 11 ionic currents, calcium, and voltage by one outer step.

        Parameters
        ----------
        current : float, default=0.0
            External injected current.

        Returns
        -------
        int
            ``1`` if membrane voltage crosses ``0.0`` mV upward, else ``0``.

        Raises
        ------
        TypeError
            If ``current`` or any float attribute is a boolean or not a real
            number, or if ``sub_steps`` is a boolean or not an integer.
        ValueError
            If ``current`` or any attribute violates its finite domain, if
            calcium overflows, if total conductance is non-positive or
            non-finite, or if candidate voltage leaves ``[-100, 60]`` mV;
            state is unchanged on error.
        """
        current = _require_finite_float(current, "current")
        self._validate_runtime()

        input_current = self.gain * current
        dt_sub = self.dt / self.sub_steps
        v_prev = self.v
        v = self.v
        m = self.m
        h = self.h
        p_na = self.p_na
        n = self.n
        a = self.a
        b = self.b
        w = self.w
        m_t = self.m_t
        s = self.s
        c_n = self.c_n
        r = self.r
        ca = self.ca

        for _ in range(self.sub_steps):
            alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
            beta_m = 4.0 * _exp(-(v + 60.0) / 18.0)
            alpha_h = 0.07 * _exp(-(v + 58.0) / 20.0)
            beta_h = 1.0 / (1.0 + _exp(-(v + 28.0) / 10.0))
            m_next = _gate_alpha_beta(m, alpha_m, beta_m, 5.0, dt_sub)
            h_next = _gate_alpha_beta(h, alpha_h, beta_h, 5.0, dt_sub)

            pna_inf = _boltz(v, -48.0, 5.0)
            tau_pna = 5.0 + 20.0 / max(0.01, 1.0 + ((v + 48.0) / 10.0) ** 2)
            p_na_next = _gate_inf(p_na, pna_inf, tau_pna, dt_sub)

            alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
            beta_n = 0.125 * _exp(-(v + 44.0) / 80.0)
            n_next = _gate_alpha_beta(n, alpha_n, beta_n, 5.0, dt_sub)

            a_inf = _boltz(v, -27.0, 16.0)
            b_inf = _boltz(v, -80.0, -6.0)
            a_next = _gate_inf(a, a_inf, 2.0, dt_sub)
            b_next = _gate_inf(b, b_inf, 15.0, dt_sub)

            w_inf = _boltz(v, -35.0, 10.0)
            tau_w = 100.0 / (3.3 * _exp((v + 35.0) / 20.0) + _exp(-(v + 35.0) / 20.0))
            w_next = _gate_inf(w, w_inf, tau_w, dt_sub)

            mt_inf = _boltz(v, -52.0, 5.0)
            s_inf = _boltz(v, -60.0, -6.5)
            tau_s = 20.0 + 50.0 / max(0.01, 1.0 + ((v + 65.0) / 10.0) ** 2)
            m_t_next = _gate_inf(m_t, mt_inf, 1.0, dt_sub)
            s_next = _gate_inf(s, s_inf, tau_s, dt_sub)

            cn_inf = _boltz(v, -20.0, 5.0)
            tau_cn = 2.0 + 10.0 / max(0.01, 1.0 + ((v + 20.0) / 10.0) ** 2)
            c_n_next = _gate_inf(c_n, cn_inf, tau_cn, dt_sub)

            r_inf = _boltz(v, -80.0, -10.0)
            tau_r = 50.0 + 200.0 / max(0.01, 1.0 + ((v + 80.0) / 20.0) ** 2)
            r_next = _gate_inf(r, r_inf, tau_r, dt_sub)

            g_cat = self.g_cat * m_t_next**2 * s_next
            g_can = self.g_can * c_n_next**2
            i_cat = g_cat * (v - self.e_ca)
            i_can = g_can * (v - self.e_ca)
            ca_entry = -(i_cat + i_can) * 0.001 if (i_cat + i_can) < 0.0 else 0.0
            ca_next = _calcium_exact(ca, ca_entry, self.tau_ca, dt_sub)

            ca2 = ca_next**2
            kd2 = self.kd_bk**2
            bk_v = _boltz(v, 100.0 - 120.0 * ca2 / (ca2 + kd2), 15.0)
            sk_inf = ca2 / (ca2 + self.kd_sk**2)

            g_na = self.g_na_t * m_next**3 * h_next + self.g_na_p * p_na_next
            g_k = (
                self.g_kdr * n_next**4
                + self.g_ka * a_next**3 * b_next
                + self.g_km * w_next
                + self.g_bk * bk_v
                + self.g_sk * sk_inf
            )
            g_ca = g_cat + g_can
            g_h = self.g_h * r_next
            g_total = g_na + g_k + g_ca + g_h + self.g_l
            if not _isfinite(g_total) or g_total <= 0.0:
                raise ValueError(
                    f"total membrane conductance must be finite and positive, got {g_total!r}."
                )
            steady_v = (
                input_current
                + g_na * self.e_na
                + g_k * self.e_k
                + g_ca * self.e_ca
                + g_h * self.e_h
                + self.g_l * self.e_l
            ) / g_total
            v_next = steady_v + (v - steady_v) * _exp(-(g_total / self.c_m) * dt_sub)
            if not _voltage(v_next):
                raise ValueError(
                    f"candidate membrane voltage must be finite and within [-100.0, 60.0], "
                    f"got {v_next!r}."
                )

            v = v_next
            m = m_next
            h = h_next
            p_na = p_na_next
            n = n_next
            a = a_next
            b = b_next
            w = w_next
            m_t = m_t_next
            s = s_next
            c_n = c_n_next
            r = r_next
            ca = ca_next

        self.v = v
        self.m = m
        self.h = h
        self.p_na = p_na
        self.n = n
        self.a = a
        self.b = b
        self.w = w
        self.m_t = m_t
        self.s = s
        self.c_n = c_n
        self.r = r
        self.ca = ca

        return 1 if self.v >= 0.0 and v_prev < 0.0 else 0

    def reset(self) -> None:
        """Restore membrane voltage, gating variables, and calcium to resting defaults."""
        self.v = -60.0
        self.m = 0.02
        self.h = 0.85
        self.p_na = 0.01
        self.n = 0.05
        self.a = 0.1
        self.b = 0.8
        self.w = 0.01
        self.m_t = 0.01
        self.s = 0.9
        self.c_n = 0.01
        self.r = 0.1
        self.ca = 0.05
