# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Av-Ron, Parnas & Segel 1993 — cardiac ganglion Type III

"""Av-Ron, Parnas & Segel cardiac ganglion Type III four-state RK4 burster."""

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


def _require_probability(value: object, name: str) -> float:
    """Validate that ``value`` is a finite probability in ``[0.0, 1.0]``."""
    result = _require_finite_float(value, name)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be within [0.0, 1.0], got {value!r}.")
    return result


@dataclass
class AvRonCardiacNeuron:
    """Av-Ron, Parnas & Segel cardiac ganglion Type III burster.

    The four-state conductance model uses instantaneous sodium activation,
    voltage-dependent h/n/s gate relaxation, and candidate-first RK4 integration
    so invalid states cannot partially mutate the neuron.

    Parameters
    ----------
    v : float, default=-60.0
        Initial membrane potential in millivolts.
    h : float, default=0.6
        Initial sodium inactivation gate in ``[0, 1]``.
    n : float, default=0.3
        Initial potassium activation gate in ``[0, 1]``.
    s : float, default=0.5
        Initial slow inward-current gate in ``[0, 1]``.
    g_na : float, default=80.0
        Maximal sodium conductance; must be non-negative.
    g_k : float, default=40.0
        Maximal potassium conductance; must be non-negative.
    g_s : float, default=20.0
        Maximal slow conductance; must be non-negative.
    g_l : float, default=0.1
        Leak conductance; must be non-negative.
    e_na : float, default=40.0
        Sodium reversal potential in millivolts.
    e_k : float, default=-80.0
        Potassium reversal potential in millivolts.
    e_s : float, default=-25.0
        Slow reversal potential in millivolts.
    e_l : float, default=-60.0
        Leak reversal potential in millivolts.
    dt : float, default=0.02
        RK4 integration timestep in milliseconds; must be strictly positive.
    v_threshold : float, default=-20.0
        Upward spike-detection threshold in millivolts.

    Raises
    ------
    TypeError
        If any parameter is a boolean or not a real number.
    ValueError
        If any parameter is non-finite or violates its positivity,
        non-negativity, or ``[0, 1]`` gating domain.
    """

    v: float = -60.0
    h: float = 0.6
    n: float = 0.3
    s: float = 0.5
    g_na: float = 80.0
    g_k: float = 40.0
    g_s: float = 20.0
    g_l: float = 0.1
    e_na: float = 40.0
    e_k: float = -80.0
    e_s: float = -25.0
    e_l: float = -60.0
    dt: float = 0.02
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        """Validate and normalise all initial state variables and parameters."""
        self._validate_runtime()

    @staticmethod
    def _finite_values(values: tuple[float, ...]) -> bool:
        return all(_isfinite(value) for value in values)

    @staticmethod
    def _gate_in_range(value: float) -> bool:
        return 0.0 <= value <= 1.0

    @staticmethod
    def _bounded_exp(value: float) -> float:
        return _exp(max(min(value, 709.0), -745.0))

    @classmethod
    def _sigmoid_pos(cls, value: float) -> float:
        return 1.0 / (1.0 + cls._bounded_exp(-value))

    @classmethod
    def _sigmoid_neg(cls, value: float) -> float:
        return 1.0 / (1.0 + cls._bounded_exp(value))

    def _validate_runtime(self) -> None:
        """Validate and normalise all floating-point state and parameter fields."""
        self.v = _require_finite_float(self.v, "v")
        self.h = _require_probability(self.h, "h")
        self.n = _require_probability(self.n, "n")
        self.s = _require_probability(self.s, "s")
        self.g_na = _require_non_negative_float(self.g_na, "g_na")
        self.g_k = _require_non_negative_float(self.g_k, "g_k")
        self.g_s = _require_non_negative_float(self.g_s, "g_s")
        self.g_l = _require_non_negative_float(self.g_l, "g_l")
        self.e_na = _require_finite_float(self.e_na, "e_na")
        self.e_k = _require_finite_float(self.e_k, "e_k")
        self.e_s = _require_finite_float(self.e_s, "e_s")
        self.e_l = _require_finite_float(self.e_l, "e_l")
        self.dt = _require_positive_float(self.dt, "dt")
        self.v_threshold = _require_finite_float(self.v_threshold, "v_threshold")

    def _rates(self, voltage: float) -> tuple[float, float, float, float, float, float, float]:
        m_inf = self._sigmoid_pos((voltage + 40.0) / 7.0)
        h_inf = self._sigmoid_neg((voltage + 45.0) / 5.0)
        n_inf = self._sigmoid_pos((voltage + 40.0) / 15.0)
        s_inf = self._sigmoid_neg((voltage + 35.0) / 3.0)
        tau_h = 1.0 + 12.0 * self._sigmoid_neg((voltage + 50.0) / 8.0)
        tau_n = 1.0 + 8.0 * self._sigmoid_neg((voltage + 35.0) / 8.0)
        tau_s = 200.0 + 1000.0 * self._sigmoid_neg((voltage + 30.0) / 5.0)
        return m_inf, h_inf, n_inf, s_inf, tau_h, tau_n, tau_s

    def _derivatives(
        self, state: tuple[float, float, float, float], current: float
    ) -> tuple[float, float, float, float]:
        voltage, h_gate, n_gate, s_gate = state
        if not self._finite_values(state) or not (
            self._gate_in_range(h_gate)
            and self._gate_in_range(n_gate)
            and self._gate_in_range(s_gate)
        ):
            return (math.nan, math.nan, math.nan, math.nan)
        m_inf, h_inf, n_inf, s_inf, tau_h, tau_n, tau_s = self._rates(voltage)
        i_na = self.g_na * m_inf**3 * h_gate * (voltage - self.e_na)
        i_k = self.g_k * n_gate**4 * (voltage - self.e_k)
        i_s = self.g_s * s_gate * (voltage - self.e_s)
        i_l = self.g_l * (voltage - self.e_l)
        return (
            -i_na - i_k - i_s - i_l + current,
            (h_inf - h_gate) / tau_h,
            (n_inf - n_gate) / tau_n,
            (s_inf - s_gate) / tau_s,
        )

    @staticmethod
    def _add_scaled(
        state: tuple[float, float, float, float],
        slope: tuple[float, float, float, float],
        scale: float,
    ) -> tuple[float, float, float, float]:
        return (
            state[0] + scale * slope[0],
            state[1] + scale * slope[1],
            state[2] + scale * slope[2],
            state[3] + scale * slope[3],
        )

    def _rk4_candidate(self, current: float) -> tuple[float, float, float, float] | None:
        state = (self.v, self.h, self.n, self.s)
        half_dt = 0.5 * self.dt
        k1 = self._derivatives(state, current)
        k2 = self._derivatives(self._add_scaled(state, k1, half_dt), current)
        k3 = self._derivatives(self._add_scaled(state, k2, half_dt), current)
        k4 = self._derivatives(self._add_scaled(state, k3, self.dt), current)
        candidate = (
            state[0] + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + self.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + self.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        )
        if self._finite_values(candidate) and all(
            self._gate_in_range(value) for value in candidate[1:]
        ):
            return candidate
        return None

    def step(self, current: float) -> int:
        """Advance membrane potential and h/n/s gates by one RK4 step.

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
            any intermediate or final RK4 candidate is non-finite or escapes
            the ``[0, 1]`` gating range; state is unchanged on error.
        """
        v = self.v
        h = self.h
        n = self.n
        s = self.s
        g_na = self.g_na
        g_k = self.g_k
        g_s = self.g_s
        g_l = self.g_l
        e_na = self.e_na
        e_k = self.e_k
        e_s = self.e_s
        e_l = self.e_l
        dt = self.dt
        v_threshold = self.v_threshold
        if not (
            current.__class__ is float
            and v.__class__ is float
            and h.__class__ is float
            and n.__class__ is float
            and s.__class__ is float
            and g_na.__class__ is float
            and g_k.__class__ is float
            and g_s.__class__ is float
            and g_l.__class__ is float
            and e_na.__class__ is float
            and e_k.__class__ is float
            and e_s.__class__ is float
            and e_l.__class__ is float
            and dt.__class__ is float
            and v_threshold.__class__ is float
            and (
                current
                + v
                + h
                + n
                + s
                + g_na
                + g_k
                + g_s
                + g_l
                + e_na
                + e_k
                + e_s
                + e_l
                + dt
                + v_threshold
            )
            * 0.0
            == 0.0
            and dt > 0.0
            and g_na >= 0.0
            and g_k >= 0.0
            and g_s >= 0.0
            and g_l >= 0.0
            and 0.0 <= h <= 1.0
            and 0.0 <= n <= 1.0
            and 0.0 <= s <= 1.0
        ):
            current = _require_finite_float(current, "current")
            self._validate_runtime()

        v_prev = self.v
        candidate = self._rk4_candidate(current)
        if candidate is None:
            raise ValueError(
                "RK4 candidate state is non-finite or a gating variable escaped the [0, 1] range."
            )
        self.v, self.h, self.n, self.s = candidate
        return int(self.v >= self.v_threshold and v_prev < self.v_threshold)

    def reset(self) -> None:
        """Restore membrane potential and h/n/s gates to resting defaults."""
        self.v = -60.0
        self.h = 0.6
        self.n = 0.3
        self.s = 0.5
