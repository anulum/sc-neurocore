# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Liu-Golowasch-Marder-Abbott 1998 STG neuron

from __future__ import annotations

import math
from dataclasses import dataclass

_GATES = ("m_na", "h_na", "m_cat", "h_cat", "m_cas", "h_cas", "m_a", "h_a", "m_kca", "m_kd", "m_h")
_STATE_NAMES = ("v", *_GATES, "ca")
_PARAM_NAMES = (
    "cm",
    "g_na",
    "g_cat",
    "g_cas",
    "g_a",
    "g_kca",
    "g_kd",
    "g_h",
    "g_l",
    "e_na",
    "e_k",
    "e_h",
    "e_l",
    "ca_out",
    "ca_rest",
    "tau_ca",
    "f_ca",
    "celsius",
    "dt",
    "v_threshold",
)
_STRICTLY_POSITIVE = ("cm", "ca_out", "tau_ca", "dt")
_NON_NEGATIVE_G = ("g_na", "g_cat", "g_cas", "g_a", "g_kca", "g_kd", "g_h", "g_l")

# State packed as (v, m_na, h_na, m_cat, h_cat, m_cas, h_cas, m_a, h_a, m_kca, m_kd, m_h, ca).
_State = tuple[
    float, float, float, float, float, float, float, float, float, float, float, float, float
]
_N = 13


def _exp(x: float) -> float:
    """Overflow-safe ``exp``: the argument is clamped to ``[-700, 700]``.

    Rate and steady-state functions saturate well within this range, so clamping
    keeps intermediate Runge-Kutta evaluations at extreme voltages finite (the
    step then fails closed on any genuinely non-finite committed state).
    """
    return math.exp(min(max(x, -700.0), 700.0))


@dataclass
class MarderSTGNeuron:
    """Liu-Golowasch-Marder-Abbott 1998 stomatogastric ganglion neuron.

    Single-compartment crustacean STG model with seven voltage-gated currents
    (Na, CaT, CaS, A, KCa, Kd, H) plus a leak, in the Prinz/LGMA unit convention
    (conductances in mS/cm², capacitance in µF/cm², calcium in µM, voltage in mV,
    time in ms). All gates use the published voltage-dependent steady states and
    time constants; the calcium reversal is computed from the Nernst equation,
    and intracellular calcium relaxes towards rest with a 20 ms time constant
    driven by the two calcium currents. The thirteen-state vector is integrated
    with classical fourth-order Runge-Kutta.

    Parameters
    ----------
    v : float
        Membrane potential (mV).
    m_na, h_na, m_cat, h_cat, m_cas, h_cas, m_a, h_a, m_kca, m_kd, m_h : float
        Gating variables in [0, 1].
    ca : float
        Intracellular calcium concentration (µM, ≥ 0).
    cm : float
        Specific membrane capacitance (µF/cm²).
    g_na, g_cat, g_cas, g_a, g_kca, g_kd, g_h, g_l : float
        Maximal conductances (mS/cm²).
    e_na, e_k, e_h, e_l : float
        Reversal potentials (mV). The calcium reversal is Nernst-derived.
    ca_out : float
        Extracellular calcium concentration (µM) for the Nernst equation.
    ca_rest : float
        Resting calcium concentration (µM).
    tau_ca : float
        Calcium relaxation time constant (ms).
    f_ca : float
        Calcium-current-to-concentration coupling (µM·cm²/µA).
    celsius : float
        Temperature (°C) for the Nernst equation.
    dt : float
        Integration time step (ms).
    v_threshold : float
        Voltage at which a spike is registered (mV).

    References
    ----------
    Liu, Z., Golowasch, J., Marder, E. & Abbott, L.F. (1998). A model neuron with
    activity-dependent conductances regulated by multiple calcium sensors.
    J. Neurosci. 18(7):2309–2320. Channel kinetics: ModelDB accession 93321.
    """

    v: float = -60.0
    m_na: float = 0.0
    h_na: float = 1.0
    m_cat: float = 0.0
    h_cat: float = 1.0
    m_cas: float = 0.0
    h_cas: float = 1.0
    m_a: float = 0.0
    h_a: float = 1.0
    m_kca: float = 0.0
    m_kd: float = 0.0
    m_h: float = 0.0
    ca: float = 0.05
    cm: float = 1.0
    g_na: float = 200.0
    g_cat: float = 2.5
    g_cas: float = 4.0
    g_a: float = 50.0
    g_kca: float = 25.0
    g_kd: float = 75.0
    g_h: float = 0.01
    g_l: float = 0.01
    e_na: float = 50.0
    e_k: float = -80.0
    e_h: float = -20.0
    e_l: float = -50.0
    ca_out: float = 3000.0
    ca_rest: float = 0.05
    tau_ca: float = 20.0
    f_ca: float = 0.94
    celsius: float = 10.0
    dt: float = 0.05
    v_threshold: float = -20.0

    def __post_init__(self) -> None:
        self._validate_configuration(coerce=True)

    def _validate_configuration(self, *, coerce: bool = False) -> None:
        for name in (*_STATE_NAMES, *_PARAM_NAMES):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if coerce:
                setattr(self, name, value)
        for name in _STRICTLY_POSITIVE:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in _NON_NEGATIVE_G:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.ca < 0.0:
            raise ValueError("ca must be non-negative")
        for name in _GATES:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} gate must remain in [0, 1]")

    @staticmethod
    def _sigmoid(v: float, v_half: float, slope: float) -> float:
        """Boltzmann steady state ``1 / (1 + exp((v_half − v)/slope))``."""
        return 1.0 / (1.0 + _exp((v_half - v) / slope))

    def _nernst_e_ca(self, ca: float) -> float:
        """Nernst calcium reversal (mV) at the configured temperature."""
        rt_zf = 1000.0 * 8.314462618 * (self.celsius + 273.15) / (2.0 * 96485.33212)
        return rt_zf * math.log(self.ca_out / max(ca, 1e-9))

    def _derivatives(self, state: _State, current: float) -> _State:
        """Return d/dt of the thirteen-state vector under injected ``current``."""
        v, m_na, h_na, m_cat, h_cat, m_cas, h_cas, m_a, h_a, m_kca, m_kd, m_h, ca = state

        m_na_inf = self._sigmoid(v, -25.5, 5.29)
        tau_m_na = 1.32 - 1.26 / (1.0 + _exp(-(v + 120.0) / 25.0))
        h_na_inf = self._sigmoid(v, -48.9, -5.18)
        tau_h_na = (0.67 / (1.0 + _exp(-(v + 62.9) / 10.0))) * (
            1.5 + 1.0 / (1.0 + _exp((v + 34.9) / 3.6))
        )
        m_cat_inf = self._sigmoid(v, -27.1, 7.2)
        tau_m_cat = 21.7 - 21.3 / (1.0 + _exp(-(v + 68.1) / 20.5))
        h_cat_inf = self._sigmoid(v, -32.1, -5.5)
        tau_h_cat = 105.0 - 89.8 / (1.0 + _exp(-(v + 55.0) / 16.9))
        m_cas_inf = self._sigmoid(v, -33.0, 8.1)
        tau_m_cas = 1.4 + 7.0 / (_exp((v + 27.0) / 10.0) + _exp(-(v + 70.0) / 13.0))
        h_cas_inf = self._sigmoid(v, -60.0, -6.2)
        tau_h_cas = 60.0 + 150.0 / (_exp((v + 55.0) / 9.0) + _exp(-(v + 65.0) / 16.0))
        m_a_inf = self._sigmoid(v, -27.2, 8.7)
        tau_m_a = 11.6 - 10.4 / (1.0 + _exp(-(v + 32.9) / 15.2))
        h_a_inf = self._sigmoid(v, -56.9, -4.9)
        tau_h_a = 38.6 - 29.2 / (1.0 + _exp(-(v + 38.9) / 26.5))
        m_kca_inf = (ca / (ca + 3.0)) * self._sigmoid(v, -28.3, 12.6)
        tau_m_kca = 90.3 - 75.1 / (1.0 + _exp(-(v + 46.0) / 22.7))
        m_kd_inf = self._sigmoid(v, -12.3, 11.8)
        tau_m_kd = 7.2 - 6.4 / (1.0 + _exp(-(v + 28.3) / 19.2))
        m_h_inf = self._sigmoid(v, -70.0, -6.0)
        tau_m_h = 272.0 + 1499.0 / (1.0 + _exp(-(v + 42.2) / 8.73))

        e_ca = self._nernst_e_ca(ca)
        i_na = self.g_na * m_na**3 * h_na * (v - self.e_na)
        i_cat = self.g_cat * m_cat**3 * h_cat * (v - e_ca)
        i_cas = self.g_cas * m_cas**3 * h_cas * (v - e_ca)
        i_a = self.g_a * m_a**3 * h_a * (v - self.e_k)
        i_kca = self.g_kca * m_kca**4 * (v - self.e_k)
        i_kd = self.g_kd * m_kd**4 * (v - self.e_k)
        i_h = self.g_h * m_h * (v - self.e_h)
        i_l = self.g_l * (v - self.e_l)

        dv = (current - i_na - i_cat - i_cas - i_a - i_kca - i_kd - i_h - i_l) / self.cm
        dca = (-self.f_ca * (i_cat + i_cas) - (ca - self.ca_rest)) / self.tau_ca
        return (
            dv,
            (m_na_inf - m_na) / tau_m_na,
            (h_na_inf - h_na) / tau_h_na,
            (m_cat_inf - m_cat) / tau_m_cat,
            (h_cat_inf - h_cat) / tau_h_cat,
            (m_cas_inf - m_cas) / tau_m_cas,
            (h_cas_inf - h_cas) / tau_h_cas,
            (m_a_inf - m_a) / tau_m_a,
            (h_a_inf - h_a) / tau_h_a,
            (m_kca_inf - m_kca) / tau_m_kca,
            (m_kd_inf - m_kd) / tau_m_kd,
            (m_h_inf - m_h) / tau_m_h,
            dca,
        )

    @staticmethod
    def _axpy(state: _State, deriv: _State, factor: float) -> _State:
        """Return ``state + factor·deriv`` componentwise for the RK4 stages."""
        return tuple(state[i] + factor * deriv[i] for i in range(_N))  # type: ignore[return-value]

    def _rk4(self, state: _State, current: float) -> _State:
        """Advance ``state`` one ``dt`` with classical fourth-order Runge-Kutta."""
        dt = self.dt
        k1 = self._derivatives(state, current)
        k2 = self._derivatives(self._axpy(state, k1, dt / 2.0), current)
        k3 = self._derivatives(self._axpy(state, k2, dt / 2.0), current)
        k4 = self._derivatives(self._axpy(state, k3, dt), current)
        return tuple(  # type: ignore[return-value]
            state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(_N)
        )

    @staticmethod
    def _commit(state: _State) -> _State:
        """Clamp gates to [0, 1] and calcium to ≥ 0, failing closed on non-finite state."""
        if not all(math.isfinite(value) for value in state):
            raise FloatingPointError("Marder-STG candidate state became non-finite")
        v = state[0]
        gates = tuple(min(max(value, 0.0), 1.0) for value in state[1:12])
        ca = max(state[12], 0.0)
        return (v, *gates, ca)  # type: ignore[return-value]

    def step(self, current: float) -> int:
        """Advance one ``dt`` and return 1 on an upward threshold crossing.

        Parameters
        ----------
        current : float
            Injected current density (µA/cm²).

        Returns
        -------
        int
            1 if ``v`` crossed ``v_threshold`` upward on this step, else 0.

        Raises
        ------
        ValueError
            If the current is non-finite or the configuration is invalid.
        FloatingPointError
            If the integrated state becomes non-finite.
        """
        current = float(current)
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_configuration()

        v_prev = self.v
        state: _State = (
            self.v,
            self.m_na,
            self.h_na,
            self.m_cat,
            self.h_cat,
            self.m_cas,
            self.h_cas,
            self.m_a,
            self.h_a,
            self.m_kca,
            self.m_kd,
            self.m_h,
            self.ca,
        )
        nxt = self._commit(self._rk4(state, current))
        (
            self.v,
            self.m_na,
            self.h_na,
            self.m_cat,
            self.h_cat,
            self.m_cas,
            self.h_cas,
            self.m_a,
            self.h_a,
            self.m_kca,
            self.m_kd,
            self.m_h,
            self.ca,
        ) = nxt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        """Restore the resting initial condition."""
        self.v = -60.0
        self.m_na, self.h_na = 0.0, 1.0
        self.m_cat, self.h_cat = 0.0, 1.0
        self.m_cas, self.h_cas = 0.0, 1.0
        self.m_a, self.h_a = 0.0, 1.0
        self.m_kca, self.m_kd, self.m_h = 0.0, 0.0, 0.0
        self.ca = 0.05
