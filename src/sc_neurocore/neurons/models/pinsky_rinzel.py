# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pinsky-Rinzel 1994 — 2-compartment CA3 pyramidal cell

from __future__ import annotations

import math
from dataclasses import dataclass

_STATE_NAMES = ("v_s", "v_d", "h", "n", "s", "c", "q", "ca")
_PARAM_NAMES = (
    "cm",
    "gc",
    "p",
    "g_na",
    "g_kdr",
    "g_ca",
    "g_kahp",
    "g_kc",
    "g_l",
    "e_na",
    "e_k",
    "e_ca",
    "e_l",
    "dt",
    "v_threshold",
)
_STRICTLY_POSITIVE_PARAMS = ("cm", "gc", "g_na", "g_kdr", "g_ca", "g_kahp", "g_kc", "g_l", "dt")
_GATE_NAMES = ("h", "n", "s", "c", "q")

# State packed as (v_s, v_d, h, n, s, c, q, ca) for the RK4 stepper.
_State = tuple[float, float, float, float, float, float, float, float]


@dataclass
class PinskyRinzelNeuron:
    """Pinsky-Rinzel 1994 two-compartment CA3 pyramidal cell.

    Reduction of the 19-compartment Traub CA3 model to a soma compartment
    carrying the fast Na⁺/delayed-rectifier K⁺ spike currents and a dendrite
    compartment carrying the Ca²⁺, Ca-dependent K⁺ afterhyperpolarisation, and
    voltage/Ca-dependent K⁺ currents, coupled by ``gc``. Eight states are
    integrated with a fixed-step fourth-order Runge-Kutta scheme; the somatic
    sodium activation ``m`` is taken at its instantaneous steady state ``m∞``.

    The voltages use the physiological convention (rest ≈ −60 mV); reversal
    potentials and gating rates equal the original rest=0 mV formulation shifted
    by −60 mV. Parameters and kinetics follow the published model and the
    ModelDB 35358 reference channels.

    Parameters
    ----------
    v_s, v_d : float
        Somatic and dendritic membrane potential (mV).
    h, n, s, c, q : float
        Gating variables in [0, 1]: Na⁺ inactivation ``h``, delayed-rectifier
        activation ``n``, Ca²⁺ activation ``s``, voltage/Ca-dependent K⁺
        activation ``c``, and Ca-dependent afterhyperpolarisation ``q``.
    ca : float
        Dimensionless dendritic calcium concentration (≥ 0).
    cm : float
        Membrane capacitance (µF/cm²); default 3.0 per Pinsky & Rinzel (1994).
    gc : float
        Soma-dendrite coupling conductance (mS/cm²).
    p : float
        Somatic membrane-area fraction in (0, 1).
    g_na, g_kdr, g_ca, g_kahp, g_kc, g_l : float
        Maximal conductances (mS/cm²) for the Na⁺, K-DR, Ca²⁺, K-AHP, K-C, and
        leak currents.
    e_na, e_k, e_ca, e_l : float
        Reversal potentials (mV).
    dt : float
        Integration time step (ms).
    v_threshold : float
        Somatic voltage at which a spike is registered (mV).

    References
    ----------
    Pinsky, P.F. & Rinzel, J. (1994). Intrinsic and network rhythmogenesis in a
    reduced Traub model for CA3 neurons. J. Comput. Neurosci. 1:39–60.
    doi:10.1007/BF00962717. Reference channel kinetics: ModelDB accession 35358.
    """

    v_s: float = -60.0
    v_d: float = -60.0
    h: float = 0.999
    n: float = 0.001
    s: float = 0.009
    c: float = 0.007
    q: float = 0.01
    ca: float = 0.2
    cm: float = 3.0
    gc: float = 2.1
    p: float = 0.5
    g_na: float = 30.0
    g_kdr: float = 15.0
    g_ca: float = 10.0
    g_kahp: float = 0.8
    g_kc: float = 15.0
    g_l: float = 0.1
    e_na: float = 60.0
    e_k: float = -75.0
    e_ca: float = 80.0
    e_l: float = -60.0
    dt: float = 0.02
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
        for name in _STRICTLY_POSITIVE_PARAMS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.p < 1.0:
            raise ValueError("p must be in (0, 1)")
        if self.ca < 0.0:
            raise ValueError("ca must be non-negative")
        for name in _GATE_NAMES:
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} gate must remain in [0, 1]")

    @staticmethod
    def _exp(value: float) -> float:
        """Return ``exp(value)``, failing closed on overflow/non-finite results."""
        try:
            out = math.exp(value)
        except OverflowError as exc:
            raise FloatingPointError("Pinsky-Rinzel rate exponential overflowed") from exc
        if not math.isfinite(out):
            raise FloatingPointError("Pinsky-Rinzel rate exponential became non-finite")
        return out

    @classmethod
    def _exprel_minus(cls, a: float, dv: float, k: float) -> float:
        """Evaluate the Traub rate ``a·dv / (1 − exp(−dv/k))`` with its limit.

        This is the activation-type ``α`` form; at ``dv → 0`` it has the finite
        removable limit ``a·k`` (since ``1 − exp(−x) ≈ x``).
        """
        if abs(dv) < 1e-6:
            return a * k
        return a * dv / (1.0 - cls._exp(-dv / k))

    @classmethod
    def _exprel_plus(cls, a: float, dv: float, k: float) -> float:
        """Evaluate the Traub rate ``a·dv / (exp(dv/k) − 1)`` with its limit.

        This is the deactivation-type ``β`` form; at ``dv → 0`` it has the finite
        removable limit ``a·k`` (since ``exp(x) − 1 ≈ x``).
        """
        if abs(dv) < 1e-6:
            return a * k
        return a * dv / (cls._exp(dv / k) - 1.0)

    def _derivatives(self, state: _State, i_s: float, i_d: float) -> _State:
        """Return the time derivatives of the eight-dimensional state.

        Parameters
        ----------
        state : tuple of float
            Packed ``(v_s, v_d, h, n, s, c, q, ca)``.
        i_s, i_d : float
            Somatic and dendritic injected current (µA/cm²).

        Returns
        -------
        tuple of float
            ``(dv_s, dv_d, dh, dn, ds, dc, dq, dca)``.
        """
        v_s, v_d, h, n, s, c, q, ca = state

        # Soma fast currents.
        am = self._exprel_minus(0.32, v_s + 46.9, 4.0)
        bm = self._exprel_plus(0.28, v_s + 19.9, 5.0)
        rate_sum = am + bm
        if rate_sum > 0.0:
            m_inf = am / rate_sum
        else:  # pragma: no cover - defensive: alpha_m, beta_m are strictly positive rates
            m_inf = 0.0
        ah = 0.128 * self._exp(-(v_s + 43.0) / 18.0)
        bh = 4.0 / (1.0 + self._exp(-(v_s + 20.0) / 5.0))
        an = self._exprel_minus(0.016, v_s + 24.9, 5.0)
        bn = 0.25 * self._exp(-1.0 - 0.025 * v_s)

        # Dendrite slow currents.
        a_s = 1.6 / (1.0 + self._exp(-0.072 * (v_d - 5.0)))
        b_s = self._exprel_plus(0.02, v_d + 8.9, 5.0)
        if v_d <= -10.0:
            ac = self._exp((v_d + 50.0) / 11.0 - (v_d + 53.5) / 27.0) / 18.975
            bc = 2.0 * self._exp((-53.5 - v_d) / 27.0) - ac
        else:
            ac = 2.0 * self._exp((-53.5 - v_d) / 27.0)
            bc = 0.0
        aq = min(0.00002 * ca, 0.01)
        bq = 0.001
        chi = min(ca / 250.0, 1.0)

        i_na = self.g_na * m_inf**2 * h * (v_s - self.e_na)
        i_kdr = self.g_kdr * n * (v_s - self.e_k)
        i_ls = self.g_l * (v_s - self.e_l)
        i_ca = self.g_ca * s**2 * (v_d - self.e_ca)
        i_kahp = self.g_kahp * q * (v_d - self.e_k)
        i_kc = self.g_kc * c * chi * (v_d - self.e_k)
        i_ld = self.g_l * (v_d - self.e_l)
        i_coupling = self.gc * (v_d - v_s)

        dv_s = (-i_ls - i_na - i_kdr + i_coupling / self.p + i_s / self.p) / self.cm
        dv_d = (
            -i_ld - i_ca - i_kahp - i_kc - i_coupling / (1.0 - self.p) + i_d / (1.0 - self.p)
        ) / self.cm
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n) - bn * n
        ds = a_s * (1.0 - s) - b_s * s
        dc = ac * (1.0 - c) - bc * c
        dq = aq * (1.0 - q) - bq * q
        dca = -0.13 * i_ca - 0.075 * ca
        return dv_s, dv_d, dh, dn, ds, dc, dq, dca

    @staticmethod
    def _axpy(state: _State, deriv: _State, factor: float) -> _State:
        """Return ``state + factor·deriv`` componentwise for the RK4 stages."""
        return (
            state[0] + factor * deriv[0],
            state[1] + factor * deriv[1],
            state[2] + factor * deriv[2],
            state[3] + factor * deriv[3],
            state[4] + factor * deriv[4],
            state[5] + factor * deriv[5],
            state[6] + factor * deriv[6],
            state[7] + factor * deriv[7],
        )

    def _rk4(self, state: _State, i_s: float, i_d: float) -> _State:
        """Advance ``state`` one ``dt`` with classical fourth-order Runge-Kutta."""
        dt = self.dt
        k1 = self._derivatives(state, i_s, i_d)
        k2 = self._derivatives(self._axpy(state, k1, dt / 2.0), i_s, i_d)
        k3 = self._derivatives(self._axpy(state, k2, dt / 2.0), i_s, i_d)
        k4 = self._derivatives(self._axpy(state, k3, dt), i_s, i_d)
        return tuple(  # type: ignore[return-value]
            state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(8)
        )

    @staticmethod
    def _validate_candidate(state: _State) -> _State:
        """Clamp gates to [0, 1] and calcium to ≥ 0, failing closed on non-finite state."""
        if not all(math.isfinite(value) for value in state):
            raise FloatingPointError("Pinsky-Rinzel candidate state became non-finite")
        v_s, v_d, h, n, s, c, q, ca = state
        return (
            v_s,
            v_d,
            min(max(h, 0.0), 1.0),
            min(max(n, 0.0), 1.0),
            min(max(s, 0.0), 1.0),
            min(max(c, 0.0), 1.0),
            min(max(q, 0.0), 1.0),
            max(ca, 0.0),
        )

    def step(self, current_soma: float, current_dend: float = 0.0) -> int:
        """Advance one ``dt`` and return 1 on a rising somatic threshold crossing.

        Parameters
        ----------
        current_soma : float
            Somatic injected current density (µA/cm²).
        current_dend : float, optional
            Dendritic injected current density (µA/cm²); default 0.

        Returns
        -------
        int
            1 if ``v_s`` crossed ``v_threshold`` upward on this step, else 0.

        Raises
        ------
        ValueError
            If either current is non-finite or the configuration is invalid.
        FloatingPointError
            If the integrated state becomes non-finite.
        """
        current_soma = float(current_soma)
        current_dend = float(current_dend)
        if not math.isfinite(current_soma) or not math.isfinite(current_dend):
            raise ValueError("current_soma and current_dend must be finite")
        self._validate_configuration()

        v_prev = self.v_s
        state: _State = (self.v_s, self.v_d, self.h, self.n, self.s, self.c, self.q, self.ca)
        nxt = self._validate_candidate(self._rk4(state, current_soma, current_dend))
        self.v_s, self.v_d, self.h, self.n, self.s, self.c, self.q, self.ca = nxt

        return 1 if (self.v_s >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self) -> None:
        """Restore the published resting initial condition."""
        self.v_s, self.v_d = -60.0, -60.0
        self.h, self.n, self.s, self.c, self.q, self.ca = 0.999, 0.001, 0.009, 0.007, 0.01, 0.2
