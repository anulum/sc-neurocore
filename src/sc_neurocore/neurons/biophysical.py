# SPDX-License-Identifier: AGPL-3.0-or-later
"""Biophysical neuron models — ion channel and conductance-based."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ConnorStevensNeuron:
    """Connor-Stevens 1977 — A-type potassium current, Type-I excitability.

    4 ODEs: v, m (Na activation), h (Na inactivation), n (K), a (A-type), b (A-type inactivation).
    """

    v: float = -68.0
    m: float = 0.01
    h: float = 0.99
    n: float = 0.1
    a: float = 0.5
    b: float = 0.1
    g_na: float = 120.0
    g_k: float = 20.0
    g_a: float = 47.7
    g_l: float = 0.3
    e_na: float = 55.0
    e_k: float = -72.0
    e_a: float = -75.0
    e_l: float = -17.0
    c_m: float = 1.0
    dt: float = 0.01
    v_threshold: float = 0.0

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(int(1.0 / max(self.dt, 0.001))):
            am = (
                0.38 * (self.v + 29.7) / (1.0 - np.exp(-(self.v + 29.7) / 10.0))
                if abs(self.v + 29.7) > 1e-6
                else 3.8
            )
            bm = 15.2 * np.exp(-(self.v + 54.7) / 18.0)
            ah = 0.266 * np.exp(-(self.v + 48.0) / 20.0)
            bh = 3.8 / (1.0 + np.exp(-(self.v + 18.0) / 10.0))
            an = (
                0.02 * (self.v + 45.7) / (1.0 - np.exp(-(self.v + 45.7) / 10.0))
                if abs(self.v + 45.7) > 1e-6
                else 0.2
            )
            bn = 0.25 * np.exp(-(self.v + 55.7) / 80.0)

            a_inf = (
                0.0761 * np.exp((self.v + 94.22) / 31.84) / (1.0 + np.exp((self.v + 1.17) / 28.93))
            ) ** (1.0 / 3.0)
            tau_a = 0.3632 + 1.158 / (1.0 + np.exp((self.v + 55.96) / 20.12))
            b_inf = 1.0 / (1.0 + np.exp((self.v + 53.3) / 14.54)) ** 4
            tau_b = 1.24 + 2.678 / (1.0 + np.exp((self.v + 50.0) / 16.027))

            self.m += (am * (1 - self.m) - bm * self.m) * self.dt
            self.h += (ah * (1 - self.h) - bh * self.h) * self.dt
            self.n += (an * (1 - self.n) - bn * self.n) * self.dt
            self.a += ((a_inf - self.a) / tau_a) * self.dt
            self.b += ((b_inf - self.b) / tau_b) * self.dt

            i_na = self.g_na * self.m**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_a = self.g_a * self.a**3 * self.b * (self.v - self.e_a)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_a - i_l + current) / self.c_m * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -68.0
        self.m, self.h, self.n, self.a, self.b = 0.01, 0.99, 0.1, 0.5, 0.1


@dataclass
class WangBuzsakiNeuron:
    """Wang-Buzsáki 1996 — fast-spiking GABAergic interneuron.

    3 ODEs. Simplified HH with only Na + K delayed rectifier.
    Designed for gamma (30-80 Hz) oscillation modelling.
    """

    v: float = -65.0
    h: float = 0.8
    n: float = 0.1
    g_na: float = 35.0
    g_k: float = 9.0
    g_l: float = 0.1
    e_na: float = 55.0
    e_k: float = -90.0
    e_l: float = -65.0
    c_m: float = 1.0
    phi: float = 5.0
    dt: float = 0.01
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(int(0.5 / max(self.dt, 0.001))):
            # m is instantaneous (m_inf)
            alpha_m = (
                0.1 * (self.v + 35.0) / (1.0 - np.exp(-(self.v + 35.0) / 10.0))
                if abs(self.v + 35.0) > 1e-6
                else 1.0
            )
            beta_m = 4.0 * np.exp(-(self.v + 60.0) / 18.0)
            m_inf = alpha_m / (alpha_m + beta_m)

            alpha_h = 0.07 * np.exp(-(self.v + 58.0) / 20.0)
            beta_h = 1.0 / (1.0 + np.exp(-(self.v + 28.0) / 10.0))
            alpha_n = (
                0.01 * (self.v + 34.0) / (1.0 - np.exp(-(self.v + 34.0) / 10.0))
                if abs(self.v + 34.0) > 1e-6
                else 0.1
            )
            beta_n = 0.125 * np.exp(-(self.v + 44.0) / 80.0)

            self.h += self.phi * (alpha_h * (1 - self.h) - beta_h * self.h) * self.dt
            self.n += self.phi * (alpha_n * (1 - self.n) - beta_n * self.n) * self.dt

            i_na = self.g_na * m_inf**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_l + current) / self.c_m * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -65.0
        self.h, self.n = 0.8, 0.1


@dataclass
class PinskyRinzelNeuron:
    """Pinsky-Rinzel 1994 — 2-compartment pyramidal cell.

    Soma (fast Na/K) coupled to dendrite (Ca/KAHP) via gc.
    Minimal model for burst generation in cortical pyramidal cells.
    """

    v_s: float = -60.0
    v_d: float = -60.0
    h: float = 0.9
    n: float = 0.1
    s: float = 0.0
    c: float = 0.0
    q: float = 0.0
    gc: float = 2.1
    p: float = 0.5
    g_na: float = 30.0
    g_kdr: float = 15.0
    g_ca: float = 10.0
    g_kahp: float = 0.8
    g_l: float = 0.1
    e_na: float = 60.0
    e_k: float = -75.0
    e_ca: float = 80.0
    e_l: float = -60.0
    dt: float = 0.02
    v_threshold: float = -20.0

    def step(self, current_soma: float, current_dend: float = 0.0) -> int:
        v_prev = self.v_s
        am = (
            0.32 * (self.v_s + 54.0) / (1.0 - np.exp(-(self.v_s + 54.0) / 4.0))
            if abs(self.v_s + 54.0) > 1e-6
            else 8.0
        )
        bm = (
            0.28 * (self.v_s + 27.0) / (np.exp((self.v_s + 27.0) / 5.0) - 1.0)
            if abs(self.v_s + 27.0) > 1e-6
            else 5.6
        )
        m_inf = am / (am + bm)

        ah = 0.128 * np.exp(-(self.v_s + 50.0) / 18.0)
        bh = 4.0 / (1.0 + np.exp(-(self.v_s + 27.0) / 5.0))
        an = (
            0.032 * (self.v_s + 52.0) / (1.0 - np.exp(-(self.v_s + 52.0) / 5.0))
            if abs(self.v_s + 52.0) > 1e-6
            else 0.32
        )
        bn = 0.5 * np.exp(-(self.v_s + 57.0) / 40.0)

        s_inf = 1.0 / (1.0 + np.exp(-(self.v_d + 20.0) / 9.0))
        c_inf = min(self.c, 1.0) if self.c > 0 else 0.0

        # Soma
        i_na = self.g_na * m_inf**2 * self.h * (self.v_s - self.e_na)
        i_kdr = self.g_kdr * self.n**2 * (self.v_s - self.e_k)
        i_ls = self.g_l * (self.v_s - self.e_l)
        i_ds = (self.gc / self.p) * (self.v_s - self.v_d)

        # Dendrite
        i_ca = self.g_ca * self.s**2 * (self.v_d - self.e_ca)
        i_kahp = self.g_kahp * self.q * (self.v_d - self.e_k)
        i_ld = self.g_l * (self.v_d - self.e_l)
        i_sd = (self.gc / (1 - self.p)) * (self.v_d - self.v_s)

        self.v_s += (-i_na - i_kdr - i_ls - i_ds + current_soma / self.p) * self.dt
        self.v_d += (-i_ca - i_kahp - i_ld - i_sd + current_dend / (1 - self.p)) * self.dt
        self.h += (ah * (1 - self.h) - bh * self.h) * self.dt
        self.n += (an * (1 - self.n) - bn * self.n) * self.dt
        self.s += ((s_inf - self.s) / 5.0) * self.dt
        self.c = max(0.0, self.c + (-0.13 * i_ca - 0.075 * self.c) * self.dt)
        q_inf = min(self.c / (self.c + 2.0), 1.0)
        self.q += ((q_inf - self.q) / 100.0) * self.dt

        return 1 if (self.v_s >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v_s, self.v_d = -60.0, -60.0
        self.h, self.n, self.s, self.c, self.q = 0.9, 0.1, 0.0, 0.0, 0.0


@dataclass
class RulkovMapNeuron:
    """Rulkov 2001 — discrete map-based neuron (no ODE, O(1) per step).

    x[n+1] = f(x[n], y[n]) + I
    y[n+1] = y[n] - μ(x[n] + 1) + μσ
    Fast iteration, exhibits spiking and bursting.
    """

    x: float = -1.0
    y: float = -3.0
    alpha: float = 4.0
    sigma: float = -1.6
    mu: float = 0.001
    x_threshold: float = 0.0

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        if self.x <= 0:
            x_new = self.alpha / (1.0 - self.x) + self.y + current
        elif self.x < self.alpha + self.y + current:
            x_new = self.alpha + self.y + current
        else:
            x_new = -1.0
        y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma
        self.x = x_new
        self.y = y_new
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self):
        self.x, self.y = -1.0, -3.0


@dataclass
class ChialvoMapNeuron:
    """Chialvo 1995 — 2D discrete map neuron.

    x[n+1] = x²·exp(y-x) + k + I
    y[n+1] = a·y - b·x + c
    """

    x: float = 0.0
    y: float = 0.0
    a: float = 0.89
    b: float = 0.6
    c: float = 0.28
    k: float = 0.04
    x_threshold: float = 1.0

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        x_new = self.x**2 * np.exp(self.y - self.x) + self.k + current
        y_new = self.a * self.y - self.b * self.x + self.c
        self.x = x_new
        self.y = y_new
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self):
        self.x, self.y = 0.0, 0.0


@dataclass
class WilsonCowanUnit:
    """Wilson-Cowan 1972 — excitatory/inhibitory population rate model.

    τ_e dE/dt = -E + S(w_ee·E - w_ei·I + I_ext)
    τ_i dI/dt = -I + S(w_ie·E - w_ii·I)
    S(x) = 1/(1 + exp(-a(x-θ))) - 1/(1 + exp(aθ))
    """

    e: float = 0.1
    i: float = 0.05
    w_ee: float = 10.0
    w_ei: float = 6.0
    w_ie: float = 10.0
    w_ii: float = 1.0
    tau_e: float = 1.0
    tau_i: float = 2.0
    a: float = 1.2
    theta: float = 4.0
    dt: float = 0.1

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-self.a * (x - self.theta)))

    def step(self, ext_input: float = 0.0) -> float:
        se = self._sigmoid(self.w_ee * self.e - self.w_ei * self.i + ext_input)
        si = self._sigmoid(self.w_ie * self.e - self.w_ii * self.i)
        self.e += (-self.e + se) / self.tau_e * self.dt
        self.i += (-self.i + si) / self.tau_i * self.dt
        return self.e

    def reset(self):
        self.e, self.i = 0.1, 0.05


@dataclass
class GalvesLocherbachNeuron:
    """Galves-Löcherbach 2013 — stochastic point process neuron.

    P(spike at t | history) = φ(V(t))
    V(t) = Σ w_j · spike_j(past) · decay + leak
    Purely probabilistic, no ODE.
    """

    v: float = 0.0
    v_rest: float = 0.0
    decay: float = 0.95
    threshold_rate: float = 0.5
    steepness: float = 5.0
    dt: float = 1.0

    def _firing_prob(self):
        return 1.0 / (1.0 + np.exp(-self.steepness * (self.v - self.threshold_rate)))

    def step(self, weighted_input: float) -> int:
        self.v = self.decay * self.v + weighted_input
        p = self._firing_prob()
        spike = 1 if np.random.random() < p * self.dt else 0
        if spike:
            self.v = self.v_rest
        return spike

    def reset(self):
        self.v = self.v_rest


@dataclass
class FractionalLIFNeuron:
    """Fractional-order LIF — memory-dependent dynamics.

    Uses Grünwald-Letnikov fractional derivative approximation.
    D^α v(t) = -(v - v_rest) + R·I, where 0 < α ≤ 1.
    α < 1 introduces memory (power-law decay instead of exponential).
    Lundstrom et al. 2008.
    """

    v: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    alpha: float = 0.8
    resistance: float = 1.0
    dt: float = 1.0
    _history: list = None
    _max_history: int = 100

    def __post_init__(self):
        self._history = [0.0] * self._max_history
        self._gl_coeffs = self._compute_gl_coefficients()

    def _compute_gl_coefficients(self):
        coeffs = [1.0]
        for k in range(1, self._max_history):
            coeffs.append(coeffs[-1] * (k - 1 - self.alpha) / k)
        return coeffs

    def step(self, current: float) -> int:
        rhs = -(self.v - self.v_rest) + self.resistance * current
        gl_sum = sum(
            self._gl_coeffs[k] * self._history[-(k + 1)]
            for k in range(1, min(len(self._history), self._max_history))
            if len(self._history) > k
        )
        self.v = rhs * self.dt**self.alpha - gl_sum
        self._history.append(self.v)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self._history[-1] = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self._history = [0.0] * self._max_history
