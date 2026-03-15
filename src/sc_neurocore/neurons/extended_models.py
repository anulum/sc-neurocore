# SPDX-License-Identifier: AGPL-3.0-or-later
"""Extended neuron model library — every published model in one file.

Models grouped by category. Each has: step(current) -> int (spike 0/1),
reset(), and documented reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# ── BIOPHYSICAL: Ion Channel Models ────────────────────────────────


@dataclass
class DestexheThalamicNeuron:
    """Destexhe 1993 — thalamocortical relay with T-current and I_h.

    6 ODEs: V, m_Na, h_Na, n_K, m_T, h_T (+ optional h-current).
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

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(5):
            m_na_inf = 1.0 / (1.0 + np.exp(-(self.v + 37.0) / 7.0))
            h_na_inf = 1.0 / (1.0 + np.exp((self.v + 41.0) / 4.0))
            n_k_inf = 1.0 / (1.0 + np.exp(-(self.v + 25.0) / 12.0))
            m_t_inf = 1.0 / (1.0 + np.exp(-(self.v + 57.0) / 6.5))
            h_t_inf = 1.0 / (1.0 + np.exp((self.v + 81.0) / 4.0))

            tau_h_na = 1.0 / (
                0.128 * np.exp(-(self.v + 46.0) / 18.0)
                + 4.0 / (1.0 + np.exp(-(self.v + 23.0) / 5.0))
            )
            tau_n_k = 1.0 / (0.032 * 5.0 + 0.5 * np.exp(-(self.v + 40.0) / 40.0)) if True else 1.0
            tau_h_t = (
                30.8
                + 211.4 * np.exp((self.v + 115.2) / 5.0) / (1.0 + np.exp((self.v + 86.0) / 3.2))
                if self.v < -81.0
                else 10.0
            )

            self.h_na += (h_na_inf - self.h_na) / max(tau_h_na, 0.1) * self.dt
            self.n_k += (n_k_inf - self.n_k) / max(tau_n_k, 0.1) * self.dt
            self.m_t = m_t_inf
            self.h_t += (h_t_inf - self.h_t) / max(tau_h_t, 0.1) * self.dt

            i_na = self.g_na * m_na_inf**3 * self.h_na * (self.v - self.e_na)
            i_k = self.g_k * self.n_k**4 * (self.v - self.e_k)
            i_t = self.g_t * self.m_t**2 * self.h_t * (self.v - self.e_ca)
            i_l = self.g_l * (self.v - self.e_l)
            self.v += (-i_na - i_k - i_t - i_l + current) * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -65.0
        self.h_na, self.n_k, self.m_t, self.h_t = 0.6, 0.3, 0.0, 1.0


@dataclass
class HuberBraunNeuron:
    """Braun, Huber et al. 1998 — cold receptor, temperature-dependent.

    4 ODEs: V, a_sd (slow depolarizing), a_sr (slow repolarizing), a_r.
    """

    v: float = -50.0
    a_sd: float = 0.0
    a_sr: float = 0.0
    g_sd: float = 1.5
    g_sr: float = 0.4
    g_l: float = 0.1
    e_sd: float = 50.0
    e_sr: float = -90.0
    e_l: float = -60.0
    tau_sd: float = 10.0
    tau_sr: float = 20.0
    eta: float = 0.012
    dt: float = 0.1
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        sd_inf = 1.0 / (1.0 + np.exp(-(self.v + 40.0) / 6.0))
        sr_inf = 1.0 / (1.0 + np.exp((self.v + 40.0) / 6.0))
        self.a_sd += (sd_inf - self.a_sd) / self.tau_sd * self.dt
        self.a_sr += (sr_inf - self.a_sr) / self.tau_sr * self.dt
        i_sd = self.g_sd * self.a_sd * (self.v - self.e_sd)
        i_sr = self.g_sr * self.a_sr * (self.v - self.e_sr)
        i_l = self.g_l * (self.v - self.e_l)
        self.v += (-i_sd - i_sr - i_l + current + self.eta * np.random.randn()) * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -50.0
        self.a_sd, self.a_sr = 0.0, 0.0


@dataclass
class GutkinErmentroutNeuron:
    """Gutkin & Ermentrout 1998 — persistent Na + K minimal conductance."""

    v: float = -65.0
    n: float = 0.1
    g_na: float = 20.0
    g_k: float = 10.0
    g_l: float = 8.0
    e_na: float = 60.0
    e_k: float = -90.0
    e_l: float = -80.0
    dt: float = 0.05
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = 1.0 / (1.0 + np.exp(-(self.v + 20.0) / 15.0))
        n_inf = 1.0 / (1.0 + np.exp(-(self.v + 25.0) / 5.0))
        tau_n = 1.0
        self.n += (n_inf - self.n) / tau_n * self.dt
        i_na = self.g_na * m_inf * (self.v - self.e_na)
        i_k = self.g_k * self.n * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)
        self.v += (-i_na - i_k - i_l + current) * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -65.0
        self.n = 0.1


# ── BURSTING MODELS ────────────────────────────────────────────────


@dataclass
class FitzHughRinzelNeuron:
    """FitzHugh 1976 / Rinzel 1987 — FHN + slow variable for bursting."""

    v: float = -1.0
    w: float = -0.5
    y: float = 0.0
    a: float = 0.7
    b: float = 0.8
    c: float = -0.775
    d: float = 1.0
    delta: float = 0.08
    mu: float = 0.0001
    dt: float = 0.1
    v_threshold: float = 1.0

    def step(self, current: float) -> int:
        v_prev = self.v
        dv = (self.v - self.v**3 / 3.0 - self.w + self.y + current) * self.dt
        dw = self.delta * (self.a + self.v - self.b * self.w) * self.dt
        dy = self.mu * (self.c - self.v - self.d * self.y) * self.dt
        self.v += dv
        self.w += dw
        self.y += dy
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v, self.w, self.y = -1.0, -0.5, 0.0


@dataclass
class ChayNeuron:
    """Chay 1985 — pancreatic beta cell burster."""

    v: float = -50.0
    n: float = 0.1
    ca: float = 0.1
    g_ca: float = 25.0
    g_k: float = 1400.0
    g_kca: float = 12.0
    g_l: float = 7.0
    e_ca: float = 100.0
    e_k: float = -75.0
    e_l: float = -40.0
    rho: float = 0.00015
    alpha_ca: float = 0.002
    k_ca: float = 0.04
    dt: float = 0.02
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = 1.0 / (1.0 + np.exp(-(self.v + 25.0) / 8.0))
        n_inf = 1.0 / (1.0 + np.exp(-(self.v + 18.0) / 14.0))
        tau_n = 1.0 / (0.01 * max(abs(self.v + 18.0), 0.01))
        i_ca = self.g_ca * m_inf * (self.v - self.e_ca)
        kca_act = self.ca / (self.ca + 1.0)
        i_k = self.g_k * self.n * (self.v - self.e_k)
        i_kca = self.g_kca * kca_act * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)
        self.v += (-i_ca - i_k - i_kca - i_l + current) * self.dt
        self.n += (n_inf - self.n) / max(tau_n, 0.01) * self.dt
        self.ca = max(
            0.0, self.ca + self.rho * (-self.alpha_ca * i_ca - self.k_ca * self.ca) * self.dt
        )
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v, self.n, self.ca = -50.0, 0.1, 0.1


@dataclass
class ButeraRespiratoryNeuron:
    """Butera, Rinzel & Smith 1999 — pre-Botzinger respiratory neuron."""

    v: float = -50.0
    n: float = 0.01
    h_nap: float = 0.5
    g_na: float = 28.0
    g_nap: float = 2.8
    g_k: float = 11.2
    g_l: float = 2.8
    e_na: float = 50.0
    e_k: float = -85.0
    e_l: float = -65.0
    e_syn: float = -10.0
    tau_h: float = 10000.0
    dt: float = 0.1
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_na_inf = 1.0 / (1.0 + np.exp(-(self.v + 34.0) / 5.0))
        m_nap_inf = 1.0 / (1.0 + np.exp(-(self.v + 40.0) / 6.0))
        h_nap_inf = 1.0 / (1.0 + np.exp((self.v + 48.0) / 6.0))
        n_inf = 1.0 / (1.0 + np.exp(-(self.v + 29.0) / 4.0))
        tau_n = 10.0 / np.cosh((self.v + 29.0) / 8.0)
        tau_h = self.tau_h / np.cosh((self.v + 48.0) / 12.0)
        i_na = self.g_na * m_na_inf**3 * (1.0 - self.n) * (self.v - self.e_na)
        i_nap = self.g_nap * m_nap_inf * self.h_nap * (self.v - self.e_na)
        i_k = self.g_k * self.n**4 * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)
        self.v += (-i_na - i_nap - i_k - i_l + current) * self.dt
        self.n += (n_inf - self.n) / max(tau_n, 0.01) * self.dt
        self.h_nap += (h_nap_inf - self.h_nap) / max(tau_h, 0.1) * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v, self.n, self.h_nap = -50.0, 0.01, 0.5


@dataclass
class ShermanRinzelKeizerNeuron:
    """Sherman, Rinzel & Keizer 1988 — pancreatic beta cell (reduced)."""

    v: float = -50.0
    n: float = 0.1
    s: float = 0.1
    g_ca: float = 3.6
    g_k: float = 10.0
    g_s: float = 4.0
    e_ca: float = 25.0
    e_k: float = -75.0
    tau_s: float = 5000.0
    dt: float = 0.5
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = 1.0 / (1.0 + np.exp(-(self.v + 20.0) / 12.0))
        n_inf = 1.0 / (1.0 + np.exp(-(self.v + 16.0) / 5.0))
        s_inf = 1.0 / (1.0 + np.exp(-(self.v + 35.0) / 10.0))
        tau_n = 9.09
        i_ca = self.g_ca * m_inf * (self.v - self.e_ca)
        i_k = self.g_k * self.n * (self.v - self.e_k)
        i_s = self.g_s * self.s * (self.v - self.e_k)
        self.v += (-i_ca - i_k - i_s + current) * self.dt
        self.n += (n_inf - self.n) / tau_n * self.dt
        self.s += (s_inf - self.s) / self.tau_s * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v, self.n, self.s = -50.0, 0.1, 0.1


# ── IF VARIANTS ────────────────────────────────────────────────────


@dataclass
class GLIFNeuron:
    """Allen Institute GLIF5 — Generalized LIF, 5-level hierarchy.

    Teeter et al. 2018, Nat Comm. Level 5: LIF + reset rules +
    instantaneous threshold + threshold adaptation + after-spike currents.
    """

    v: float = -70.0
    theta: float = -50.0
    theta_inf: float = -50.0
    i_asc1: float = 0.0
    i_asc2: float = 0.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    tau_m: float = 10.0
    tau_theta: float = 100.0
    tau_asc1: float = 10.0
    tau_asc2: float = 200.0
    a_theta: float = 0.01
    delta_theta: float = 2.0
    r_asc1: float = 1.0
    r_asc2: float = 0.5
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        dv = (
            (-(self.v - self.v_rest) + self.resistance * current + self.i_asc1 + self.i_asc2)
            / self.tau_m
            * self.dt
        )
        dtheta = (
            (self.theta_inf - self.theta + self.a_theta * (self.v - self.v_rest))
            / self.tau_theta
            * self.dt
        )
        self.i_asc1 *= np.exp(-self.dt / self.tau_asc1)
        self.i_asc2 *= np.exp(-self.dt / self.tau_asc2)
        self.v += dv
        self.theta += dtheta
        if self.v >= self.theta:
            self.v = self.v_reset
            self.theta = max(self.theta, self.theta + self.delta_theta)
            self.i_asc1 += self.r_asc1
            self.i_asc2 += self.r_asc2
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.theta = self.theta_inf
        self.i_asc1, self.i_asc2 = 0.0, 0.0


@dataclass
class MATNeuron:
    """Kobayashi 2009 — Multi-timescale Adaptive Threshold."""

    v: float = -70.0
    theta1: float = 0.0
    theta2: float = 0.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold_base: float = -50.0
    tau_m: float = 10.0
    tau_1: float = 10.0
    tau_2: float = 200.0
    h1: float = 5.0
    h2: float = 3.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        self.v += (-(self.v - self.v_rest) + self.resistance * current) / self.tau_m * self.dt
        self.theta1 *= np.exp(-self.dt / self.tau_1)
        self.theta2 *= np.exp(-self.dt / self.tau_2)
        threshold = self.v_threshold_base + self.theta1 + self.theta2
        if self.v >= threshold:
            self.v = self.v_reset
            self.theta1 += self.h1
            self.theta2 += self.h2
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.theta1, self.theta2 = 0.0, 0.0


@dataclass
class SFANeuron:
    """Benda & Herz 2003 — Spike Frequency Adaptation IF."""

    v: float = -70.0
    g_sfa: float = 0.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    tau_sfa: float = 200.0
    delta_g: float = 0.5
    e_k: float = -80.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        self.v += (
            (-(self.v - self.v_rest) - self.g_sfa * (self.v - self.e_k) + self.resistance * current)
            / self.tau_m
            * self.dt
        )
        self.g_sfa *= np.exp(-self.dt / self.tau_sfa)
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.g_sfa += self.delta_g
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.g_sfa = 0.0


@dataclass
class StochasticIFNeuron:
    """Brunel & Hakim 1999 — Ornstein-Uhlenbeck driven IF."""

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    mu: float = 0.0
    sigma: float = 3.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        noise = self.sigma * np.sqrt(self.dt / self.tau_m) * np.random.randn()
        self.v += (-(self.v - self.v_rest) + self.mu + current) / self.tau_m * self.dt + noise
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest


@dataclass
class EscapeRateNeuron:
    """Gerstner 2000 — stochastic threshold (escape noise model)."""

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    rho_0: float = 0.001
    delta_u: float = 3.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        self.v += (-(self.v - self.v_rest) + self.resistance * current) / self.tau_m * self.dt
        rate = self.rho_0 * np.exp((self.v - self.v_threshold) / self.delta_u)
        p_spike = rate * self.dt
        if np.random.random() < p_spike:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest


@dataclass
class SigmaDeltaNeuron:
    """Yoon 2017 — event-driven sigma-delta encoding."""

    sigma: float = 0.0
    v_threshold: float = 1.0

    def step(self, current: float) -> int:
        self.sigma += current
        if self.sigma >= self.v_threshold:
            self.sigma -= self.v_threshold
            return 1
        elif self.sigma <= -self.v_threshold:
            self.sigma += self.v_threshold
            return -1
        return 0

    def reset(self):
        self.sigma = 0.0


@dataclass
class GatedLIFNeuron:
    """Yao et al. 2022 NeurIPS — LIF with learnable gates."""

    v: float = 0.0
    gate_v: float = 0.9
    gate_i: float = 1.0
    v_threshold: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        self.v = self.gate_v * self.v + self.gate_i * current
        if self.v >= self.v_threshold:
            self.v -= self.v_threshold
            return 1
        return 0

    def reset(self):
        self.v = 0.0


# ── POPULATION / RATE / NEURAL MASS ───────────────────────────────


@dataclass
class JansenRitUnit:
    """Jansen & Rit 1995 — neural mass model for EEG generation.

    6 ODEs: 3 populations (pyramidal, excitatory, inhibitory) x 2 states.
    """

    y0: float = 0.0
    y3: float = 0.0
    y1: float = 0.0
    y4: float = 0.0
    y2: float = 0.0
    y5: float = 0.0
    a_exc: float = 3.25
    b_exc: float = 22.0
    a_rate: float = 100.0
    b_rate: float = 50.0
    c: float = 135.0
    e0: float = 2.5
    v0: float = 6.0
    r: float = 0.56
    dt: float = 0.001

    def _sigmoid(self, x):
        return 2.0 * self.e0 / (1.0 + np.exp(self.r * (self.v0 - x)))

    def step(self, p_ext: float = 220.0) -> float:
        s1 = self._sigmoid(self.y1 - self.y2)
        s0 = self._sigmoid(self.c * 0.8 * self.y0)
        s2 = self._sigmoid(self.c * 0.25 * self.y0)
        dy0 = self.y3
        dy3 = self.a_exc * self.a_rate * s1 - 2.0 * self.a_rate * self.y3 - self.a_rate**2 * self.y0
        dy1 = self.y4
        dy4 = (
            self.a_exc * self.a_rate * (p_ext + self.c * 0.8 * s0)
            - 2.0 * self.a_rate * self.y4
            - self.a_rate**2 * self.y1
        )
        dy2 = self.y5
        dy5 = (
            self.b_exc * self.b_rate * self.c * 0.25 * s2
            - 2.0 * self.b_rate * self.y5
            - self.b_rate**2 * self.y2
        )
        self.y0 += dy0 * self.dt
        self.y3 += dy3 * self.dt
        self.y1 += dy1 * self.dt
        self.y4 += dy4 * self.dt
        self.y2 += dy2 * self.dt
        self.y5 += dy5 * self.dt
        return self.y1 - self.y2

    def reset(self):
        self.y0 = self.y1 = self.y2 = self.y3 = self.y4 = self.y5 = 0.0


@dataclass
class WongWangUnit:
    """Wong & Wang 2006 — reduced decision-making attractor model."""

    s1: float = 0.1
    s2: float = 0.1
    tau_s: float = 0.1
    gamma: float = 0.641
    j_n: float = 0.2609
    j_cross: float = 0.0497
    i_0: float = 0.3255
    sigma: float = 0.02
    dt: float = 0.001

    def _phi(self, i_syn):
        a, b, d = 270.0, 108.0, 0.154
        x = a * i_syn - b
        if abs(x) < 1e-6:
            return 1.0 / d
        return x / (1.0 - np.exp(-d * x))

    def step(self, stim1: float = 0.0, stim2: float = 0.0) -> tuple:
        i1 = (
            self.j_n * self.s1
            - self.j_cross * self.s2
            + self.i_0
            + stim1
            + self.sigma * np.random.randn()
        )
        i2 = (
            self.j_n * self.s2
            - self.j_cross * self.s1
            + self.i_0
            + stim2
            + self.sigma * np.random.randn()
        )
        r1, r2 = self._phi(i1), self._phi(i2)
        self.s1 += (-self.s1 / self.tau_s + (1.0 - self.s1) * self.gamma * r1) * self.dt
        self.s2 += (-self.s2 / self.tau_s + (1.0 - self.s2) * self.gamma * r2) * self.dt
        self.s1 = np.clip(self.s1, 0.0, 1.0)
        self.s2 = np.clip(self.s2, 0.0, 1.0)
        return (r1, r2)

    def reset(self):
        self.s1, self.s2 = 0.1, 0.1


@dataclass
class ErmentroutKopellPopulation:
    """Montbrio, Pazo & Roxin 2015 — exact mean-field of QIF/theta network."""

    r: float = 0.1
    v: float = -2.0
    tau: float = 1.0
    delta: float = 1.0
    eta_bar: float = -5.0
    j: float = 15.0
    dt: float = 0.01

    def step(self, ext_input: float = 0.0) -> float:
        dr = (self.delta / (np.pi * self.tau) + 2.0 * self.r * self.v) / self.tau * self.dt
        dv = (
            (
                self.v**2
                + self.eta_bar
                + ext_input
                + self.j * self.tau * self.r
                - (np.pi * self.tau * self.r) ** 2
            )
            / self.tau
            * self.dt
        )
        self.r = max(0.0, self.r + dr)
        self.v += dv
        return self.r

    def reset(self):
        self.r, self.v = 0.1, -2.0


# ── MAP-BASED ──────────────────────────────────────────────────────


@dataclass
class CourageNekorkinMapNeuron:
    """Courbage, Nekorkin & Vdovin 2007 — piecewise-linear Lorenz-type map."""

    x: float = 0.0
    y: float = 0.0
    alpha: float = 3.0
    beta: float = 0.001
    j: float = 0.1
    x_threshold: float = 1.0

    def _f(self, x):
        if x < 0:
            return self.alpha * x
        return self.alpha * x / (1.0 + self.alpha * x)

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        x_new = self._f(self.x) + self.y + current + self.j
        y_new = self.y - self.beta * (self.x + 1.0)
        self.x, self.y = x_new, y_new
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self):
        self.x, self.y = 0.0, 0.0


@dataclass
class MedvedevMapNeuron:
    """Medvedev 2005 — 1D piecewise-monotone spiking map."""

    x: float = 0.0
    alpha: float = 3.5
    beta: float = 0.5
    x_threshold: float = 0.9

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        if self.x < self.beta:
            self.x = self.alpha * self.x + current
        else:
            self.x = self.alpha * (1.0 - self.x) + current
        self.x = self.x % 1.0
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self):
        self.x = 0.0


# ── HARDWARE-SPECIFIC ──────────────────────────────────────────────


@dataclass
class LoihiCUBANeuron:
    """Loihi CUBA LIF — Intel Loihi fixed-point neuron. Davies 2018."""

    v: int = 0
    u: int = 0
    tau_v: int = 10
    tau_u: int = 5
    v_threshold: int = 1000
    v_reset: int = 0

    def step(self, weighted_input: int) -> int:
        self.u = self.u - self.u // self.tau_u + weighted_input
        self.v = self.v - self.v // self.tau_v + self.u
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v, self.u = 0, 0


@dataclass
class TrueNorthNeuron:
    """Merolla 2014 — IBM TrueNorth digital neuron."""

    v: int = 0
    leak: int = 0
    threshold: int = 100
    v_reset: int = 0

    def step(self, weighted_input: int) -> int:
        self.v = self.v + weighted_input - self.leak
        if self.v >= self.threshold:
            self.v = self.v_reset
            return 1
        if self.v < -self.threshold:
            self.v = self.v_reset
        return 0

    def reset(self):
        self.v = 0


@dataclass
class BrainScaleSAdExNeuron:
    """BrainScaleS-2 — analog AdEx (1000x real-time). Schemmel 2010."""

    v: float = -65.0
    w: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    delta_t: float = 2.0
    v_rh: float = -55.0
    tau: float = 20.0
    tau_w: float = 100.0
    a: float = 0.5
    b: float = 7.0
    hw_speedup: float = 1000.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        dt_hw = self.dt * self.hw_speedup
        exp_arg = np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0)
        exp_term = self.delta_t * np.exp(exp_arg)
        dv = (
            (-(self.v - self.v_rest) + exp_term - self.w + current)
            / self.tau
            * (dt_hw / self.hw_speedup)
        )
        dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * (dt_hw / self.hw_speedup)
        self.v += dv
        self.w += dw
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.w += self.b
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.w = 0.0


@dataclass
class SpiNNakerLIFNeuron:
    """SpiNNaker LIF — ARM Cortex-M4 digital. Furber 2014."""

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    i_offset: float = 0.0
    tau_refrac: float = 2.0
    refrac_count: float = 0.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        if self.refrac_count > 0:
            self.refrac_count -= self.dt
            return 0
        self.v += (-(self.v - self.v_rest) + (current + self.i_offset)) / self.tau_m * self.dt
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.refrac_count = self.tau_refrac
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.refrac_count = 0.0


# ── SPECIALIZED / MODERN ──────────────────────────────────────────


@dataclass
class InhomogeneousPoissonNeuron:
    """Cox 1955 — doubly stochastic Poisson (time-varying rate)."""

    dt_ms: float = 1.0

    def step(self, rate_hz: float) -> int:
        p = max(0.0, rate_hz) * self.dt_ms / 1000.0
        return 1 if np.random.random() < p else 0

    def reset(self):
        pass


@dataclass
class EnergyLIFNeuron:
    """Fardet & Levina 2020 — LIF with metabolic energy constraint."""

    v: float = -70.0
    epsilon: float = 1.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    tau_e: float = 500.0
    alpha: float = 0.1
    epsilon_0: float = 1.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        effective_r = self.resistance * self.epsilon
        self.v += (-(self.v - self.v_rest) + effective_r * current) / self.tau_m * self.dt
        self.epsilon += (self.epsilon_0 - self.epsilon) / self.tau_e * self.dt
        if self.v >= self.v_threshold and self.epsilon > 0.1:
            self.v = self.v_reset
            self.epsilon -= self.alpha
            self.epsilon = max(0.0, self.epsilon)
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.epsilon = self.epsilon_0


@dataclass
class LeakyCompeteFireNeuron:
    """Oster, Douglas & Liu 2009 — winner-take-all with lateral inhibition."""

    n_units: int = 4
    v: list = field(default_factory=lambda: [0.0] * 4)
    tau: float = 10.0
    v_threshold: float = 1.0
    w_inh: float = 0.5
    dt: float = 1.0

    def __post_init__(self):
        self.v = [0.0] * self.n_units

    def step(self, currents: list) -> list:
        spikes = [0] * self.n_units
        for i in range(self.n_units):
            self.v[i] += (-self.v[i] + currents[i]) / self.tau * self.dt
        for i in range(self.n_units):
            if self.v[i] >= self.v_threshold:
                spikes[i] = 1
                self.v[i] = 0.0
                for j in range(self.n_units):
                    if j != i:
                        self.v[j] -= self.w_inh
                        self.v[j] = max(0.0, self.v[j])
        return spikes

    def reset(self):
        self.v = [0.0] * self.n_units


@dataclass
class PrescottNeuron:
    """Prescott 2008 — Type I/II/III excitability via M-current tuning."""

    v: float = -65.0
    w: float = 0.0
    g_fast: float = 20.0
    g_slow: float = 20.0
    g_l: float = 2.0
    e_fast: float = 50.0
    e_slow: float = -100.0
    e_l: float = -70.0
    beta_w: float = -21.0
    gamma_w: float = 15.0
    tau_w: float = 100.0
    phi: float = 0.15
    dt: float = 0.1
    v_threshold: float = -20.0

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = 1.0 / (1.0 + np.exp(-(self.v + 20.0) / 15.0))
        w_inf = 1.0 / (1.0 + np.exp(-(self.v - self.beta_w) / self.gamma_w))
        i_fast = self.g_fast * m_inf * (self.v - self.e_fast)
        i_slow = self.g_slow * self.w * (self.v - self.e_slow)
        i_l = self.g_l * (self.v - self.e_l)
        self.v += (-i_fast - i_slow - i_l + current) * self.dt
        self.w += self.phi * (w_inf - self.w) / self.tau_w * self.dt
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -65.0
        self.w = 0.0
