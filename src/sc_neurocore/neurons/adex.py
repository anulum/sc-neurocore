# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AdExNeuron:
    """Adaptive Exponential Integrate-and-Fire. Brette & Gerstner 2005.

    dv/dt = -(v - v_rest)/tau + delta_T * exp((v - v_rh)/delta_T) / tau - w/C + I/C
    dw/dt = (a * (v - v_rest) - w) / tau_w
    if v >= v_threshold: v = v_reset, w += b
    """

    v: float = -65.0
    w: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    v_rh: float = -55.0
    delta_t: float = 2.0
    tau: float = 20.0
    tau_w: float = 100.0
    a: float = 0.5
    b: float = 7.0
    c_m: float = 200.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        exp_term = self.delta_t * np.exp(np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0))
        dv = (-(self.v - self.v_rest) + exp_term - self.w + current) / self.tau * self.dt
        dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt

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
class ExpIFNeuron:
    """Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003."""

    v: float = -65.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = -50.0
    v_rh: float = -55.0
    delta_t: float = 2.0
    tau: float = 20.0
    dt: float = 0.1

    def step(self, current: float) -> int:
        exp_term = self.delta_t * np.exp(np.clip((self.v - self.v_rh) / self.delta_t, -20.0, 20.0))
        dv = (-(self.v - self.v_rest) + exp_term + current) / self.tau * self.dt
        self.v += dv

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest


@dataclass
class LapicqueNeuron:
    """Lapicque 1907 — classical RC integrate-and-fire.

    tau * dv/dt = -(v - v_rest) + R * I
    """

    v: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    tau: float = 20.0
    resistance: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        dv = (-(self.v - self.v_rest) + self.resistance * current) / self.tau * self.dt
        self.v += dv

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest


@dataclass
class AlphaNeuron:
    """Alpha-synapse neuron. Rall 1967.

    Dual excitatory/inhibitory synaptic currents with alpha-function kinetics.
    """

    v: float = 0.0
    i_exc: float = 0.0
    i_inh: float = 0.0
    v_rest: float = 0.0
    v_threshold: float = 1.0
    tau_v: float = 20.0
    tau_exc: float = 5.0
    tau_inh: float = 10.0
    dt: float = 1.0

    def step(self, exc_current: float, inh_current: float = 0.0) -> int:
        self.i_exc += (-self.i_exc / self.tau_exc + exc_current) * self.dt
        self.i_inh += (-self.i_inh / self.tau_inh + inh_current) * self.dt
        dv = (-(self.v - self.v_rest) + self.i_exc - self.i_inh) / self.tau_v * self.dt
        self.v += dv

        if self.v >= self.v_threshold:
            self.v = self.v_rest
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.i_exc = 0.0
        self.i_inh = 0.0


@dataclass
class HodgkinHuxleyNeuron:
    """Hodgkin-Huxley 1952 — 4-ODE ion channel model.

    C_m dv/dt = -g_Na m³h(v-E_Na) - g_K n⁴(v-E_K) - g_L(v-E_L) + I
    dm/dt = α_m(1-m) - β_m·m
    dh/dt = α_h(1-h) - β_h·h
    dn/dt = α_n(1-n) - β_n·n
    """

    v: float = -65.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.32
    c_m: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_l: float = -54.4
    dt: float = 0.01
    v_threshold: float = 0.0

    def _alpha_m(self, v):
        d = v + 40.0
        if abs(d) < 1e-7:
            return 1.0
        return 0.1 * d / (1.0 - np.exp(-d / 10.0))

    def _beta_m(self, v):
        return 4.0 * np.exp(-(v + 65.0) / 18.0)

    def _alpha_h(self, v):
        return 0.07 * np.exp(-(v + 65.0) / 20.0)

    def _beta_h(self, v):
        return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))

    def _alpha_n(self, v):
        d = v + 55.0
        if abs(d) < 1e-7:
            return 0.1
        return 0.01 * d / (1.0 - np.exp(-d / 10.0))

    def _beta_n(self, v):
        return 0.125 * np.exp(-(v + 65.0) / 80.0)

    def step(self, current: float) -> int:
        v_prev = self.v
        for _ in range(int(1.0 / self.dt)):
            am, bm = self._alpha_m(self.v), self._beta_m(self.v)
            ah, bh = self._alpha_h(self.v), self._beta_h(self.v)
            an, bn = self._alpha_n(self.v), self._beta_n(self.v)

            self.m += (am * (1 - self.m) - bm * self.m) * self.dt
            self.h += (ah * (1 - self.h) - bh * self.h) * self.dt
            self.n += (an * (1 - self.n) - bn * self.n) * self.dt

            i_na = self.g_na * self.m**3 * self.h * (self.v - self.e_na)
            i_k = self.g_k * self.n**4 * (self.v - self.e_k)
            i_l = self.g_l * (self.v - self.e_l)

            self.v += (-i_na - i_k - i_l + current) / self.c_m * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -65.0
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32


@dataclass
class FitzHughNagumoNeuron:
    """FitzHugh-Nagumo 1961 — 2D qualitative spike model.

    dv/dt = v - v³/3 - w + I
    dw/dt = ε(v + a - bw)
    """

    v: float = -1.0
    w: float = -0.5
    a: float = 0.7
    b: float = 0.8
    epsilon: float = 0.08
    dt: float = 0.1
    v_threshold: float = 1.0

    def step(self, current: float) -> int:
        v_prev = self.v
        dv = (self.v - self.v**3 / 3.0 - self.w + current) * self.dt
        dw = self.epsilon * (self.v + self.a - self.b * self.w) * self.dt
        self.v += dv
        self.w += dw
        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -1.0
        self.w = -0.5


@dataclass
class MorrisLecarNeuron:
    """Morris-Lecar 1981 — calcium-potassium oscillator.

    C dv/dt = -g_Ca m_∞(v)(v-E_Ca) - g_K w(v-E_K) - g_L(v-E_L) + I
    dw/dt = λ(v)(w_∞(v) - w)
    """

    v: float = -60.0
    w: float = 0.0
    c_m: float = 20.0
    g_ca: float = 4.0
    g_k: float = 8.0
    g_l: float = 2.0
    e_ca: float = 120.0
    e_k: float = -84.0
    e_l: float = -60.0
    v1: float = -1.2
    v2: float = 18.0
    v3: float = 12.0
    v4: float = 17.4
    phi: float = 1.0 / 15.0
    dt: float = 0.1
    v_threshold: float = 0.0

    def _m_inf(self, v):
        return 0.5 * (1.0 + np.tanh((v - self.v1) / self.v2))

    def _w_inf(self, v):
        return 0.5 * (1.0 + np.tanh((v - self.v3) / self.v4))

    def _lam(self, v):
        return self.phi * np.cosh((v - self.v3) / (2.0 * self.v4))

    def step(self, current: float) -> int:
        v_prev = self.v
        m_inf = self._m_inf(self.v)
        w_inf = self._w_inf(self.v)
        lam = self._lam(self.v)

        i_ca = self.g_ca * m_inf * (self.v - self.e_ca)
        i_k = self.g_k * self.w * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)

        self.v += (-i_ca - i_k - i_l + current) / self.c_m * self.dt
        self.w += lam * (w_inf - self.w) * self.dt

        return 1 if (self.v >= self.v_threshold and v_prev < self.v_threshold) else 0

    def reset(self):
        self.v = -60.0
        self.w = 0.0


@dataclass
class QuadraticIFNeuron:
    """Quadratic Integrate-and-Fire — canonical Type-I excitability.

    dv/dt = v² + I
    Reset when v >= v_peak.
    """

    v: float = -1.0
    v_reset: float = -1.0
    v_peak: float = 1.0
    dt: float = 0.01

    def step(self, current: float) -> int:
        self.v += (self.v**2 + current) * self.dt
        if self.v >= self.v_peak:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self):
        self.v = self.v_reset


@dataclass
class HindmarshRoseNeuron:
    """Hindmarsh-Rose 1984 — 3D chaotic bursting model.

    dx/dt = y - x³ + bx² - z + I
    dy/dt = 1 - 5x² - y
    dz/dt = r(s(x - x_rest) - z)
    """

    x: float = -1.6
    y: float = -10.0
    z: float = 2.0
    b: float = 3.0
    r: float = 0.001
    s: float = 4.0
    x_rest: float = -1.6
    dt: float = 0.1
    x_threshold: float = 1.0

    def step(self, current: float) -> int:
        x_prev = self.x
        dx = (self.y - self.x**3 + self.b * self.x**2 - self.z + current) * self.dt
        dy = (1.0 - 5.0 * self.x**2 - self.y) * self.dt
        dz = self.r * (self.s * (self.x - self.x_rest) - self.z) * self.dt
        self.x += dx
        self.y += dy
        self.z += dz
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def reset(self):
        self.x = -1.6
        self.y = -10.0
        self.z = 2.0


@dataclass
class ThetaNeuron:
    """Theta neuron — canonical Type-I on the unit circle.

    dθ/dt = (1 - cos θ) + (1 + cos θ) · I
    Spike when θ crosses π.
    Ermentrout & Kopell 1986.
    """

    theta: float = 0.0
    dt: float = 0.01

    def step(self, current: float) -> int:
        theta_prev = self.theta
        dtheta = ((1.0 - np.cos(self.theta)) + (1.0 + np.cos(self.theta)) * current) * self.dt
        self.theta += dtheta
        # Wrap to [-π, π]
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi
        return 1 if (theta_prev < np.pi * 0.99 and self.theta >= np.pi * 0.99) else 0

    def reset(self):
        self.theta = 0.0


@dataclass
class ResonateAndFireNeuron:
    """Resonate-and-Fire — subthreshold oscillation + threshold.

    Izhikevich 2001. Complex dynamics: z = x + i*y,
    dz/dt = (b + iω)z + I, fire when |z| > threshold.
    Implemented as 2 real ODEs.
    """

    x: float = 0.0
    y: float = 0.0
    b: float = -0.1
    omega: float = 1.0
    threshold: float = 1.0
    dt: float = 0.05

    def step(self, current: float) -> int:
        dx = (self.b * self.x - self.omega * self.y + current) * self.dt
        dy = (self.omega * self.x + self.b * self.y) * self.dt
        self.x += dx
        self.y += dy
        r = np.sqrt(self.x**2 + self.y**2)
        if r >= self.threshold:
            self.x = 0.0
            self.y = 0.0
            return 1
        return 0

    def reset(self):
        self.x = 0.0
        self.y = 0.0


@dataclass
class PoissonNeuron:
    """Poisson spike generator — stochastic firing at rate λ.

    P(spike in dt) = λ · dt. Essential for input layer generation.
    """

    rate_hz: float = 100.0
    dt_ms: float = 1.0
    _rng: object = None

    def __post_init__(self):
        self._rng = np.random.default_rng()

    def step(self, rate_override: float = -1.0) -> int:
        r = self.rate_hz if rate_override < 0 else rate_override
        p = r * self.dt_ms / 1000.0
        return 1 if self._rng.random() < p else 0

    def reset(self):
        pass


@dataclass
class SpikeResponseNeuron:
    """Spike Response Model (SRM0) — kernel-based, no ODEs.

    v(t) = η(t - t_last) + Σ κ(t - t_in) · w
    Spike when v(t) ≥ threshold.
    Gerstner 1995.
    """

    v: float = 0.0
    v_threshold: float = 1.0
    tau_eta: float = 10.0
    tau_kappa: float = 5.0
    eta_reset: float = -5.0
    time_since_spike: float = 1000.0
    dt: float = 1.0

    def step(self, weighted_input: float) -> int:
        # Refractory kernel (spike afterpotential)
        eta = (
            self.eta_reset * np.exp(-self.time_since_spike / self.tau_eta)
            if self.time_since_spike < 100.0
            else 0.0
        )
        # Input kernel
        kappa = weighted_input * (1.0 - np.exp(-self.dt / self.tau_kappa))
        self.v = eta + kappa
        self.time_since_spike += self.dt

        if self.v >= self.v_threshold:
            self.time_since_spike = 0.0
            self.v = 0.0
            return 1
        return 0

    def reset(self):
        self.v = 0.0
        self.time_since_spike = 1000.0


@dataclass
class MihalasNieburNeuron:
    """Mihalas-Niebur Generalized IF — captures 20 spike patterns.

    Mihalas & Niebur 2009. Multiple internal thresholds and
    adaptation currents enable tonic/phasic/burst/accommodation patterns.
    """

    v: float = 0.0
    theta: float = 1.0
    i1: float = 0.0
    i2: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    theta_reset: float = 1.0
    theta_inf: float = 1.0
    tau_v: float = 10.0
    tau_theta: float = 100.0
    tau_1: float = 10.0
    tau_2: float = 200.0
    a: float = 0.0
    b: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        dv = (-(self.v - self.v_rest) + self.i1 + self.i2 + current) / self.tau_v * self.dt
        dtheta = (
            (self.theta_inf - self.theta + self.a * (self.v - self.v_rest))
            / self.tau_theta
            * self.dt
        )
        di1 = -self.i1 / self.tau_1 * self.dt
        di2 = -self.i2 / self.tau_2 * self.dt
        self.v += dv
        self.theta += dtheta
        self.i1 += di1
        self.i2 += di2

        if self.v >= self.theta:
            self.v = self.v_reset
            self.theta = max(self.theta, self.theta_reset)
            self.i1 += self.r1
            self.i2 += self.r2
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.theta = self.theta_reset
        self.i1 = 0.0
        self.i2 = 0.0
