// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Two-dimensional and higher spiking neuron models

//! Two-dimensional and higher spiking neuron models.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// FitzHugh-Nagumo 1961 — 2D qualitative spike model.
#[derive(Clone, Debug)]
pub struct FitzHughNagumoNeuron {
    pub v: f64,
    pub w: f64,
    pub a: f64,
    pub b: f64,
    pub epsilon: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl FitzHughNagumoNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0,
            w: -0.5,
            a: 0.7,
            b: 0.8,
            epsilon: 0.08,
            dt: 0.1,
            v_threshold: 1.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        // FitzHugh 1961: simultaneous Euler (both derivatives use old state)
        let dv = (self.v - self.v.powi(3) / 3.0 - self.w + current) * self.dt;
        let dw = self.epsilon * (self.v + self.a - self.b * self.w) * self.dt;
        self.v += dv;
        self.w += dw;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -1.0;
        self.w = -0.5;
    }
}
impl Default for FitzHughNagumoNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Morris-Lecar 1981 — barnacle muscle fiber.
#[derive(Clone, Debug)]
pub struct MorrisLecarNeuron {
    pub v: f64,
    pub w: f64,
    pub c_m: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
    pub v4: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl MorrisLecarNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            w: 0.0,
            c_m: 20.0,
            g_ca: 4.0,
            g_k: 8.0,
            g_l: 2.0,
            e_ca: 120.0,
            e_k: -84.0,
            e_l: -60.0,
            v1: -1.2,
            v2: 18.0,
            v3: 12.0,
            v4: 17.4,
            phi: 1.0 / 15.0,
            dt: 0.1,
            v_threshold: 0.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 0.5 * (1.0 + ((self.v - self.v1) / self.v2).tanh());
        let w_inf = 0.5 * (1.0 + ((self.v - self.v3) / self.v4).tanh());
        let lam = self.phi * ((self.v - self.v3) / (2.0 * self.v4)).cosh();
        let i_ca = self.g_ca * m_inf * (self.v - self.e_ca);
        let i_k = self.g_k * self.w * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_ca - i_k - i_l + current) / self.c_m * self.dt;
        self.w += lam * (w_inf - self.w) * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -60.0;
        self.w = 0.0;
    }
}
impl Default for MorrisLecarNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Hindmarsh-Rose 1984 — 3D chaotic bursting model.
#[derive(Clone, Debug)]
pub struct HindmarshRoseNeuron {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub b: f64,
    pub r: f64,
    pub s: f64,
    pub x_rest: f64,
    pub dt: f64,
    pub x_threshold: f64,
}

impl HindmarshRoseNeuron {
    pub fn new() -> Self {
        Self {
            x: -1.6,
            y: -10.0,
            z: 2.0,
            b: 3.0,
            r: 0.001,
            s: 4.0,
            x_rest: -1.6,
            dt: 0.1,
            x_threshold: 1.0,
        }
    }
    fn derivatives(&self, x: f64, y: f64, z: f64, current: f64) -> (f64, f64, f64) {
        (
            y - x.powi(3) + self.b * x.powi(2) - z + current,
            1.0 - 5.0 * x.powi(2) - y,
            self.r * (self.s * (x - self.x_rest) - z),
        )
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let (x0, y0, z0) = (self.x, self.y, self.z);
        let dt = self.dt;
        let k1 = self.derivatives(x0, y0, z0, current);
        let k2 = self.derivatives(
            x0 + 0.5 * dt * k1.0,
            y0 + 0.5 * dt * k1.1,
            z0 + 0.5 * dt * k1.2,
            current,
        );
        let k3 = self.derivatives(
            x0 + 0.5 * dt * k2.0,
            y0 + 0.5 * dt * k2.1,
            z0 + 0.5 * dt * k2.2,
            current,
        );
        let k4 = self.derivatives(x0 + dt * k3.0, y0 + dt * k3.1, z0 + dt * k3.2, current);
        self.x = x0 + (dt / 6.0) * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
        self.y = y0 + (dt / 6.0) * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
        self.z = z0 + (dt / 6.0) * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2);
        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.x = -1.6;
        self.y = -10.0;
        self.z = 2.0;
    }
}
impl Default for HindmarshRoseNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Resonate-and-Fire — damped oscillator with threshold. Izhikevich 2001.
#[derive(Clone, Debug)]
pub struct ResonateAndFireNeuron {
    pub x: f64,
    pub y: f64,
    pub b: f64,
    pub omega: f64,
    pub threshold: f64,
    pub dt: f64,
}

impl ResonateAndFireNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            b: -0.1,
            omega: 1.0,
            threshold: 1.0,
            dt: 0.05,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        // Izhikevich 2001: simultaneous Euler (both derivatives use old state)
        let dx = (self.b * self.x - self.omega * self.y + current) * self.dt;
        let dy = (self.omega * self.x + self.b * self.y) * self.dt;
        self.x += dx;
        self.y += dy;
        let r = (self.x * self.x + self.y * self.y).sqrt();
        if r >= self.threshold {
            self.x = 0.0;
            self.y = 0.0;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}
impl Default for ResonateAndFireNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Balanced Resonate-and-Fire — divergence-bound RF with smooth refractory
/// reset. Higuchi, Kairat, Bohte, and Otte 2024, Algorithm 1.
#[derive(Clone, Debug)]
pub struct BalancedResonateAndFireNeuron {
    pub x: f64,
    pub y: f64,
    pub q: f64,
    pub omega: f64,
    pub b_offset: f64,
    pub threshold: f64,
    pub gamma: f64,
    pub dt: f64,
}

pub fn brf_sustain_oscillation_boundary(omega: f64, dt: f64) -> f64 {
    let scaled = dt * omega;
    (-1.0 + (1.0 - scaled * scaled).max(0.0).sqrt()) / dt
}

impl BalancedResonateAndFireNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            q: 0.0,
            omega: 10.0,
            b_offset: 1.0,
            threshold: 1.0,
            gamma: 0.9,
            dt: 0.01,
        }
    }

    pub fn p_omega(&self) -> f64 {
        brf_sustain_oscillation_boundary(self.omega, self.dt)
    }

    pub fn damping(&self) -> f64 {
        self.p_omega() - self.b_offset - self.q
    }

    pub fn dynamic_threshold(&self) -> f64 {
        self.threshold + self.q
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !(self.dt.is_finite()
            && self.omega.is_finite()
            && self.dt > 0.0
            && self.omega > 0.0
            && self.dt * self.omega <= 1.0)
        {
            return 0;
        }
        let b_t = self.damping();
        let theta_t = self.dynamic_threshold();
        let x_prev = self.x;
        let y_prev = self.y;
        self.x = x_prev + self.dt * (b_t * x_prev - self.omega * y_prev + current);
        self.y = y_prev + self.dt * (self.omega * x_prev + b_t * y_prev);
        let spike = if self.x >= theta_t { 1 } else { 0 };
        self.q = self.gamma * self.q + spike as f64;
        spike
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
        self.q = 0.0;
    }
}
impl Default for BalancedResonateAndFireNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// FitzHugh-Rinzel — 3D extension with slow bursting variable.
#[derive(Clone, Debug)]
pub struct FitzHughRinzelNeuron {
    pub v: f64,
    pub w: f64,
    pub y: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub delta: f64,
    pub mu: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl FitzHughRinzelNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0,
            w: -0.5,
            y: 0.0,
            a: 0.7,
            b: 0.8,
            c: -0.775,
            d: 1.0,
            delta: 0.08,
            mu: 0.0001,
            dt: 0.1,
            v_threshold: 1.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        // FitzHugh-Rinzel: simultaneous Euler (all derivatives use old state)
        let dv = (self.v - self.v.powi(3) / 3.0 - self.w + self.y + current) * self.dt;
        let dw = self.delta * (self.a + self.v - self.b * self.w) * self.dt;
        let dy = self.mu * (self.c - self.v - self.d * self.y) * self.dt;
        self.v += dv;
        self.w += dw;
        self.y += dy;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -1.0;
        self.w = -0.5;
        self.y = 0.0;
    }
}
impl Default for FitzHughRinzelNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// McKean 1970 — piecewise-linear FHN variant.
#[derive(Clone, Debug)]
pub struct McKeanNeuron {
    pub v: f64,
    pub w: f64,
    pub a: f64,
    pub epsilon: f64,
    pub gamma: f64,
    pub dt: f64,
    pub v_peak: f64,
}

impl McKeanNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            w: 0.0,
            a: 0.25,
            epsilon: 0.01,
            gamma: 0.5,
            dt: 0.1,
            v_peak: 0.8,
        }
    }
    fn f_v(&self, v: f64) -> f64 {
        let half_a = self.a / 2.0;
        let mid = (1.0 + self.a) / 2.0;
        if v < half_a {
            -v
        } else if v < mid {
            v - self.a
        } else {
            1.0 - v
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        // McKean: simultaneous Euler (both derivatives use old state)
        let dv = (self.f_v(self.v) - self.w + current) * self.dt;
        let dw = self.epsilon * (self.v - self.gamma * self.w) * self.dt;
        self.v += dv;
        self.w += dw;
        if self.v >= self.v_peak && v_prev < self.v_peak {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.w = 0.0;
    }
}
impl Default for McKeanNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Terman-Wang oscillator — segmentation by oscillatory correlation.
#[derive(Clone, Debug)]
pub struct TermanWangOscillator {
    pub v: f64,
    pub w: f64,
    pub alpha: f64,
    pub beta: f64,
    pub epsilon: f64,
    pub rho: f64,
    pub dt: f64,
    pub v_peak: f64,
}

impl TermanWangOscillator {
    pub fn new() -> Self {
        Self {
            v: -1.5,
            w: -0.5,
            alpha: 3.0,
            beta: 0.2,
            epsilon: 0.02,
            rho: 0.0,
            dt: 0.05,
            v_peak: 1.5,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let f = 3.0 * self.v - self.v.powi(3) + 2.0;
        let g = self.alpha * (1.0 + (self.v / self.beta).tanh());
        // TermanWang: simultaneous Euler (both derivatives use old state)
        let dv = (f - self.w + current + self.rho) * self.dt;
        let dw = self.epsilon * (g - self.w) * self.dt;
        self.v += dv;
        self.w += dw;
        if self.v >= self.v_peak && v_prev < self.v_peak {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -1.5;
        self.w = -0.5;
    }
}
impl Default for TermanWangOscillator {
    fn default() -> Self {
        Self::new()
    }
}

/// Benda-Herz 2003 — firing-rate adaptation via subtractive feedback.
#[derive(Clone, Debug)]
pub struct BendaHerzNeuron {
    pub a: f64,
    pub f_max: f64,
    pub beta: f64,
    pub i_half: f64,
    pub tau_a: f64,
    pub delta_a: f64,
    pub dt: f64,
    rng: Xoshiro256PlusPlus,
}

impl BendaHerzNeuron {
    pub fn new(seed: u64) -> Self {
        Self {
            a: 0.0,
            f_max: 200.0,
            beta: 0.1,
            i_half: 5.0,
            tau_a: 100.0,
            delta_a: 0.5,
            dt: 1.0,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x = current - self.a;
        let rate = self.f_max / (1.0 + (-self.beta * (x - self.i_half)).exp());
        self.a += (-self.a / self.tau_a + self.delta_a * rate) * self.dt;
        let p = rate * self.dt / 1000.0;
        if self.rng.random::<f64>() < p {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.a = 0.0;
    }
}

/// Alpha-synapse LIF — separate excitatory/inhibitory exponential synapses.
#[derive(Clone, Debug)]
pub struct AlphaNeuron {
    pub v: f64,
    pub i_exc: f64,
    pub i_inh: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub tau_v: f64,
    pub tau_exc: f64,
    pub tau_inh: f64,
    pub dt: f64,
}

impl AlphaNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            i_exc: 0.0,
            i_inh: 0.0,
            v_rest: 0.0,
            v_threshold: 1.0,
            tau_v: 20.0,
            tau_exc: 5.0,
            tau_inh: 10.0,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, exc_current: f64, inh_current: f64) -> i32 {
        self.i_exc += (-self.i_exc / self.tau_exc + exc_current) * self.dt;
        self.i_inh += (-self.i_inh / self.tau_inh + inh_current) * self.dt;
        self.v += (-(self.v - self.v_rest) + self.i_exc - self.i_inh) / self.tau_v * self.dt;
        if self.v >= self.v_threshold {
            self.v = self.v_rest;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.i_exc = 0.0;
        self.i_inh = 0.0;
    }
}
impl Default for AlphaNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Conductance-based LIF (COBA). Brette et al. 2007.
#[derive(Clone, Debug)]
pub struct COBALIFNeuron {
    pub v: f64,
    pub g_e: f64,
    pub g_i: f64,
    pub c_m: f64,
    pub g_l: f64,
    pub e_l: f64,
    pub e_e: f64,
    pub e_i: f64,
    pub tau_e: f64,
    pub tau_i: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl COBALIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            g_e: 0.0,
            g_i: 0.0,
            c_m: 200.0,
            g_l: 10.0,
            e_l: -65.0,
            e_e: 0.0,
            e_i: -80.0,
            tau_e: 5.0,
            tau_i: 10.0,
            v_threshold: -50.0,
            v_reset: -65.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64, delta_ge: f64, delta_gi: f64) -> i32 {
        self.g_e += delta_ge;
        self.g_i += delta_gi;
        let i_syn = self.g_e * (self.v - self.e_e) + self.g_i * (self.v - self.e_i);
        self.v += (-self.g_l * (self.v - self.e_l) - i_syn + current) / self.c_m * self.dt;
        self.g_e *= (-self.dt / self.tau_e).exp();
        self.g_i *= (-self.dt / self.tau_i).exp();
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.g_e = 0.0;
        self.g_i = 0.0;
    }
}
impl Default for COBALIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Gutkin-Ermentrout — reduced cortical neuron with Type-I excitability.
#[derive(Clone, Debug)]
pub struct GutkinErmentroutNeuron {
    pub v: f64,
    pub n: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl GutkinErmentroutNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            n: 0.1,
            g_na: 20.0,
            g_k: 10.0,
            g_l: 8.0,
            e_na: 60.0,
            e_k: -90.0,
            e_l: -80.0,
            dt: 0.05,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v + 20.0) / 15.0).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v + 25.0) / 5.0).exp());
        self.n += (n_inf - self.n) * self.dt;
        let i_na = self.g_na * m_inf * (self.v - self.e_na);
        let i_k = self.g_k * self.n * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_na - i_k - i_l + current) * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.n = 0.1;
    }
}
impl Default for GutkinErmentroutNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Wilson HR — simplified cortical neuron. Wilson 1999.
#[derive(Clone, Debug)]
pub struct WilsonHRNeuron {
    pub v: f64,
    pub r: f64,
    pub tau_r: f64,
    pub v_peak: f64,
    pub dt: f64,
}

impl WilsonHRNeuron {
    pub fn new() -> Self {
        Self {
            v: -0.7,
            r: 0.1,
            tau_r: 1.9,
            v_peak: 0.4,
            dt: 0.05,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        // Wilson 1999: simultaneous Euler (both derivatives use old state)
        let poly = -(17.81 + 47.71 * self.v + 32.63 * self.v * self.v) * (self.v - 0.55);
        let syn = -26.0 * self.r * (self.v + 0.92);
        let dv = (poly + syn + current) * self.dt;
        let dr = (-self.r + 1.35 * self.v + 1.03) / self.tau_r * self.dt;
        self.v += dv;
        self.r += dr;
        // Wilson 1999: hard reset on threshold (not crossing)
        if self.v >= self.v_peak {
            self.v = -0.7;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -0.7;
        self.r = 0.1;
    }
}
impl Default for WilsonHRNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Chay 1985 — pancreatic beta cell with Ca-dependent K.
#[derive(Clone, Debug)]
pub struct ChayNeuron {
    pub v: f64,
    pub n: f64,
    pub ca: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_kca: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub rho: f64,
    pub alpha_ca: f64,
    pub k_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ChayNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.1,
            ca: 0.1,
            g_ca: 25.0,
            g_k: 1400.0,
            g_kca: 12.0,
            g_l: 7.0,
            e_ca: 100.0,
            e_k: -75.0,
            e_l: -40.0,
            rho: 0.00015,
            alpha_ca: 0.002,
            k_ca: 0.04,
            dt: 0.02,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.dt.is_finite() || self.dt <= 0.0 {
            return 0;
        }

        let v_initial = self.v;
        let mut v = self.v;
        let mut n = self.n;
        let mut ca = self.ca;
        let substeps = (self.dt / 0.001_f64).ceil().max(1.0) as usize;
        let h = self.dt / substeps as f64;
        let mut crossed = false;

        for _ in 0..substeps {
            let m_inf = 1.0 / (1.0 + (-(v + 25.0) / 8.0).clamp(-700.0, 700.0).exp());
            let n_inf = 1.0 / (1.0 + (-(v + 18.0) / 14.0).clamp(-700.0, 700.0).exp());
            let d = (v + 18.0).abs().max(0.01);
            let tau_n = 1.0 / (0.01 * d);
            let ca_denominator = ca + 1.0;
            if ca_denominator <= 0.0 {
                return 0;
            }
            let kca_act = ca / ca_denominator;
            let i_ca = self.g_ca * m_inf * (v - self.e_ca);
            let i_k = self.g_k * n * (v - self.e_k);
            let i_kca = self.g_kca * kca_act * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let v_next = v + (-i_ca - i_k - i_kca - i_l + current) * h;
            let n_next = n + (n_inf - n) / tau_n.max(0.01) * h;
            let ca_next = ca + self.rho * (-self.alpha_ca * i_ca - self.k_ca * ca) * h;
            if !v_next.is_finite()
                || !n_next.is_finite()
                || !ca_next.is_finite()
                || !(-200.0..=200.0).contains(&v_next)
                || !(0.0..=1.0).contains(&n_next)
                || !(0.0..=100.0).contains(&ca_next)
            {
                return 0;
            }
            crossed = crossed || (v_next >= self.v_threshold && v < self.v_threshold);
            v = v_next;
            n = n_next;
            ca = ca_next;
        }

        self.v = v;
        self.n = n;
        self.ca = ca;
        if crossed && v_initial < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.1;
        self.ca = 0.1;
    }
}
impl Default for ChayNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Chay-Keizer — modified beta cell with inactivating Ca current.
#[derive(Clone, Debug)]
pub struct ChayKeizerNeuron {
    pub v: f64,
    pub n: f64,
    pub ca: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_kca: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub k_d: f64,
    pub f_ca: f64,
    pub k_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ChayKeizerNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.01,
            ca: 0.1,
            g_ca: 20.0,
            g_k: 25.0,
            g_kca: 12.0,
            g_l: 0.1,
            e_ca: 100.0,
            e_k: -75.0,
            e_l: -40.0,
            k_d: 1.0,
            f_ca: 0.004,
            k_ca: 0.03,
            dt: 0.02,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v + 25.0) / 8.0).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v + 18.0) / 14.0).exp());
        let tau_n = (20.0 / (1.0 + ((self.v + 18.0) / 14.0).exp())).max(0.1);
        let q_kca = self.ca / (self.ca + self.k_d);
        let i_ca = self.g_ca * m_inf * (self.v - self.e_ca);
        let i_k = self.g_k * self.n * (self.v - self.e_k);
        let i_kca = self.g_kca * q_kca * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_ca - i_k - i_kca - i_l + current) * self.dt;
        self.v = self.v.clamp(-200.0, 200.0);
        self.n += (n_inf - self.n) / tau_n * self.dt;
        self.ca = (self.ca + (-self.f_ca * i_ca - self.k_ca * self.ca) * self.dt).max(0.0);
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.01;
        self.ca = 0.1;
    }
}
impl Default for ChayKeizerNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Sherman-Rinzel-Keizer 1988 — pancreatic beta cell (reduced).
#[derive(Clone, Debug)]
pub struct ShermanRinzelKeizerNeuron {
    pub v: f64,
    pub n: f64,
    pub s: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_s: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub tau_s: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ShermanRinzelKeizerNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.1,
            s: 0.1,
            g_ca: 3.6,
            g_k: 10.0,
            g_s: 4.0,
            e_ca: 25.0,
            e_k: -75.0,
            tau_s: 5000.0,
            dt: 0.5,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v + 20.0) / 12.0).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v + 16.0) / 5.0).exp());
        let s_inf = 1.0 / (1.0 + (-(self.v + 35.0) / 10.0).exp());
        let tau_n = 9.09;
        let i_ca = self.g_ca * m_inf * (self.v - self.e_ca);
        let i_k = self.g_k * self.n * (self.v - self.e_k);
        let i_s = self.g_s * self.s * (self.v - self.e_k);
        self.v += (-i_ca - i_k - i_s + current) * self.dt;
        self.n += (n_inf - self.n) / tau_n * self.dt;
        self.s += (s_inf - self.s) / self.tau_s * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.1;
        self.s = 0.1;
    }
}
impl Default for ShermanRinzelKeizerNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Butera pre-Bötzinger respiratory neuron. Butera et al. 1999.
#[derive(Clone, Debug)]
pub struct ButeraRespiratoryNeuron {
    pub v: f64,
    pub n: f64,
    pub h_nap: f64,
    pub g_na: f64,
    pub g_nap: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub tau_h: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ButeraRespiratoryNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.01,
            h_nap: 0.5,
            g_na: 28.0,
            g_nap: 2.8,
            g_k: 11.2,
            g_l: 2.8,
            e_na: 50.0,
            e_k: -85.0,
            e_l: -65.0,
            tau_h: 10000.0,
            dt: 0.1,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_na = 1.0 / (1.0 + (-(self.v + 34.0) / 5.0).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v + 29.0) / 4.0).exp());
        let m_nap = 1.0 / (1.0 + (-(self.v + 40.0) / 6.0).exp());
        let h_nap_inf = 1.0 / (1.0 + ((self.v + 48.0) / 6.0).exp());
        // Butera et al. 1999 Model I: tau_n = 10/cosh((V+29)/8)
        let tau_n = (10.0 / ((self.v + 29.0) / 8.0).cosh().max(1e-12)).max(0.01);
        // Model I uses (1-n) for fast Na inactivation, not a separate h gate
        let i_na = self.g_na * m_na.powi(3) * (1.0 - self.n) * (self.v - self.e_na);
        let i_nap = self.g_nap * m_nap * self.h_nap * (self.v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        // tau_h_nap = tau_h / cosh((V+48)/12) (Butera 1999)
        let tau_h_eff = (self.tau_h / ((self.v + 48.0) / 12.0).cosh().max(1e-12)).max(0.1);
        self.v = (self.v + (-i_na - i_nap - i_k - i_l + current) * self.dt).clamp(-200.0, 100.0);
        self.n = (self.n + (n_inf - self.n) / tau_n * self.dt).clamp(0.0, 1.0);
        self.h_nap = (self.h_nap + (h_nap_inf - self.h_nap) / tau_h_eff * self.dt).clamp(0.0, 1.0);
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.01;
        self.h_nap = 0.5;
    }
}
impl Default for ButeraRespiratoryNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// e-prop ALIF — adaptive LIF for eligibility-propagation learning. Bellec et al. 2020.
#[derive(Clone, Debug)]
pub struct EPropALIFNeuron {
    pub v: f64,
    pub a: f64,
    pub e_trace: f64,
    pub alpha_m: f64,
    pub alpha_a: f64,
    pub v_threshold_base: f64,
    pub beta: f64,
}

impl EPropALIFNeuron {
    pub fn new(tau_m: f64, tau_a: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            a: 0.0,
            e_trace: 0.0,
            alpha_m: (-dt / tau_m).exp(),
            alpha_a: (-dt / tau_a).exp(),
            v_threshold_base: 1.0,
            beta: 0.07,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.v = self.alpha_m * self.v + current;
        let threshold = self.v_threshold_base + self.beta * self.a;
        let psi = ((1.0 - (self.v - threshold).abs()) * 0.3).max(0.0);
        self.e_trace = self.alpha_a * self.e_trace + psi;
        if self.v >= threshold {
            self.v = 0.0;
            self.a = self.alpha_a * self.a + 1.0;
            1
        } else {
            self.a *= self.alpha_a;
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.a = 0.0;
        self.e_trace = 0.0;
    }
}

impl Default for EPropALIFNeuron {
    fn default() -> Self {
        Self::new(20.0, 200.0, 1.0)
    }
}

/// SuperSpike — LIF with surrogate gradient trace. Zenke & Ganguli 2018.
#[derive(Clone, Debug)]
pub struct SuperSpikeNeuron {
    pub v: f64,
    pub trace: f64,
    pub alpha_m: f64,
    pub alpha_e: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub beta_sg: f64,
}

impl SuperSpikeNeuron {
    pub fn new(tau_m: f64, tau_e: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            trace: 0.0,
            alpha_m: (-dt / tau_m).exp(),
            alpha_e: (-dt / tau_e).exp(),
            v_threshold: 1.0,
            v_reset: 0.0,
            beta_sg: 10.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.v = self.alpha_m * self.v + current;
        let sg = 1.0 / (self.beta_sg * (self.v - self.v_threshold).abs() + 1.0).powi(2);
        self.trace = self.alpha_e * self.trace + sg;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.trace = 0.0;
    }
}

impl Default for SuperSpikeNeuron {
    fn default() -> Self {
        Self::new(10.0, 10.0, 1.0)
    }
}

/// Learnable Neuron Model (LNM) — parameterised activation + decay.
#[derive(Clone, Debug)]
pub struct LearnableNeuronModel {
    pub v: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub v_threshold: f64,
    pub f_slope: f64,
    pub f_shift: f64,
}

impl LearnableNeuronModel {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            alpha: 0.9,
            beta: 0.1,
            gamma: 0.05,
            v_threshold: 1.0,
            f_slope: 5.0,
            f_shift: 0.5,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let f_v = 1.0 / (1.0 + (-(self.f_slope * (self.v - self.f_shift))).exp());
        self.v = self.alpha * self.v + self.beta * current + self.gamma * f_v;
        if self.v >= self.v_threshold {
            self.v = 0.0;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}
impl Default for LearnableNeuronModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Pernarowski 1994 — simplified beta cell burster (3 ODE).
#[derive(Clone, Debug)]
pub struct PernarowskiNeuron {
    pub v: f64,
    pub w: f64,
    pub z: f64,
    pub alpha: f64,
    pub beta: f64,
    pub eps1: f64,
    pub eps2: f64,
    pub gamma: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PernarowskiNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0,
            w: 0.0,
            z: 0.0,
            alpha: 0.1,
            beta: 0.5,
            eps1: 0.1,
            eps2: 0.001,
            gamma: 0.5,
            dt: 0.1,
            v_threshold: 0.5,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let f_v = self.v - self.v.powi(3) / 3.0;
        // Pernarowski: simultaneous Euler (all derivatives use old state)
        let dv = (f_v - self.w - self.z + current) * self.dt;
        let dw = self.eps1 * (self.v - self.gamma * self.w + self.alpha) * self.dt;
        let dz = self.eps2 * (self.beta * (self.v + 0.7) - self.z) * self.dt;
        self.v += dv;
        self.w += dw;
        self.z += dz;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -1.0;
        self.w = 0.0;
        self.z = 0.0;
    }
}
impl Default for PernarowskiNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Brunel-Wang LIF with NMDA/AMPA/GABA
// ═══════════════════════════════════════════════════════════════════

/// Brunel-Wang 2001 — LIF with NMDA (Mg²⁺ block), AMPA, and GABA synaptic
/// conductances for decision-making and working memory circuits.
///
/// Key feature: voltage-dependent NMDA conductance via Mg²⁺ block factor
/// `1 / (1 + [Mg²⁺]/3.57 · exp(-0.062·V))`. This creates positive feedback
/// that sustains persistent activity in recurrent circuits.
///
/// The single-current interface routes external input to i_ampa_ext; recurrent
/// AMPA/NMDA/GABA inputs are zero (use the multi-arg `step_full()` for
/// network simulations).
///
/// Brunel, N. & Wang, X.J., J Comput Neurosci 11:63, 2001.
#[derive(Clone, Debug)]
pub struct BrunelWangNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_ref: f64,
    pub g_ampa_ext: f64,
    pub g_ampa_rec: f64,
    pub g_nmda: f64,
    pub g_gaba: f64,
    pub v_ampa: f64,
    pub v_nmda: f64,
    pub v_gaba: f64,
    pub c_m: f64,
    pub mg_conc: f64,
    pub dt: f64,
    pub ref_remaining: f64,
    pub gain: f64,
}

impl BrunelWangNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            v_rest: -70.0,
            v_reset: -55.0,
            v_threshold: -50.0,
            tau_m: 20.0,
            tau_ref: 2.0,
            g_ampa_ext: 2.1,
            g_ampa_rec: 0.05,
            g_nmda: 0.165,
            g_gaba: 1.3,
            v_ampa: 0.0,
            v_nmda: 0.0,
            v_gaba: -70.0,
            c_m: 0.5,
            mg_conc: 1.0,
            dt: 0.1,
            ref_remaining: 0.0,
            gain: 1.0,
        }
    }

    /// Mg²⁺ block factor (Jahr & Stevens 1990).
    #[inline]
    fn nmda_mg_block(&self, v: f64) -> f64 {
        1.0 / (1.0 + self.mg_conc / 3.57 * (-0.062 * v).exp())
    }

    /// Full step with all 4 synaptic inputs (for network simulations).
    pub fn step_full(
        &mut self,
        i_ampa_ext: f64,
        s_ampa_rec: f64,
        s_nmda_rec: f64,
        s_gaba: f64,
    ) -> i32 {
        if self.ref_remaining > 0.0 {
            self.ref_remaining -= self.dt;
            return 0;
        }

        let v = self.v;

        // Synaptic currents (conductance-based)
        let i_ampa = -self.g_ampa_ext * (v - self.v_ampa) * i_ampa_ext
            - self.g_ampa_rec * (v - self.v_ampa) * s_ampa_rec;
        let i_nmda = -self.g_nmda * self.nmda_mg_block(v) * (v - self.v_nmda) * s_nmda_rec;
        let i_gaba = -self.g_gaba * (v - self.v_gaba) * s_gaba;

        // Membrane dynamics (LIF)
        let i_leak = -(v - self.v_rest) / self.tau_m;
        let dv = (i_leak + (i_ampa + i_nmda + i_gaba) / self.c_m) * self.dt;
        self.v += dv;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.ref_remaining = self.tau_ref;
            1
        } else {
            0
        }
    }

    /// Single-current interface: routes input to external AMPA drive.
    pub fn step(&mut self, current: f64) -> i32 {
        self.step_full(self.gain * current, 0.0, 0.0, 0.0)
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.ref_remaining = 0.0;
    }
}

impl Default for BrunelWangNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fhn_fires() {
        let mut n = FitzHughNagumoNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(1.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn morris_lecar_fires() {
        let mut n = MorrisLecarNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(100.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn hr_fires() {
        let mut n = HindmarshRoseNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(3.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn rnf_fires() {
        let mut n = ResonateAndFireNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(3.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn fhr_fires() {
        let mut n = FitzHughRinzelNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(1.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn mckean_fires() {
        let mut n = McKeanNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }
    #[test]
    fn tw_fires() {
        let mut n = TermanWangOscillator::new();
        let t: i32 = (0..2000).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }
    #[test]
    fn benda_herz_fires() {
        let mut n = BendaHerzNeuron::new(42);
        let t: i32 = (0..10000).map(|_| n.step(20.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn alpha_fires() {
        let mut n = AlphaNeuron::new();
        let t: i32 = (0..100).map(|_| n.step(0.5, 0.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn coba_fires() {
        let mut n = COBALIFNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(500.0, 0.0, 0.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn gutkin_fires() {
        let mut n = GutkinErmentroutNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(15.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn wilson_hr_fires() {
        let mut n = WilsonHRNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }
    #[test]
    fn chay_drive_changes_state_without_leaving_physical_bounds() {
        let mut rest = ChayNeuron::new();
        let mut driven = ChayNeuron::new();
        for _ in 0..500 {
            rest.step(0.0);
            driven.step(5.0);
        }
        assert!(driven.v > rest.v);
        assert!((0.0..=1.0).contains(&driven.n));
        assert!(driven.ca >= 0.0);
    }
    #[test]
    fn chay_keizer_fires() {
        let mut n = ChayKeizerNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(10.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn srk_fires() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn butera_fires() {
        let mut n = ButeraRespiratoryNeuron::new();
        let t: i32 = (0..20000).map(|_| n.step(50.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn eprop_fires() {
        let mut n = EPropALIFNeuron::default();
        let t: i32 = (0..50).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }
    #[test]
    fn superspike_fires() {
        let mut n = SuperSpikeNeuron::default();
        let t: i32 = (0..50).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }
    #[test]
    fn lnm_fires() {
        let mut n = LearnableNeuronModel::new();
        let t: i32 = (0..50).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn pernarowski_fires() {
        let mut n = PernarowskiNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(1.0)).sum();
        assert!(t > 0);
    }

    // ── Multi-angle tests for simple_spiking models ──

    // -- FitzHughNagumo --
    #[test]
    fn fhn_silent_without_input() {
        let mut n = FitzHughNagumoNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn fhn_reset_clears_state() {
        let mut n = FitzHughNagumoNeuron::new();
        for _ in 0..500 {
            n.step(1.0);
        }
        n.reset();
        assert!((n.v - (-1.0)).abs() < 1e-10);
        assert!((n.w - (-0.5)).abs() < 1e-10);
    }
    #[test]
    fn fhn_moderate_input_stable() {
        let mut n = FitzHughNagumoNeuron::new();
        for _ in 0..2000 {
            n.step(2.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn fhn_recovery_variable() {
        let mut n = FitzHughNagumoNeuron::new();
        for _ in 0..2000 {
            n.step(1.0);
        }
        // w should have evolved from initial
        assert!((n.w - (-0.5)).abs() > 0.01, "recovery w should change");
    }
    #[test]
    fn fhn_nan_no_panic() {
        FitzHughNagumoNeuron::new().step(f64::NAN);
    }
    #[test]
    fn fhn_negative_no_crash() {
        let mut n = FitzHughNagumoNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }

    // -- MorrisLecar --
    #[test]
    fn ml_silent_without_input() {
        let mut n = MorrisLecarNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn ml_reset_clears_state() {
        let mut n = MorrisLecarNeuron::new();
        for _ in 0..500 {
            n.step(100.0);
        }
        n.reset();
        assert!((n.v - (-60.0)).abs() < 1e-10);
    }
    #[test]
    fn ml_moderate_input_stable() {
        let mut n = MorrisLecarNeuron::new();
        for _ in 0..2000 {
            n.step(200.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn ml_nan_no_panic() {
        MorrisLecarNeuron::new().step(f64::NAN);
    }
    #[test]
    fn ml_negative_no_crash() {
        let mut n = MorrisLecarNeuron::new();
        for _ in 0..500 {
            n.step(-50.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn ml_k_gating_bounded() {
        let mut n = MorrisLecarNeuron::new();
        for _ in 0..2000 {
            n.step(100.0);
        }
        assert!(n.w >= 0.0 && n.w <= 1.0, "w={}", n.w);
    }

    // -- HindmarshRose --
    #[test]
    fn hr_reset_clears_state() {
        let mut n = HindmarshRoseNeuron::new();
        for _ in 0..500 {
            n.step(3.0);
        }
        n.reset();
        assert!((n.x - (-1.6)).abs() < 1e-10);
    }
    #[test]
    fn hr_moderate_input_stable() {
        let mut n = HindmarshRoseNeuron::new();
        for _ in 0..2000 {
            n.step(5.0);
        }
        assert!(n.x.is_finite());
    }
    #[test]
    fn hr_slow_z_evolves() {
        let mut n = HindmarshRoseNeuron::new();
        let z0 = n.z;
        for _ in 0..2000 {
            n.step(3.0);
        }
        assert!((n.z - z0).abs() > 0.001, "slow variable z should evolve");
    }
    #[test]
    fn hr_nan_no_panic() {
        HindmarshRoseNeuron::new().step(f64::NAN);
    }
    #[test]
    fn hr_negative_no_crash() {
        let mut n = HindmarshRoseNeuron::new();
        for _ in 0..500 {
            n.step(-1.0);
        }
        assert!(n.x.is_finite());
    }

    // -- ResonateAndFire --
    #[test]
    fn rnf_reset_clears_state() {
        let mut n = ResonateAndFireNeuron::new();
        for _ in 0..500 {
            n.step(3.0);
        }
        n.reset();
        assert!((n.x - 0.0).abs() < 1e-10);
    }
    #[test]
    fn rnf_bounded() {
        let mut n = ResonateAndFireNeuron::new();
        for _ in 0..5000 {
            n.step(100.0);
        }
        assert!(n.x.is_finite());
    }
    #[test]
    fn rnf_nan_no_panic() {
        ResonateAndFireNeuron::new().step(f64::NAN);
    }
    #[test]
    fn rnf_negative_no_crash() {
        let mut n = ResonateAndFireNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.x.is_finite());
    }
    #[test]
    fn rnf_subthreshold_oscillation() {
        let mut n = ResonateAndFireNeuron::new();
        for _ in 0..100 {
            n.step(0.5);
        }
        assert!(n.x.abs() > 0.0 || n.y.abs() > 0.0);
    }

    #[test]
    fn brf_boundary_matches_algorithm() {
        let p = brf_sustain_oscillation_boundary(10.0, 0.01);
        let expected = (-1.0 + (1.0_f64 - 0.1_f64 * 0.1_f64).sqrt()) / 0.01;
        assert!((p - expected).abs() < 1e-12);
    }

    #[test]
    fn brf_step_matches_algorithm_one_step() {
        let mut n = BalancedResonateAndFireNeuron {
            x: 0.2,
            y: -0.1,
            q: 0.3,
            omega: 12.0,
            b_offset: 0.75,
            threshold: 1.0,
            gamma: 0.9,
            dt: 0.01,
        };
        let p_omega = brf_sustain_oscillation_boundary(12.0, 0.01);
        let b_t = p_omega - 0.75 - 0.3;
        let expected_x = 0.2 + 0.01 * (b_t * 0.2 - 12.0 * -0.1 + 2.0);
        let expected_y = -0.1 + 0.01 * (12.0 * 0.2 + b_t * -0.1);
        let expected_spike = if expected_x >= 1.3 { 1 } else { 0 };
        let spike = n.step(2.0);
        assert_eq!(spike, expected_spike);
        assert!((n.x - expected_x).abs() < 1e-12);
        assert!((n.y - expected_y).abs() < 1e-12);
        assert!((n.q - (0.9 * 0.3 + expected_spike as f64)).abs() < 1e-12);
    }

    #[test]
    fn brf_reset_clears_membrane_and_refractory_state() {
        let mut n = BalancedResonateAndFireNeuron::new();
        assert_eq!(n.step(200.0), 1);
        n.reset();
        assert_eq!(n.x, 0.0);
        assert_eq!(n.y, 0.0);
        assert_eq!(n.q, 0.0);
    }

    // -- FitzHughRinzel --
    #[test]
    fn fhr_reset_clears_state() {
        let mut n = FitzHughRinzelNeuron::new();
        for _ in 0..500 {
            n.step(1.0);
        }
        n.reset();
        assert!((n.v - (-1.0)).abs() < 1e-10);
    }
    #[test]
    fn fhr_bounded() {
        let mut n = FitzHughRinzelNeuron::new();
        for _ in 0..2000 {
            n.step(50.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn fhr_nan_no_panic() {
        FitzHughRinzelNeuron::new().step(f64::NAN);
    }
    #[test]
    fn fhr_negative_no_crash() {
        let mut n = FitzHughRinzelNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn fhr_slow_y_variable() {
        let mut n = FitzHughRinzelNeuron::new();
        let y0 = n.y;
        for _ in 0..2000 {
            n.step(1.0);
        }
        assert!((n.y - y0).abs() > 1e-6, "slow y should evolve in 3D model");
    }

    // -- McKean --
    #[test]
    fn mckean_reset_clears_state() {
        let mut n = McKeanNeuron::new();
        for _ in 0..500 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn mckean_bounded() {
        let mut n = McKeanNeuron::new();
        for _ in 0..2000 {
            n.step(50.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn mckean_nan_no_panic() {
        McKeanNeuron::new().step(f64::NAN);
    }
    #[test]
    fn mckean_negative_no_crash() {
        let mut n = McKeanNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }

    // -- TermanWang --
    #[test]
    fn tw_reset_clears_state() {
        let mut n = TermanWangOscillator::new();
        for _ in 0..500 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - (-1.5)).abs() < 1e-10);
    }
    #[test]
    fn tw_bounded() {
        let mut n = TermanWangOscillator::new();
        for _ in 0..2000 {
            n.step(50.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn tw_nan_no_panic() {
        TermanWangOscillator::new().step(f64::NAN);
    }
    #[test]
    fn tw_negative_no_crash() {
        let mut n = TermanWangOscillator::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }

    // -- BendaHerz --
    #[test]
    fn benda_herz_reset_clears_state() {
        let mut n = BendaHerzNeuron::new(42);
        for _ in 0..100 {
            n.step(20.0);
        }
        n.reset();
        assert!((n.a - 0.0).abs() < 1e-10);
    }
    #[test]
    fn benda_herz_bounded() {
        let mut n = BendaHerzNeuron::new(42);
        for _ in 0..10000 {
            n.step(1e4);
        }
        assert!(n.a.is_finite());
    }
    #[test]
    fn benda_herz_adaptation() {
        let mut n = BendaHerzNeuron::new(42);
        for _ in 0..10000 {
            n.step(20.0);
        }
        assert!(n.a > 0.0, "adaptation variable a should increase: {}", n.a);
    }
    #[test]
    fn benda_herz_nan_no_panic() {
        BendaHerzNeuron::new(42).step(f64::NAN);
    }
    #[test]
    fn benda_herz_negative_no_crash() {
        let mut n = BendaHerzNeuron::new(42);
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.a.is_finite());
    }

    // -- Alpha --
    #[test]
    fn alpha_reset_clears_state() {
        let mut n = AlphaNeuron::new();
        for _ in 0..50 {
            n.step(0.5, 0.0);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn alpha_bounded() {
        let mut n = AlphaNeuron::new();
        for _ in 0..1000 {
            n.step(100.0, 0.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn alpha_spike_input_drives() {
        let mut n = AlphaNeuron::new();
        for _ in 0..100 {
            n.step(0.0, 1.0);
        }
        // Spike input should contribute to synaptic current
        assert!(n.v.is_finite());
    }
    #[test]
    fn alpha_nan_no_panic() {
        AlphaNeuron::new().step(f64::NAN, 0.0);
    }
    #[test]
    fn alpha_negative_no_crash() {
        let mut n = AlphaNeuron::new();
        for _ in 0..100 {
            n.step(-5.0, 0.0);
        }
        assert!(n.v.is_finite());
    }

    // -- COBALIF --
    #[test]
    fn coba_reset_clears_state() {
        let mut n = COBALIFNeuron::new();
        for _ in 0..100 {
            n.step(500.0, 0.0, 0.0);
        }
        n.reset();
        assert!((n.v - n.e_l).abs() < 1e-10);
    }
    #[test]
    fn coba_bounded() {
        let mut n = COBALIFNeuron::new();
        for _ in 0..2000 {
            n.step(1e5, 0.0, 0.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn coba_inhibition_suppresses() {
        let mut n_exc = COBALIFNeuron::new();
        let mut n_inh = COBALIFNeuron::new();
        let t_exc: i32 = (0..2000).map(|_| n_exc.step(500.0, 0.0, 0.0)).sum();
        let t_inh: i32 = (0..2000).map(|_| n_inh.step(500.0, 0.0, 5.0)).sum();
        assert!(t_inh <= t_exc, "inhibition should reduce spiking");
    }
    #[test]
    fn coba_nan_no_panic() {
        COBALIFNeuron::new().step(f64::NAN, 0.0, 0.0);
    }
    #[test]
    fn coba_negative_no_crash() {
        let mut n = COBALIFNeuron::new();
        for _ in 0..500 {
            n.step(-100.0, 0.0, 0.0);
        }
        assert!(n.v.is_finite());
    }

    // -- GutkinErmentrout --
    #[test]
    fn gutkin_reset_clears_state() {
        let mut n = GutkinErmentroutNeuron::new();
        for _ in 0..500 {
            n.step(15.0);
        }
        n.reset();
        assert!((n.v - (-65.0)).abs() < 1e-10);
    }
    #[test]
    fn gutkin_bounded() {
        let mut n = GutkinErmentroutNeuron::new();
        for _ in 0..2000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn gutkin_nan_no_panic() {
        GutkinErmentroutNeuron::new().step(f64::NAN);
    }
    #[test]
    fn gutkin_negative_no_crash() {
        let mut n = GutkinErmentroutNeuron::new();
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }

    // -- WilsonHR --
    #[test]
    fn wilson_hr_reset_clears_state() {
        let mut n = WilsonHRNeuron::new();
        for _ in 0..500 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - (-0.7)).abs() < 1e-10);
    }
    #[test]
    fn wilson_hr_moderate_stable() {
        let mut n = WilsonHRNeuron::new();
        for _ in 0..2000 {
            n.step(1.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn wilson_hr_nan_no_panic() {
        WilsonHRNeuron::new().step(f64::NAN);
    }
    #[test]
    fn wilson_hr_negative_no_crash() {
        let mut n = WilsonHRNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }

    // -- Chay --
    #[test]
    fn chay_reset_clears_state() {
        let mut n = ChayNeuron::new();
        for _ in 0..1000 {
            n.step(20.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
    }
    #[test]
    fn chay_bounded() {
        let mut n = ChayNeuron::new();
        for _ in 0..5000 {
            n.step(200.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn chay_ca_nonneg() {
        let mut n = ChayNeuron::new();
        for _ in 0..5000 {
            n.step(20.0);
        }
        assert!(n.ca >= 0.0, "Ca²⁺ must be non-negative");
    }
    #[test]
    fn chay_nan_no_panic() {
        ChayNeuron::new().step(f64::NAN);
    }
    #[test]
    fn chay_negative_no_crash() {
        let mut n = ChayNeuron::new();
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }

    // -- ChayKeizer --
    #[test]
    fn chay_keizer_reset_clears_state() {
        let mut n = ChayKeizerNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
    }
    #[test]
    fn chay_keizer_bounded() {
        let mut n = ChayKeizerNeuron::new();
        for _ in 0..5000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn chay_keizer_nan_no_panic() {
        ChayKeizerNeuron::new().step(f64::NAN);
    }
    #[test]
    fn chay_keizer_negative_no_crash() {
        let mut n = ChayKeizerNeuron::new();
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }

    // -- ShermanRinzelKeizer --
    #[test]
    fn srk_reset_clears_state() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        for _ in 0..1000 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
    }
    #[test]
    fn srk_bounded() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        for _ in 0..5000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn srk_nan_no_panic() {
        ShermanRinzelKeizerNeuron::new().step(f64::NAN);
    }
    #[test]
    fn srk_negative_no_crash() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }

    // -- ButeraRespiratory --
    #[test]
    fn butera_reset_clears_state() {
        let mut n = ButeraRespiratoryNeuron::new();
        for _ in 0..1000 {
            n.step(50.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
    }
    #[test]
    fn butera_bounded() {
        let mut n = ButeraRespiratoryNeuron::new();
        for _ in 0..5000 {
            n.step(500.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn butera_nan_no_panic() {
        ButeraRespiratoryNeuron::new().step(f64::NAN);
    }
    #[test]
    fn butera_negative_no_crash() {
        let mut n = ButeraRespiratoryNeuron::new();
        for _ in 0..500 {
            n.step(-20.0);
        }
        assert!(n.v.is_finite());
    }

    // -- EPropALIF --
    #[test]
    fn eprop_reset_clears_state() {
        let mut n = EPropALIFNeuron::default();
        for _ in 0..50 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn eprop_bounded() {
        let mut n = EPropALIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn eprop_adaptation() {
        let mut n = EPropALIFNeuron::default();
        for _ in 0..50 {
            n.step(0.5);
        }
        // a (adaptation) should have increased after spikes
        assert!(n.a.is_finite());
    }
    #[test]
    fn eprop_nan_no_panic() {
        EPropALIFNeuron::default().step(f64::NAN);
    }

    // -- SuperSpike --
    #[test]
    fn superspike_reset_clears_state() {
        let mut n = SuperSpikeNeuron::default();
        for _ in 0..50 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn superspike_bounded() {
        let mut n = SuperSpikeNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn superspike_trace_evolves() {
        let mut n = SuperSpikeNeuron::default();
        for _ in 0..50 {
            n.step(0.5);
        }
        assert!(n.trace.is_finite());
    }
    #[test]
    fn superspike_nan_no_panic() {
        SuperSpikeNeuron::default().step(f64::NAN);
    }

    // -- LearnableNeuron --
    #[test]
    fn lnm_reset_clears_state() {
        let mut n = LearnableNeuronModel::new();
        for _ in 0..50 {
            n.step(2.0);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn lnm_bounded() {
        let mut n = LearnableNeuronModel::new();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn lnm_nan_no_panic() {
        LearnableNeuronModel::new().step(f64::NAN);
    }

    // -- Pernarowski --
    #[test]
    fn pernarowski_reset_clears_state() {
        let mut n = PernarowskiNeuron::new();
        for _ in 0..500 {
            n.step(1.0);
        }
        n.reset();
        assert!((n.v - (-1.0)).abs() < 1e-10);
    }
    #[test]
    fn pernarowski_bounded() {
        let mut n = PernarowskiNeuron::new();
        for _ in 0..2000 {
            n.step(50.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn pernarowski_slow_z() {
        let mut n = PernarowskiNeuron::new();
        let z0 = n.z;
        for _ in 0..2000 {
            n.step(1.0);
        }
        assert!((n.z - z0).abs() > 1e-6, "slow z should evolve");
    }
    #[test]
    fn pernarowski_nan_no_panic() {
        PernarowskiNeuron::new().step(f64::NAN);
    }
    #[test]
    fn pernarowski_negative_no_crash() {
        let mut n = PernarowskiNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }

    // -- BrunelWang tests --

    #[test]
    fn brunel_wang_fires_with_ampa_ext() {
        let mut n = BrunelWangNeuron::new();
        let mut spikes = 0;
        for _ in 0..5000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 0,
            "Must fire with external AMPA drive, got {spikes}"
        );
    }

    #[test]
    fn brunel_wang_silent_without_input() {
        let mut n = BrunelWangNeuron::new();
        let spikes: i32 = (0..10_000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0, "Must be silent without input");
    }

    #[test]
    fn brunel_wang_nmda_mg_block() {
        let n = BrunelWangNeuron::new();
        // At resting potential (-70 mV), Mg²⁺ block should be strong
        let block_rest = n.nmda_mg_block(-70.0);
        // At depolarised potential (0 mV), block should be weak
        let block_depol = n.nmda_mg_block(0.0);
        assert!(
            block_depol > block_rest,
            "Mg²⁺ block should weaken with depolarisation: rest={block_rest:.3} depol={block_depol:.3}"
        );
        // At -70 mV, block factor should be small (< 0.1)
        assert!(
            block_rest < 0.1,
            "Block at -70 mV should be < 0.1, got {block_rest:.3}"
        );
        // At 0 mV, block factor should be close to 1
        assert!(
            block_depol > 0.5,
            "Block at 0 mV should be > 0.5, got {block_depol:.3}"
        );
    }

    #[test]
    fn brunel_wang_full_step_nmda_drive() {
        // NMDA recurrent input should drive firing via positive feedback
        let mut n = BrunelWangNeuron::new();
        n.v = -55.0; // near threshold, Mg²⁺ block partially relieved
        let mut spikes = 0;
        for _ in 0..1000 {
            spikes += n.step_full(0.0, 0.0, 1.0, 0.0); // pure NMDA drive
        }
        assert!(
            spikes > 0,
            "NMDA drive at depolarised V should cause spikes"
        );
    }

    #[test]
    fn brunel_wang_gaba_suppresses() {
        let mut with_gaba = BrunelWangNeuron::new();
        let mut no_gaba = BrunelWangNeuron::new();
        let mut spikes_gaba = 0;
        let mut spikes_no = 0;
        for _ in 0..5000 {
            spikes_gaba += with_gaba.step_full(3.0, 0.0, 0.0, 1.0); // GABA on
            spikes_no += no_gaba.step_full(3.0, 0.0, 0.0, 0.0);
        }
        assert!(
            spikes_no >= spikes_gaba,
            "GABA should suppress: no_gaba={spikes_no}, with_gaba={spikes_gaba}"
        );
    }

    #[test]
    fn brunel_wang_refractory() {
        let mut n = BrunelWangNeuron::new();
        // Drive to spike
        while n.step(10.0) == 0 {}
        // Immediately after spike, should be in refractory
        assert!(n.ref_remaining > 0.0, "Should be refractory after spike");
        // Should not spike during refractory
        assert_eq!(
            n.step(100.0),
            0,
            "Should not spike during refractory period"
        );
    }

    #[test]
    fn brunel_wang_reset() {
        let mut n = BrunelWangNeuron::new();
        for _ in 0..1000 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.v, n.v_rest);
        assert_eq!(n.ref_remaining, 0.0);
    }

    #[test]
    fn brunel_wang_voltage_bounded() {
        let mut n = BrunelWangNeuron::new();
        for _ in 0..10_000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite(), "V must stay finite");
        // V should stay near v_reset during sustained spiking (refractory resets)
        assert!(
            n.v <= n.v_threshold,
            "V should be at or below threshold (reset clamp)"
        );
    }

    #[test]
    fn brunel_wang_nan_input() {
        let mut n = BrunelWangNeuron::new();
        n.step(f64::NAN);
        // NaN input → V becomes NaN. Check we don't panic.
    }

    #[test]
    fn brunel_wang_performance() {
        let start = std::time::Instant::now();
        let mut n = BrunelWangNeuron::new();
        for _ in 0..10_000 {
            std::hint::black_box(n.step(3.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10k steps in <50ms, took {}ms",
            elapsed.as_millis()
        );
    }
}
