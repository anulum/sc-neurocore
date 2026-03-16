// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

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
        self.v += (self.v - self.v.powi(3) / 3.0 - self.w + current) * self.dt;
        self.w += self.epsilon * (self.v + self.a - self.b * self.w) * self.dt;
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
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let dx = (self.y - self.x.powi(3) + self.b * self.x.powi(2) - self.z + current) * self.dt;
        let dy = (1.0 - 5.0 * self.x.powi(2) - self.y) * self.dt;
        let dz = self.r * (self.s * (self.x - self.x_rest) - self.z) * self.dt;
        self.x += dx;
        self.y += dy;
        self.z += dz;
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
        self.x += (self.b * self.x - self.omega * self.y + current) * self.dt;
        self.y += (self.omega * self.x + self.b * self.y) * self.dt;
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
        self.v += (self.v - self.v.powi(3) / 3.0 - self.w + self.y + current) * self.dt;
        self.w += self.delta * (self.a + self.v - self.b * self.w) * self.dt;
        self.y += self.mu * (self.c - self.v - self.d * self.y) * self.dt;
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
        self.v += (self.f_v(self.v) - self.w + current) * self.dt;
        self.w += self.epsilon * (self.v - self.gamma * self.w) * self.dt;
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
        self.v += (f - self.w + current + self.rho) * self.dt;
        self.w += self.epsilon * (g - self.w) * self.dt;
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
        let v_prev = self.v;
        let poly = -(17.81 + 47.71 * self.v + 32.63 * self.v * self.v) * (self.v - 0.55);
        let syn = -26.0 * self.r * (self.v + 0.92);
        self.v += (poly + syn + current) * self.dt;
        self.r += (-self.r + 1.35 * self.v + 1.03) / self.tau_r * self.dt;
        if self.v >= self.v_peak && v_prev < self.v_peak {
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
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v + 25.0) / 8.0).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v + 18.0) / 14.0).exp());
        let d = (self.v + 18.0).abs().max(0.01);
        let tau_n = 1.0 / (0.01 * d);
        let kca_act = self.ca / (self.ca + 1.0);
        let i_ca = self.g_ca * m_inf * (self.v - self.e_ca);
        let i_k = self.g_k * self.n * (self.v - self.e_k);
        let i_kca = self.g_kca * kca_act * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_ca - i_k - i_kca - i_l + current) * self.dt;
        self.n += (n_inf - self.n) / tau_n.max(0.01) * self.dt;
        self.ca =
            (self.ca + self.rho * (-self.alpha_ca * i_ca - self.k_ca * self.ca) * self.dt).max(0.0);
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
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
        let n_inf = 1.0 / (1.0 + (-(self.v + 18.0) / 7.0).exp());
        let tau_n = 20.0;
        let q_kca = self.ca / (self.ca + self.k_d);
        let i_ca = self.g_ca * m_inf * (self.v - self.e_ca);
        let i_k = self.g_k * self.n * (self.v - self.e_k);
        let i_kca = self.g_kca * q_kca * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_ca - i_k - i_kca - i_l + current) * self.dt;
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
        let n_inf = 1.0 / (1.0 + (-(self.v + 16.0) / 5.6).exp());
        let s_inf = 1.0 / (1.0 + (-(self.v + 35.0) / 10.0).exp());
        let tau_n = 9.09;
        let i_ca = self.g_ca * m_inf * (self.v - self.e_ca);
        let i_k = self.g_k * self.n * (self.v - self.e_k);
        let i_s = self.g_s * self.s * (self.v - self.e_k);
        self.v += (-i_ca - i_k - i_s + current) * self.dt / 5.3;
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
        let tau_n = 10.0;
        let h_na = 1.0 / (1.0 + ((self.v + 48.0) / 4.0).exp());
        let i_na = self.g_na * m_na.powi(3) * h_na * (self.v - self.e_na);
        let i_nap = self.g_nap * m_nap * self.h_nap * (self.v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_na - i_nap - i_k - i_l + current) * self.dt;
        self.n += (n_inf - self.n) / tau_n * self.dt;
        self.h_nap += (h_nap_inf - self.h_nap) / self.tau_h * self.dt;
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
        self.v += (f_v - self.w - self.z + current) * self.dt;
        self.w += self.eps1 * (self.v - self.gamma * self.w + self.alpha) * self.dt;
        self.z += self.eps2 * (self.beta * (self.v + 0.7) - self.z) * self.dt;
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
    fn chay_fires() {
        let mut n = ChayNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(20.0)).sum();
        assert!(t > 0);
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
}
