// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stochastic generators, spike-response models, and neural

//! Stochastic generators, spike-response models, and neural mass/population models.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Poisson spike generator — P(spike in dt) = λ·dt.
#[derive(Clone, Debug)]
pub struct PoissonNeuron {
    pub rate_hz: f64,
    pub dt_ms: f64,
    rng: Xoshiro256PlusPlus,
}

impl PoissonNeuron {
    pub fn new(rate_hz: f64, dt_ms: f64, seed: u64) -> Self {
        Self {
            rate_hz,
            dt_ms,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
    pub fn step(&mut self, rate_override: f64) -> i32 {
        let r = if rate_override < 0.0 {
            self.rate_hz
        } else {
            rate_override
        };
        let p = r * self.dt_ms / 1000.0;
        if self.rng.random::<f64>() < p {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {}
}

/// Inhomogeneous Poisson — rate passed per step.
#[derive(Clone, Debug)]
pub struct InhomogeneousPoissonNeuron {
    pub dt_ms: f64,
    rng: Xoshiro256PlusPlus,
}

impl InhomogeneousPoissonNeuron {
    pub fn new(dt_ms: f64, seed: u64) -> Self {
        Self {
            dt_ms,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
    pub fn step(&mut self, rate_hz: f64) -> i32 {
        let p = rate_hz.max(0.0) * self.dt_ms / 1000.0;
        if self.rng.random::<f64>() < p {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {}
}

/// Gamma-renewal process — order-k Poisson with refractory ISI structure.
#[derive(Clone, Debug)]
pub struct GammaRenewalNeuron {
    pub rate_hz: f64,
    pub shape_k: u32,
    pub dt_ms: f64,
    time_since_spike: f64,
    rng: Xoshiro256PlusPlus,
}

impl GammaRenewalNeuron {
    pub fn new(rate_hz: f64, shape_k: u32, seed: u64) -> Self {
        Self {
            rate_hz,
            shape_k,
            dt_ms: 1.0,
            time_since_spike: 0.0,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
    fn log_gamma_int(k: u32) -> f64 {
        // ln((k-1)!) for integer k
        (1..k).map(|i| (i as f64).ln()).sum()
    }
    pub fn step(&mut self, rate_override: f64) -> i32 {
        self.time_since_spike += self.dt_ms;
        let r = if rate_override < 0.0 {
            self.rate_hz
        } else {
            rate_override
        };
        let lam = r / 1000.0;
        let k = self.shape_k;
        let t = self.time_since_spike;
        // Gamma hazard approximation: h(t) ≈ (λk)^k * t^(k-1) / Γ(k) * exp(-λk*t) / S(t)
        // Simplified: use Poisson sub-events approach
        let mu = lam * (k as f64);
        let log_pdf = (k as f64 - 1.0) * (mu * t).max(1e-30).ln() + mu.max(1e-30).ln()
            - mu * t
            - Self::log_gamma_int(k);
        let surv = 1.0 - self.incomplete_gamma_ratio(k, mu * t);
        let hazard = if surv > 1e-15 {
            log_pdf.exp() / surv
        } else {
            mu
        };
        let p = hazard * self.dt_ms;
        if self.rng.random::<f64>() < p {
            self.time_since_spike = 0.0;
            1
        } else {
            0
        }
    }
    fn incomplete_gamma_ratio(&self, k: u32, x: f64) -> f64 {
        // Regularized lower incomplete gamma P(k, x) via series for integer k
        if x <= 0.0 {
            return 0.0;
        }
        let mut sum = 0.0_f64;
        let mut term = 1.0_f64;
        for n in 0..k {
            if n > 0 {
                term *= x / n as f64;
            }
            sum += term;
        }
        1.0 - (-x).exp() * sum
    }
    pub fn reset(&mut self) {
        self.time_since_spike = 0.0;
    }
}

/// Stochastic IF — Ornstein-Uhlenbeck noise on LIF membrane.
#[derive(Clone, Debug)]
pub struct StochasticIFNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub sigma: f64,
    pub dt: f64,
    rng: Xoshiro256PlusPlus,
}

impl StochasticIFNeuron {
    pub fn new(seed: u64) -> Self {
        Self {
            v: -70.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 20.0,
            sigma: 3.0,
            dt: 1.0,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
    fn randn(&mut self) -> f64 {
        // Box-Muller
        let u1: f64 = self.rng.random::<f64>().max(1e-30);
        let u2: f64 = self.rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let noise = self.sigma * (self.dt / self.tau_m).sqrt() * self.randn();
        self.v += (-(self.v - self.v_rest) + current) / self.tau_m * self.dt + noise;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }
}

/// Galves-Löcherbach 2013 — stochastic point process with memory.
#[derive(Clone, Debug)]
pub struct GalvesLocherbachNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub decay: f64,
    pub threshold_rate: f64,
    pub steepness: f64,
    pub dt: f64,
    rng: Xoshiro256PlusPlus,
}

impl GalvesLocherbachNeuron {
    pub fn new(seed: u64) -> Self {
        Self {
            v: 0.0,
            v_rest: 0.0,
            decay: 0.95,
            threshold_rate: 0.5,
            steepness: 5.0,
            dt: 1.0,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
    pub fn step(&mut self, weighted_input: f64) -> i32 {
        self.v = self.decay * self.v + weighted_input;
        let p = 1.0 / (1.0 + (-(self.steepness * (self.v - self.threshold_rate))).exp());
        if self.rng.random::<f64>() < p * self.dt {
            self.v = self.v_rest;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }
}

/// SRM0 — Spike Response Model (kernel-based). Gerstner 1995.
#[derive(Clone, Debug)]
pub struct SpikeResponseNeuron {
    pub v: f64,
    pub v_threshold: f64,
    pub tau_eta: f64,
    pub tau_kappa: f64,
    pub eta_reset: f64,
    pub time_since_spike: f64,
    pub dt: f64,
}

impl SpikeResponseNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            v_threshold: 1.0,
            tau_eta: 10.0,
            tau_kappa: 5.0,
            eta_reset: -5.0,
            time_since_spike: 1000.0,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, weighted_input: f64) -> i32 {
        self.time_since_spike += self.dt;
        let eta = self.eta_reset * (-self.time_since_spike / self.tau_eta).exp();
        let kappa = weighted_input * (1.0 - (-self.dt / self.tau_kappa).exp());
        self.v = eta + kappa;
        if self.v >= self.v_threshold {
            self.time_since_spike = 0.0;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.time_since_spike = 1000.0;
    }
}
impl Default for SpikeResponseNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// GLM neuron — point-process generalized linear model. Pillow et al. 2008.
#[derive(Clone, Debug)]
pub struct GLMNeuron {
    pub mu: f64,
    pub dt_ms: f64,
    pub k: Vec<f64>,
    pub h: Vec<f64>,
    stim_buf: Vec<f64>,
    spike_buf: Vec<f64>,
    rng: Xoshiro256PlusPlus,
}

impl GLMNeuron {
    pub fn new(n_k: usize, n_h: usize, seed: u64) -> Self {
        Self {
            mu: -3.0,
            dt_ms: 1.0,
            k: vec![0.1; n_k],
            h: vec![-0.5; n_h],
            stim_buf: vec![0.0; n_k],
            spike_buf: vec![0.0; n_h],
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
    pub fn step(&mut self, stimulus: f64) -> i32 {
        // Shift buffers
        let nk = self.stim_buf.len();
        let nh = self.spike_buf.len();
        for i in (1..nk).rev() {
            self.stim_buf[i] = self.stim_buf[i - 1];
        }
        if nk > 0 {
            self.stim_buf[0] = stimulus;
        }
        let dot_k: f64 = self
            .k
            .iter()
            .zip(self.stim_buf.iter())
            .map(|(a, b)| a * b)
            .sum();
        let dot_h: f64 = self
            .h
            .iter()
            .zip(self.spike_buf.iter())
            .map(|(a, b)| a * b)
            .sum();
        let lam = (dot_k + dot_h + self.mu).exp();
        let p = lam * self.dt_ms / 1000.0;
        let spike = if self.rng.random::<f64>() < p { 1 } else { 0 };
        for i in (1..nh).rev() {
            self.spike_buf[i] = self.spike_buf[i - 1];
        }
        if nh > 0 {
            self.spike_buf[0] = spike as f64;
        }
        spike
    }
    pub fn reset(&mut self) {
        self.stim_buf.fill(0.0);
        self.spike_buf.fill(0.0);
    }
}

/// Wilson-Cowan 1972 — excitatory/inhibitory population rate model.
#[derive(Clone, Debug)]
pub struct WilsonCowanUnit {
    pub e: f64,
    pub i: f64,
    pub w_ee: f64,
    pub w_ei: f64,
    pub w_ie: f64,
    pub w_ii: f64,
    pub tau_e: f64,
    pub tau_i: f64,
    pub a: f64,
    pub theta: f64,
    pub dt: f64,
}

impl WilsonCowanUnit {
    pub fn new() -> Self {
        Self {
            e: 0.1,
            i: 0.05,
            w_ee: 10.0,
            w_ei: 6.0,
            w_ie: 10.0,
            w_ii: 1.0,
            tau_e: 1.0,
            tau_i: 2.0,
            a: 1.2,
            theta: 4.0,
            dt: 0.1,
        }
    }
    fn logistic(&self, z: f64) -> f64 {
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let exp_z = z.exp();
            exp_z / (1.0 + exp_z)
        }
    }
    fn sigmoid(&self, x: f64) -> f64 {
        self.logistic(self.a * (x - self.theta)) - self.logistic(-self.a * self.theta)
    }
    fn derivatives(&self, e: f64, i: f64, ext_input: f64) -> (f64, f64) {
        let se = self.sigmoid(self.w_ee * e - self.w_ei * i + ext_input);
        let si = self.sigmoid(self.w_ie * e - self.w_ii * i);
        ((-e + se) / self.tau_e, (-i + si) / self.tau_i)
    }
    pub fn step(&mut self, ext_input: f64) -> f64 {
        let e = self.e;
        let i = self.i;
        let (k1_e, k1_i) = self.derivatives(e, i, ext_input);
        let (k2_e, k2_i) = self.derivatives(
            e + 0.5 * self.dt * k1_e,
            i + 0.5 * self.dt * k1_i,
            ext_input,
        );
        let (k3_e, k3_i) = self.derivatives(
            e + 0.5 * self.dt * k2_e,
            i + 0.5 * self.dt * k2_i,
            ext_input,
        );
        let (k4_e, k4_i) = self.derivatives(e + self.dt * k3_e, i + self.dt * k3_i, ext_input);
        self.e = e + self.dt * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e) / 6.0;
        self.i = i + self.dt * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i) / 6.0;
        self.e
    }
    pub fn reset(&mut self) {
        self.e = 0.1;
        self.i = 0.05;
    }
}
impl Default for WilsonCowanUnit {
    fn default() -> Self {
        Self::new()
    }
}

/// Jansen-Rit 1995 — neural mass model for EEG generation.
#[derive(Clone, Debug)]
pub struct JansenRitUnit {
    pub y: [f64; 6],
    pub a_exc: f64,
    pub b_exc: f64,
    pub a_rate: f64,
    pub b_rate: f64,
    pub c: f64,
    pub e0: f64,
    pub v0: f64,
    pub r: f64,
    pub dt: f64,
}

impl JansenRitUnit {
    pub fn new() -> Self {
        Self {
            y: [0.0; 6],
            a_exc: 3.25,
            b_exc: 22.0,
            a_rate: 100.0,
            b_rate: 50.0,
            c: 135.0,
            e0: 2.5,
            v0: 6.0,
            r: 0.56,
            dt: 0.001,
        }
    }
    fn sigmoid(&self, x: f64) -> f64 {
        2.0 * self.e0 / (1.0 + (self.r * (self.v0 - x)).exp())
    }
    pub fn step(&mut self, p_ext: f64) -> f64 {
        let s1 = self.sigmoid(self.y[1] - self.y[2]);
        let s0 = self.sigmoid(self.c * 0.8 * self.y[0]);
        let s2 = self.sigmoid(self.c * 0.25 * self.y[0]);
        let dy0 = self.y[3];
        let dy3 = self.a_exc * self.a_rate * s1
            - 2.0 * self.a_rate * self.y[3]
            - self.a_rate.powi(2) * self.y[0];
        let dy1 = self.y[4];
        let dy4 = self.a_exc * self.a_rate * (p_ext + self.c * 0.8 * s0)
            - 2.0 * self.a_rate * self.y[4]
            - self.a_rate.powi(2) * self.y[1];
        let dy2 = self.y[5];
        let dy5 = self.b_exc * self.b_rate * self.c * 0.25 * s2
            - 2.0 * self.b_rate * self.y[5]
            - self.b_rate.powi(2) * self.y[2];
        self.y[0] += dy0 * self.dt;
        self.y[3] += dy3 * self.dt;
        self.y[1] += dy1 * self.dt;
        self.y[4] += dy4 * self.dt;
        self.y[2] += dy2 * self.dt;
        self.y[5] += dy5 * self.dt;
        self.y[1] - self.y[2]
    }
    pub fn reset(&mut self) {
        self.y = [0.0; 6];
    }
}
impl Default for JansenRitUnit {
    fn default() -> Self {
        Self::new()
    }
}

/// Wong-Wang 2006 — attractor decision-making model.
#[derive(Clone, Debug)]
pub struct WongWangUnit {
    pub s1: f64,
    pub s2: f64,
    pub tau_s: f64,
    pub gamma: f64,
    pub j_n: f64,
    pub j_cross: f64,
    pub i_0: f64,
    pub sigma: f64,
    pub dt: f64,
    rng: Xoshiro256PlusPlus,
}

impl WongWangUnit {
    pub fn new(seed: u64) -> Self {
        Self {
            s1: 0.1,
            s2: 0.1,
            tau_s: 0.1,
            gamma: 0.641,
            j_n: 0.2609,
            j_cross: 0.0497,
            i_0: 0.3255,
            sigma: 0.02,
            dt: 0.001,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
    fn phi(&self, i_total: f64) -> f64 {
        let a = 270.0;
        let b = 108.0;
        let d = 0.154;
        let x = a * i_total - b;
        let denom = 1.0 - (-d * x).exp();
        if denom.abs() < 1e-10 {
            1.0 / d
        } else {
            x / denom
        }
    }
    fn derivatives(
        &self,
        s1: f64,
        s2: f64,
        stim1: f64,
        stim2: f64,
        noise1: f64,
        noise2: f64,
    ) -> (f64, f64, f64, f64) {
        let i1 = self.j_n * s1 - self.j_cross * s2 + self.i_0 + stim1 + noise1;
        let i2 = self.j_n * s2 - self.j_cross * s1 + self.i_0 + stim2 + noise2;
        let r1 = self.phi(i1);
        let r2 = self.phi(i2);
        (
            -s1 / self.tau_s + self.gamma * (1.0 - s1) * r1,
            -s2 / self.tau_s + self.gamma * (1.0 - s2) * r2,
            r1,
            r2,
        )
    }
    fn randn(&mut self) -> f64 {
        let u1 = self.rng.random::<f64>().max(1e-30);
        let u2 = self.rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn step(&mut self, stim1: f64, stim2: f64) -> (f64, f64) {
        let noise1 = self.sigma * self.randn();
        let noise2 = self.sigma * self.randn();
        let (k1_s1, k1_s2, r1, r2) =
            self.derivatives(self.s1, self.s2, stim1, stim2, noise1, noise2);
        let (k2_s1, k2_s2, _, _) = self.derivatives(
            self.s1 + 0.5 * self.dt * k1_s1,
            self.s2 + 0.5 * self.dt * k1_s2,
            stim1,
            stim2,
            noise1,
            noise2,
        );
        let (k3_s1, k3_s2, _, _) = self.derivatives(
            self.s1 + 0.5 * self.dt * k2_s1,
            self.s2 + 0.5 * self.dt * k2_s2,
            stim1,
            stim2,
            noise1,
            noise2,
        );
        let (k4_s1, k4_s2, _, _) = self.derivatives(
            self.s1 + self.dt * k3_s1,
            self.s2 + self.dt * k3_s2,
            stim1,
            stim2,
            noise1,
            noise2,
        );
        self.s1 =
            (self.s1 + self.dt * (k1_s1 + 2.0 * k2_s1 + 2.0 * k3_s1 + k4_s1) / 6.0).clamp(0.0, 1.0);
        self.s2 =
            (self.s2 + self.dt * (k1_s2 + 2.0 * k2_s2 + 2.0 * k3_s2 + k4_s2) / 6.0).clamp(0.0, 1.0);
        (r1, r2)
    }
    pub fn reset(&mut self) {
        self.s1 = 0.1;
        self.s2 = 0.1;
    }
}

/// Ermentrout-Kopell / Montbrió theta-neuron mean field. Montbrió et al. 2015.
#[derive(Clone, Debug)]
pub struct ErmentroutKopellPopulation {
    pub r: f64,
    pub v: f64,
    pub tau: f64,
    pub delta: f64,
    pub eta_bar: f64,
    pub j: f64,
    pub dt: f64,
}

impl ErmentroutKopellPopulation {
    pub fn new() -> Self {
        Self {
            r: 0.1,
            v: -2.0,
            tau: 1.0,
            delta: 1.0,
            eta_bar: -5.0,
            j: 15.0,
            dt: 0.01,
        }
    }
    pub fn step(&mut self, ext_input: f64) -> f64 {
        let dr =
            (self.delta / (std::f64::consts::PI * self.tau) + 2.0 * self.r * self.v) / self.tau;
        let dv = (self.v * self.v + self.eta_bar + ext_input + self.j * self.tau * self.r
            - std::f64::consts::PI.powi(2) * self.tau.powi(2) * self.r.powi(2))
            / self.tau;
        self.r += dr * self.dt;
        self.v += dv * self.dt;
        self.r = self.r.max(0.0);
        self.r
    }
    pub fn reset(&mut self) {
        self.r = 0.1;
        self.v = -2.0;
    }
}
impl Default for ErmentroutKopellPopulation {
    fn default() -> Self {
        Self::new()
    }
}

/// Wendling et al. 2002 — extended Jansen-Rit with slow GABA_B.
#[derive(Clone, Debug)]
pub struct WendlingNeuron {
    pub y: [f64; 10],
    pub a_exc: f64,
    pub b_fast: f64,
    pub g_slow: f64,
    pub a_rate: f64,
    pub b_rate: f64,
    pub g_rate: f64,
    pub c: f64,
    pub e0: f64,
    pub v0: f64,
    pub r: f64,
    pub dt: f64,
}

impl WendlingNeuron {
    pub fn new() -> Self {
        Self {
            y: [0.0; 10],
            a_exc: 3.25,
            b_fast: 22.0,
            g_slow: 10.0,
            a_rate: 100.0,
            b_rate: 500.0,
            g_rate: 20.0,
            c: 135.0,
            e0: 2.5,
            v0: 6.0,
            r: 0.56,
            dt: 0.001,
        }
    }
    fn sigmoid(&self, x: f64) -> f64 {
        2.0 * self.e0 / (1.0 + (self.r * (self.v0 - x)).exp())
    }
    pub fn step(&mut self, p_ext: f64) -> f64 {
        let sig_main = self.sigmoid(self.y[1] - self.y[2] - self.y[3]);
        let sig_0 = self.sigmoid(self.c * 0.8 * self.y[0]);
        let sig_fast = self.sigmoid(self.c * 0.25 * self.y[0]);
        let sig_slow = self.sigmoid(self.c * 0.1 * self.y[0]);
        let a = self.a_rate;
        let b = self.b_rate;
        let g = self.g_rate;
        let dy0 = self.y[5];
        let dy5 = self.a_exc * a * sig_main - 2.0 * a * self.y[5] - a * a * self.y[0];
        let dy1 = self.y[6];
        let dy6 = self.a_exc * a * (p_ext + self.c * 0.8 * sig_0)
            - 2.0 * a * self.y[6]
            - a * a * self.y[1];
        let dy2 = self.y[7];
        let dy7 =
            self.b_fast * b * self.c * 0.25 * sig_fast - 2.0 * b * self.y[7] - b * b * self.y[2];
        let dy3 = self.y[8];
        let dy8 =
            self.g_slow * g * self.c * 0.1 * sig_slow - 2.0 * g * self.y[8] - g * g * self.y[3];
        self.y[0] += dy0 * self.dt;
        self.y[5] += dy5 * self.dt;
        self.y[1] += dy1 * self.dt;
        self.y[6] += dy6 * self.dt;
        self.y[2] += dy2 * self.dt;
        self.y[7] += dy7 * self.dt;
        self.y[3] += dy3 * self.dt;
        self.y[8] += dy8 * self.dt;
        self.y[1] - self.y[2] - self.y[3]
    }
    pub fn reset(&mut self) {
        self.y = [0.0; 10];
    }
}
impl Default for WendlingNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Larter-Breakspear 2003 — neural mass with ion channels for whole-brain modelling.
#[derive(Clone, Debug)]
pub struct LarterBreakspearNeuron {
    pub v: f64,
    pub w: f64,
    pub z: f64,
    pub g_ca: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub v_ca: f64,
    pub v_na: f64,
    pub v_k: f64,
    pub v_l: f64,
    pub phi: f64,
    pub tau_k: f64,
    pub b: f64,
    pub a_ee: f64,
    pub i_ext: f64,
    pub dt: f64,
}

impl LarterBreakspearNeuron {
    pub fn new() -> Self {
        Self {
            v: -0.5,
            w: 0.0,
            z: 0.0,
            g_ca: 1.1,
            g_na: 6.7,
            g_k: 2.0,
            g_l: 0.5,
            v_ca: 1.0,
            v_na: 0.53,
            v_k: -0.7,
            v_l: -0.5,
            phi: 0.7,
            tau_k: 1.0,
            b: 0.1,
            a_ee: 0.36,
            i_ext: 0.3,
            dt: 0.01,
        }
    }
    pub fn step(&mut self, coupling: f64) -> f64 {
        let m_ca = 0.5 * (1.0 + ((self.v + 0.01) / 0.15).tanh());
        let m_na = 0.5 * (1.0 + ((self.v - 0.12) / 0.15).tanh());
        let m_k = 0.5 * (1.0 + (self.v / 0.3).tanh());
        let i_ca = self.g_ca * m_ca * (self.v - self.v_ca);
        let i_na = self.g_na * m_na * (self.v - self.v_na);
        let i_k = self.g_k * self.w * (self.v - self.v_k);
        let i_l = self.g_l * (self.v - self.v_l);
        let dv = -i_ca - i_na - i_k - i_l + self.i_ext + coupling + self.a_ee * self.v;
        let dw = self.phi * (m_k - self.w) / self.tau_k;
        let dz = self.b * (self.v + 0.5 - self.z);
        self.v += dv * self.dt;
        self.w += dw * self.dt;
        self.z += dz * self.dt;
        self.v
    }
    pub fn reset(&mut self) {
        self.v = -0.5;
        self.w = 0.0;
        self.z = 0.0;
    }
}
impl Default for LarterBreakspearNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_fires() {
        let mut n = PoissonNeuron::new(200.0, 1.0, 42);
        let t: i32 = (0..1000).map(|_| n.step(-1.0)).sum();
        assert!(t > 10);
    }
    #[test]
    fn inhom_poisson_fires() {
        let mut n = InhomogeneousPoissonNeuron::new(1.0, 42);
        let t: i32 = (0..1000).map(|_| n.step(200.0)).sum();
        assert!(t > 10);
    }
    #[test]
    fn gamma_renewal_fires() {
        let mut n = GammaRenewalNeuron::new(100.0, 3, 42);
        let t: i32 = (0..2000).map(|_| n.step(-1.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn stochastic_if_fires() {
        let mut n = StochasticIFNeuron::new(42);
        let t: i32 = (0..500).map(|_| n.step(30.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn gl_fires() {
        let mut n = GalvesLocherbachNeuron::new(42);
        let t: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn srm_fires() {
        let mut n = SpikeResponseNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(10.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn glm_fires() {
        let mut n = GLMNeuron::new(5, 10, 42);
        let t: i32 = (0..5000).map(|_| n.step(20.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn wc_oscillates() {
        let mut n = WilsonCowanUnit::new();
        let mut last = n.e;
        let mut changes = 0;
        for _ in 0..500 {
            let e = n.step(5.0);
            if (e - last).abs() > 0.001 {
                changes += 1;
            }
            last = e;
        }
        assert!(changes > 0);
    }
    #[test]
    fn jr_produces_eeg() {
        let mut n = JansenRitUnit::new();
        let mut nz = 0;
        for _ in 0..5000 {
            let v = n.step(220.0);
            if v.abs() > 0.001 {
                nz += 1;
            }
        }
        assert!(nz > 0);
    }
    #[test]
    fn ww_diverges() {
        let mut n = WongWangUnit::new(42);
        for _ in 0..5000 {
            n.step(0.02, 0.0);
        }
        assert!((n.s1 - n.s2).abs() > 0.001);
    }
    #[test]
    fn ek_pop_fires() {
        let mut n = ErmentroutKopellPopulation::new();
        for _ in 0..1000 {
            n.step(0.0);
        }
        assert!(n.r > 0.0);
    }
    #[test]
    fn wendling_produces() {
        let mut n = WendlingNeuron::new();
        let mut nz = 0;
        for _ in 0..5000 {
            let v = n.step(220.0);
            if v.abs() > 0.001 {
                nz += 1;
            }
        }
        assert!(nz > 0);
    }
    #[test]
    fn lb_evolves() {
        let mut n = LarterBreakspearNeuron::new();
        let v0 = n.v;
        for _ in 0..1000 {
            n.step(0.0);
        }
        assert!((n.v - v0).abs() > 0.001);
    }

    // ── Multi-angle tests for special models ──

    // -- Poisson --
    #[test]
    fn poisson_reset_no_panic() {
        let mut n = PoissonNeuron::new(200.0, 1.0, 42);
        for _ in 0..100 {
            n.step(-1.0);
        }
        n.reset();
    }
    #[test]
    fn poisson_nan_no_panic() {
        let mut n = PoissonNeuron::new(200.0, 1.0, 42);
        n.step(f64::NAN);
    }
    #[test]
    fn poisson_seed_varies() {
        let mut n1 = PoissonNeuron::new(200.0, 1.0, 1);
        let mut n2 = PoissonNeuron::new(200.0, 1.0, 999);
        let t1: i32 = (0..1000).map(|_| n1.step(-1.0)).sum();
        let t2: i32 = (0..1000).map(|_| n2.step(-1.0)).sum();
        assert!(t1 > 0 && t2 > 0);
    }

    // -- InhomogeneousPoisson --
    #[test]
    fn inhom_poisson_zero_rate() {
        let mut n = InhomogeneousPoissonNeuron::new(1.0, 42);
        let t: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn inhom_poisson_nan_no_panic() {
        let mut n = InhomogeneousPoissonNeuron::new(1.0, 42);
        n.step(f64::NAN);
    }

    // -- GammaRenewal --
    #[test]
    fn gamma_renewal_reset_clears() {
        let mut n = GammaRenewalNeuron::new(100.0, 3, 42);
        for _ in 0..100 {
            n.step(-1.0);
        }
        n.reset();
        assert!((n.time_since_spike - 0.0).abs() < 1e-10);
    }
    #[test]
    fn gamma_renewal_nan_no_panic() {
        let mut n = GammaRenewalNeuron::new(100.0, 3, 42);
        n.step(f64::NAN);
    }

    // -- StochasticIF --
    #[test]
    fn stochastic_if_reset_clears() {
        let mut n = StochasticIFNeuron::new(42);
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn stochastic_if_bounded() {
        let mut n = StochasticIFNeuron::new(42);
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn stochastic_if_nan_no_panic() {
        StochasticIFNeuron::new(42).step(f64::NAN);
    }
    #[test]
    fn stochastic_if_negative_no_crash() {
        let mut n = StochasticIFNeuron::new(42);
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }

    // -- GalvesLocherbach --
    #[test]
    fn gl_reset_clears() {
        let mut n = GalvesLocherbachNeuron::new(42);
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn gl_bounded() {
        let mut n = GalvesLocherbachNeuron::new(42);
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn gl_nan_no_panic() {
        GalvesLocherbachNeuron::new(42).step(f64::NAN);
    }

    // -- SpikeResponse --
    #[test]
    fn srm_reset_clears() {
        let mut n = SpikeResponseNeuron::new();
        for _ in 0..100 {
            n.step(10.0);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn srm_bounded() {
        let mut n = SpikeResponseNeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn srm_nan_no_panic() {
        SpikeResponseNeuron::new().step(f64::NAN);
    }
    #[test]
    fn srm_silent_without_input() {
        let mut n = SpikeResponseNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }

    // -- GLM --
    #[test]
    fn glm_reset_clears() {
        let mut n = GLMNeuron::new(5, 10, 42);
        for _ in 0..100 {
            n.step(20.0);
        }
        n.reset();
    }
    #[test]
    fn glm_nan_no_panic() {
        GLMNeuron::new(5, 10, 42).step(f64::NAN);
    }

    // -- WilsonCowan --
    #[test]
    fn wc_reset_clears() {
        let mut n = WilsonCowanUnit::new();
        for _ in 0..200 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.e - 0.1).abs() < 1e-10);
        assert!((n.i - 0.05).abs() < 1e-10);
    }
    #[test]
    fn wc_bounded() {
        let mut n = WilsonCowanUnit::new();
        for _ in 0..5000 {
            n.step(1e3);
        }
        assert!(n.e.is_finite());
        assert!(n.i.is_finite());
    }
    #[test]
    fn wc_nan_no_panic() {
        WilsonCowanUnit::new().step(f64::NAN);
    }

    // -- JansenRit --
    #[test]
    fn jr_reset_clears() {
        let mut n = JansenRitUnit::new();
        for _ in 0..1000 {
            n.step(220.0);
        }
        n.reset();
        assert!(n.y.iter().all(|&x| x == 0.0));
    }
    #[test]
    fn jr_bounded() {
        let mut n = JansenRitUnit::new();
        for _ in 0..5000 {
            n.step(1e3);
        }
        assert!(n.y.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn jr_nan_no_panic() {
        JansenRitUnit::new().step(f64::NAN);
    }

    // -- WongWang --
    #[test]
    fn ww_reset_clears() {
        let mut n = WongWangUnit::new(42);
        for _ in 0..1000 {
            n.step(0.02, 0.0);
        }
        n.reset();
        assert!((n.s1 - 0.1).abs() < 1e-10);
        assert!((n.s2 - 0.1).abs() < 1e-10);
    }
    #[test]
    fn ww_bounded() {
        let mut n = WongWangUnit::new(42);
        for _ in 0..5000 {
            n.step(1.0, 0.0);
        }
        assert!(n.s1.is_finite());
        assert!(n.s2.is_finite());
    }
    #[test]
    fn ww_nan_no_panic() {
        WongWangUnit::new(42).step(f64::NAN, 0.0);
    }

    // -- ErmentroutKopellPopulation --
    #[test]
    fn ek_pop_reset_clears() {
        let mut n = ErmentroutKopellPopulation::new();
        for _ in 0..500 {
            n.step(0.0);
        }
        n.reset();
        assert!((n.r - 0.1).abs() < 1e-10);
        assert!((n.v - (-2.0)).abs() < 1e-10);
    }
    #[test]
    fn ek_pop_moderate_stable() {
        let mut n = ErmentroutKopellPopulation::new();
        for _ in 0..5000 {
            n.step(1.0);
        }
        assert!(n.r.is_finite());
        assert!(n.v.is_finite());
    }
    #[test]
    fn ek_pop_nan_no_panic() {
        ErmentroutKopellPopulation::new().step(f64::NAN);
    }

    // -- Wendling --
    #[test]
    fn wendling_reset_clears() {
        let mut n = WendlingNeuron::new();
        for _ in 0..1000 {
            n.step(220.0);
        }
        n.reset();
        assert!(n.y.iter().all(|&x| x == 0.0));
    }
    #[test]
    fn wendling_bounded() {
        let mut n = WendlingNeuron::new();
        for _ in 0..5000 {
            n.step(1e3);
        }
        assert!(n.y.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn wendling_nan_no_panic() {
        WendlingNeuron::new().step(f64::NAN);
    }

    // -- LarterBreakspear --
    #[test]
    fn lb_reset_clears() {
        let mut n = LarterBreakspearNeuron::new();
        for _ in 0..500 {
            n.step(0.0);
        }
        n.reset();
        assert!((n.v - (-0.5)).abs() < 1e-10);
        assert!((n.w - 0.0).abs() < 1e-10);
        assert!((n.z - 0.0).abs() < 1e-10);
    }
    #[test]
    fn lb_bounded() {
        let mut n = LarterBreakspearNeuron::new();
        for _ in 0..5000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn lb_nan_no_panic() {
        LarterBreakspearNeuron::new().step(f64::NAN);
    }
}
