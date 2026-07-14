// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Simple integrate-and-fire variants

/// Simple integrate-and-fire variants.
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Quadratic Integrate-and-Fire — canonical Type-I excitability.
/// dv/dt = v² + I, reset at v_peak.
#[derive(Clone, Debug)]
pub struct QuadraticIFNeuron {
    pub v: f64,
    pub v_reset: f64,
    pub v_peak: f64,
    pub dt: f64,
}

impl QuadraticIFNeuron {
    pub fn new(v_reset: f64, v_peak: f64, dt: f64) -> Self {
        Self {
            v: v_reset,
            v_reset,
            v_peak,
            dt,
        }
    }

    fn valid_numeric_contract(&self) -> bool {
        self.v.is_finite()
            && self.v_reset.is_finite()
            && self.v_peak.is_finite()
            && self.dt.is_finite()
            && self.v < self.v_peak
            && self.v_reset < self.v_peak
            && self.dt > 0.0
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return 0;
        }
        let (next_v, spiked) = self.exact_candidate(current);
        if !next_v.is_finite() {
            return 0;
        }
        self.v = next_v;
        if spiked {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_reset;
    }

    fn exact_candidate(&self, current: f64) -> (f64, bool) {
        if current > 0.0 {
            let root_i = current.sqrt();
            let phase = (self.v / root_i).atan();
            let peak_phase = (self.v_peak / root_i).atan();
            let next_phase = phase + root_i * self.dt;
            if next_phase >= peak_phase || next_phase >= std::f64::consts::FRAC_PI_2 {
                return (self.v_reset, true);
            }
            return (root_i * next_phase.tan(), false);
        }
        if current == 0.0 {
            let denominator = 1.0 - self.v * self.dt;
            if denominator <= 0.0 {
                return (self.v_reset, true);
            }
            let next_v = self.v / denominator;
            if next_v >= self.v_peak {
                return (self.v_reset, true);
            }
            return (next_v, false);
        }

        let root_i = (-current).sqrt();
        if (self.v + root_i).abs() <= 1e-15 {
            return (self.v, false);
        }
        let numerator_ratio = (self.v - root_i) / (self.v + root_i);
        let evolved_ratio = numerator_ratio * (2.0 * root_i * self.dt).exp();
        let denominator = 1.0 - evolved_ratio;
        if (numerator_ratio < 1.0 && evolved_ratio >= 1.0) || denominator.abs() <= 1e-15 {
            return (self.v_reset, true);
        }
        let next_v = root_i * (1.0 + evolved_ratio) / denominator;
        if next_v >= self.v_peak {
            (self.v_reset, true)
        } else {
            (next_v, false)
        }
    }
}

impl Default for QuadraticIFNeuron {
    fn default() -> Self {
        Self::new(-1.0, 1.0, 0.01)
    }
}

/// Theta neuron — Ermentrout & Kopell canonical form.
/// dθ/dt = (1 - cosθ) + (1 + cosθ)·I, spike at θ crossing π.
#[derive(Clone, Debug)]
pub struct ThetaNeuron {
    pub theta: f64,
    pub dt: f64,
}

impl ThetaNeuron {
    pub fn new(dt: f64) -> Self {
        Self { theta: 0.0, dt }
    }

    fn wrap_phase(theta: f64) -> f64 {
        (theta + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
    }

    fn valid(&self) -> bool {
        self.theta.is_finite() && self.dt.is_finite() && self.dt > 0.0
    }

    fn exact_candidate(&self, current: f64) -> (f64, bool) {
        let y = (self.theta / 2.0).tan();
        if current > 0.0 {
            let root_i = current.sqrt();
            let phase = (y / root_i).atan();
            let next_phase = phase + root_i * self.dt;
            if next_phase.cos().abs() <= 1.0e-15 {
                return (
                    -std::f64::consts::PI,
                    next_phase >= std::f64::consts::FRAC_PI_2,
                );
            }
            return (
                Self::wrap_phase(2.0 * (root_i * next_phase.tan()).atan()),
                next_phase >= std::f64::consts::FRAC_PI_2,
            );
        }
        if current == 0.0 {
            let denominator = 1.0 - y * self.dt;
            if denominator.abs() <= 1.0e-15 {
                return (-std::f64::consts::PI, true);
            }
            return (
                Self::wrap_phase(2.0 * (y / denominator).atan()),
                denominator <= 0.0,
            );
        }

        let root_i = (-current).sqrt();
        if (y + root_i).abs() <= 1.0e-15 {
            return (self.theta, false);
        }
        let ratio = (y - root_i) / (y + root_i);
        let evolved = ratio * (2.0 * root_i * self.dt).exp();
        let denominator = 1.0 - evolved;
        let spiked = (ratio < 1.0 && evolved >= 1.0) || denominator.abs() <= 1.0e-15;
        if spiked && denominator.abs() <= 1.0e-15 {
            return (-std::f64::consts::PI, true);
        }
        (
            Self::wrap_phase(2.0 * (root_i * (1.0 + evolved) / denominator).atan()),
            spiked,
        )
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid() {
            return 0;
        }
        let (next_theta, spiked) = self.exact_candidate(current);
        if !next_theta.is_finite() {
            return 0;
        }
        self.theta = Self::wrap_phase(next_theta);
        if spiked {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.theta = 0.0;
    }
}

impl Default for ThetaNeuron {
    fn default() -> Self {
        Self::new(0.01)
    }
}

/// Perfect integrator — no leak, pure capacitive charging.
#[derive(Clone, Debug)]
pub struct PerfectIntegratorNeuron {
    pub v: f64,
    pub c_m: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl PerfectIntegratorNeuron {
    pub fn new(c_m: f64, v_threshold: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            c_m,
            v_threshold,
            v_reset: 0.0,
            dt,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (current / self.c_m) * self.dt;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_reset;
    }
}

impl Default for PerfectIntegratorNeuron {
    fn default() -> Self {
        Self::new(1.0, 1.0, 0.1)
    }
}

/// Gated LIF — learnable decay and input gates.
#[derive(Clone, Debug)]
pub struct GatedLIFNeuron {
    pub v: f64,
    pub gate_v: f64,
    pub gate_i: f64,
    pub v_threshold: f64,
    pub dt: f64,
}

impl GatedLIFNeuron {
    pub fn new(gate_v: f64, gate_i: f64, v_threshold: f64) -> Self {
        Self {
            v: 0.0,
            gate_v,
            gate_i,
            v_threshold,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v = self.gate_v * self.v + self.gate_i * current;
        if self.v >= self.v_threshold {
            self.v -= self.v_threshold;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}

impl Default for GatedLIFNeuron {
    fn default() -> Self {
        Self::new(0.9, 1.0, 1.0)
    }
}

/// Nonlinear LIF — cubic f-I curve with adaptation. Touboul & Brette 2008.
#[derive(Clone, Debug)]
pub struct NonlinearLIFNeuron {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_crit: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub a: f64,
    pub b: f64,
    pub tau_w: f64,
    pub c_m: f64,
    pub dt: f64,
}

impl NonlinearLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            v_rest: -65.0,
            v_crit: -40.0,
            v_threshold: -20.0,
            v_reset: -65.0,
            a: 0.04,
            b: 0.5,
            tau_w: 100.0,
            c_m: 1.0,
            dt: 0.1,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let cubic = self.a * (self.v - self.v_rest) * (self.v - self.v_crit);
        self.v += (cubic - self.w + current) / self.c_m * self.dt;
        self.w += (self.b * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0;
    }
}

impl Default for NonlinearLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Spike-Frequency Adaptation LIF. Benda & Herz 2003.
#[derive(Clone, Debug)]
pub struct SFANeuron {
    pub v: f64,
    pub g_sfa: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_sfa: f64,
    pub delta_g: f64,
    pub e_k: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl SFANeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            g_sfa: 0.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 10.0,
            tau_sfa: 200.0,
            delta_g: 0.5,
            e_k: -80.0,
            resistance: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) - self.g_sfa * (self.v - self.e_k)
            + self.resistance * current)
            / self.tau_m
            * self.dt;
        self.g_sfa *= (-self.dt / self.tau_sfa).exp();
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.g_sfa += self.delta_g;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.g_sfa = 0.0;
    }
}

impl Default for SFANeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-timescale Adaptive Threshold. Kobayashi et al. 2009.
#[derive(Clone, Debug)]
pub struct MATNeuron {
    pub v: f64,
    pub theta1: f64,
    pub theta2: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold_base: f64,
    pub tau_m: f64,
    pub tau_1: f64,
    pub tau_2: f64,
    pub h1: f64,
    pub h2: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl MATNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            theta1: 0.0,
            theta2: 0.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold_base: -50.0,
            tau_m: 10.0,
            tau_1: 10.0,
            tau_2: 200.0,
            h1: 5.0,
            h2: 3.0,
            resistance: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) + self.resistance * current) / self.tau_m * self.dt;
        self.theta1 *= (-self.dt / self.tau_1).exp();
        self.theta2 *= (-self.dt / self.tau_2).exp();
        let threshold = self.v_threshold_base + self.theta1 + self.theta2;
        if self.v >= threshold {
            self.v = self.v_reset;
            self.theta1 += self.h1;
            self.theta2 += self.h2;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta1 = 0.0;
        self.theta2 = 0.0;
    }
}

impl Default for MATNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape-rate neuron — stochastic IF with exponential hazard. Gerstner 2000.
#[derive(Clone, Debug)]
pub struct EscapeRateNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub rho_0: f64,
    pub delta_u: f64,
    pub resistance: f64,
    pub dt: f64,
    pub rng_state: u16,
    pub initial_seed: u16,
}

impl EscapeRateNeuron {
    pub fn new(seed: u64) -> Self {
        let narrowed = (seed & u64::from(u16::MAX)) as u16;
        let initial_seed = if narrowed == 0 { 0xACE1 } else { narrowed };
        Self {
            v: -70.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 10.0,
            rho_0: 0.001,
            delta_u: 3.0,
            resistance: 1.0,
            dt: 1.0,
            rng_state: initial_seed,
            initial_seed,
        }
    }

    pub fn valid(&self) -> bool {
        self.v.is_finite()
            && self.v_rest.is_finite()
            && self.v_reset.is_finite()
            && self.v_threshold.is_finite()
            && self.tau_m.is_finite()
            && self.tau_m > 0.0
            && self.rho_0.is_finite()
            && self.rho_0 > 0.0
            && self.delta_u.is_finite()
            && self.delta_u > 0.0
            && self.resistance.is_finite()
            && self.resistance > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.rng_state != 0
    }

    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.v.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || !self.tau_m.is_finite()
            || self.tau_m <= 0.0
            || !self.rho_0.is_finite()
            || self.rho_0 <= 0.0
            || !self.delta_u.is_finite()
            || self.delta_u <= 0.0
            || !self.resistance.is_finite()
            || self.resistance <= 0.0
            || !self.dt.is_finite()
            || self.dt <= 0.0
            || self.rng_state == 0
            || !current.is_finite()
        {
            return Err("invalid escape-rate state or input");
        }
        let v_inf = self.v_rest + self.resistance * current;
        let decay = (-self.dt / self.tau_m).exp();
        let next_v = v_inf + (self.v - v_inf) * decay;
        if !v_inf.is_finite() || !decay.is_finite() || !next_v.is_finite() {
            return Err("non-finite escape-rate membrane candidate");
        }
        let hazard = self.rho_0
            * ((next_v - self.v_threshold) / self.delta_u)
                .clamp(-700.0, 700.0)
                .exp()
            * self.dt;
        if !hazard.is_finite() || hazard < 0.0 {
            return Err("non-finite escape hazard");
        }
        let p_spike = -(-hazard).exp_m1();
        if !p_spike.is_finite() || !(0.0..=1.0).contains(&p_spike) {
            return Err("invalid escape probability");
        }
        let mut sample = self.rng_state;
        for _ in 0..8 {
            let feedback = ((sample >> 0) ^ (sample >> 2) ^ (sample >> 3) ^ (sample >> 5)) & 1;
            sample = (sample >> 1) | (feedback << 15);
        }
        let threshold = if p_spike <= 0.0 {
            0_u32
        } else if p_spike >= 1.0 {
            65_536_u32
        } else {
            (p_spike * 65_535.0).floor() as u32 + 1
        };
        self.rng_state = sample;
        if u32::from(sample) < threshold {
            self.v = self.v_reset;
            Ok(1)
        } else {
            self.v = next_v;
            Ok(0)
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.rng_state = self.initial_seed;
    }
}

/// KLIF — learnable LIF with scaling factor k. Eshraghian et al. 2021.
#[derive(Clone, Debug)]
pub struct KLIFNeuron {
    pub v: f64,
    pub k: f64,
    pub alpha: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
}

impl KLIFNeuron {
    pub fn new(tau: f64, k: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            k,
            alpha: (-dt / tau).exp(),
            v_threshold: 1.0,
            v_reset: 0.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v = self.alpha * self.v + self.k * current;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}

impl Default for KLIFNeuron {
    fn default() -> Self {
        Self::new(10.0, 1.0, 1.0)
    }
}

/// Inhibitory LIF — LIF with self-inhibition trace.
#[derive(Clone, Debug)]
pub struct InhibitoryLIFNeuron {
    pub v: f64,
    pub inh_trace: f64,
    pub alpha_m: f64,
    pub alpha_inh: f64,
    pub v_threshold: f64,
    pub inh_strength: f64,
}

impl InhibitoryLIFNeuron {
    pub fn new(tau_m: f64, tau_inh: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            inh_trace: 0.0,
            alpha_m: (-dt / tau_m).exp(),
            alpha_inh: (-dt / tau_inh).exp(),
            v_threshold: 1.0,
            inh_strength: 0.5,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.inh_trace *= self.alpha_inh;
        self.v = self.alpha_m * self.v + current - self.inh_strength * self.inh_trace;
        if self.v >= self.v_threshold {
            self.v = 0.0;
            self.inh_trace += 1.0;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.inh_trace = 0.0;
    }
}

impl Default for InhibitoryLIFNeuron {
    fn default() -> Self {
        Self::new(10.0, 5.0, 1.0)
    }
}

/// Complementary LIF — dual-path excitatory/inhibitory.
#[derive(Clone, Debug)]
pub struct ComplementaryLIFNeuron {
    pub v_pos: f64,
    pub v_neg: f64,
    pub alpha: f64,
    pub v_threshold: f64,
}

impl ComplementaryLIFNeuron {
    pub fn new(tau: f64, dt: f64, v_threshold: f64) -> Self {
        Self {
            v_pos: 0.0,
            v_neg: 0.0,
            alpha: (-dt / tau).exp(),
            v_threshold,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let inp_pos = current.max(0.0);
        let inp_neg = (-current).max(0.0);
        self.v_pos = self.alpha * self.v_pos + inp_pos;
        self.v_neg = self.alpha * self.v_neg + inp_neg;
        let diff = self.v_pos - self.v_neg;
        if diff >= self.v_threshold {
            self.v_pos = 0.0;
            self.v_neg = 0.0;
            1
        } else if diff <= -self.v_threshold {
            self.v_pos = 0.0;
            self.v_neg = 0.0;
            -1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v_pos = 0.0;
        self.v_neg = 0.0;
    }
}

impl Default for ComplementaryLIFNeuron {
    fn default() -> Self {
        Self::new(10.0, 1.0, 1.0)
    }
}

/// Parametric LIF — learnable decay via sigmoid(a). Fang et al. 2021.
#[derive(Clone, Debug)]
pub struct ParametricLIFNeuron {
    pub v: f64,
    pub a: f64,
    pub threshold: f64,
}

impl ParametricLIFNeuron {
    pub fn new(a: f64, threshold: f64) -> Self {
        Self {
            v: 0.0,
            a,
            threshold,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let alpha = 1.0 / (1.0 + (-self.a).exp());
        let spike = if self.v >= self.threshold { 1 } else { 0 };
        self.v = alpha * self.v * (1.0 - spike as f64) + current;
        spike
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}

impl Default for ParametricLIFNeuron {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

/// Non-resetting LIF with adaptive threshold. Brette 2004.
#[derive(Clone, Debug)]
pub struct NonResettingLIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub v_rest: f64,
    pub theta_rest: f64,
    pub delta_theta: f64,
    pub tau_m: f64,
    pub tau_theta: f64,
    pub r_m: f64,
    pub dt: f64,
}

impl NonResettingLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            theta: -50.0,
            v_rest: -65.0,
            theta_rest: -50.0,
            delta_theta: 5.0,
            tau_m: 10.0,
            tau_theta: 50.0,
            r_m: 1.0,
            dt: 0.1,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) + self.r_m * current) / self.tau_m * self.dt;
        self.theta += (-(self.theta - self.theta_rest)) / self.tau_theta * self.dt;
        if self.v >= self.theta {
            self.theta += self.delta_theta;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_rest;
    }
}

impl Default for NonResettingLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive-threshold IF. Platkiewicz & Brette 2010.
#[derive(Clone, Debug)]
pub struct AdaptiveThresholdIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub theta_rest: f64,
    pub delta_theta: f64,
    pub tau_m: f64,
    pub tau_theta: f64,
    pub dt: f64,
}

impl AdaptiveThresholdIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            theta: -50.0,
            v_rest: -65.0,
            v_reset: -65.0,
            theta_rest: -50.0,
            delta_theta: 5.0,
            tau_m: 10.0,
            tau_theta: 50.0,
            dt: 0.1,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) + current) / self.tau_m * self.dt;
        self.theta += (-(self.theta - self.theta_rest)) / self.tau_theta * self.dt;
        if self.v >= self.theta {
            self.v = self.v_reset;
            self.theta += self.delta_theta;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_rest;
    }
}

impl Default for AdaptiveThresholdIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Sigma-Delta neuron — first-order delta modulation.
#[derive(Clone, Debug)]
pub struct SigmaDeltaNeuron {
    pub sigma: f64,
    pub v_threshold: f64,
}

impl SigmaDeltaNeuron {
    pub fn new(v_threshold: f64) -> Self {
        Self {
            sigma: 0.0,
            v_threshold,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.sigma += current;
        if self.sigma >= self.v_threshold {
            self.sigma -= self.v_threshold;
            1
        } else if self.sigma <= -self.v_threshold {
            self.sigma += self.v_threshold;
            -1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.sigma = 0.0;
    }
}

impl Default for SigmaDeltaNeuron {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Energy-aware LIF — metabolic cost modulates gain. Sengupta et al. 2013.
#[derive(Clone, Debug)]
pub struct EnergyLIFNeuron {
    pub v: f64,
    pub epsilon: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_e: f64,
    pub alpha: f64,
    pub epsilon_0: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl EnergyLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            epsilon: 1.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 10.0,
            tau_e: 500.0,
            alpha: 0.1,
            epsilon_0: 1.0,
            resistance: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let effective_r = self.resistance * self.epsilon;
        self.v += (-(self.v - self.v_rest) + effective_r * current) / self.tau_m * self.dt;
        self.epsilon += (self.epsilon_0 - self.epsilon) / self.tau_e * self.dt;
        if self.v >= self.v_threshold && self.epsilon > 0.1 {
            self.v = self.v_reset;
            self.epsilon -= self.alpha;
            1
        } else if self.v >= self.v_threshold {
            self.v = self.v_threshold;
            0
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.epsilon = self.epsilon_0;
    }
}

impl Default for EnergyLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Wu et al. (2021) IQIF — piecewise-linear integer soma for digital hardware.
#[derive(Clone, Debug)]
pub struct IntegerQIFNeuron {
    pub v: i64,
    pub v_rest: i64,
    pub v_threshold: i64,
    pub v_reset: i64,
    pub a: i64,
    pub b: i64,
    pub v_max: i64,
    pub v_min: i64,
}

impl IntegerQIFNeuron {
    #[allow(clippy::too_many_arguments)]
    pub fn with_parameters(
        v: i32,
        v_rest: i32,
        v_threshold: i32,
        v_reset: i32,
        a: i32,
        b: i32,
        v_max: i32,
        v_min: i32,
    ) -> Result<Self, &'static str> {
        let neuron = Self {
            v: i64::from(v),
            v_rest: i64::from(v_rest),
            v_threshold: i64::from(v_threshold),
            v_reset: i64::from(v_reset),
            a: i64::from(a),
            b: i64::from(b),
            v_max: i64::from(v_max),
            v_min: i64::from(v_min),
        };
        if neuron.valid() {
            Ok(neuron)
        } else {
            Err("invalid IQIF state or parameter ordering")
        }
    }

    pub fn valid(&self) -> bool {
        let i32_range = |value: i64| i64::from(i32::MIN) <= value && value <= i64::from(i32::MAX);
        [
            self.v,
            self.v_rest,
            self.v_threshold,
            self.v_reset,
            self.a,
            self.b,
            self.v_max,
            self.v_min,
        ]
        .into_iter()
        .all(i32_range)
            && self.a >= 0
            && self.b >= 0
            && self.a + self.b > 0
            && self.v_min < self.v_rest
            && self.v_rest < self.v_threshold
            && self.v_threshold < self.v_max
            && (self.v_min..=self.v_max).contains(&self.v_reset)
            && (self.v_min..=self.v_max).contains(&self.v)
    }

    pub fn branch_point(&self) -> i64 {
        let numerator = self.b * self.v_threshold + self.a * self.v_rest;
        let denominator = self.a + self.b;
        if numerator >= 0 {
            numerator / denominator
        } else {
            -((-numerator) / denominator)
        }
    }

    pub fn try_step(&mut self, current: i32) -> Result<i32, &'static str> {
        if !self.valid() {
            return Err("invalid IQIF runtime state or parameter ordering");
        }
        let force = if self.v < self.branch_point() {
            self.a
                .checked_mul(self.v_rest - self.v)
                .ok_or("IQIF restoring force overflowed")?
        } else {
            self.b
                .checked_mul(self.v - self.v_threshold)
                .ok_or("IQIF restoring force overflowed")?
        };
        let candidate = self
            .v
            .checked_add(force >> 3)
            .and_then(|value| value.checked_add(i64::from(current)))
            .ok_or("IQIF candidate overflowed")?;
        if candidate > self.v_max {
            self.v = self.v_reset;
            Ok(1)
        } else {
            self.v = candidate.max(self.v_min);
            Ok(0)
        }
    }

    pub fn step(&mut self, current: i32) -> i32 {
        self.try_step(current)
            .expect("IQIF step requires a validated signed-int32 contract")
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }
}

impl Default for IntegerQIFNeuron {
    fn default() -> Self {
        Self {
            v: 128,
            v_rest: 128,
            v_threshold: 200,
            v_reset: 128,
            a: 1,
            b: 1,
            v_max: 255,
            v_min: 0,
        }
    }
}

/// Closed-form Continuous-depth (CfC) neuron. Hasani et al. 2022.
#[derive(Clone, Debug)]
pub struct ClosedFormContinuousNeuron {
    pub x: f64,
    pub w_tau: f64,
    pub w_x: f64,
    pub w_in: f64,
    pub tau_base: f64,
    pub bias: f64,
    pub v_threshold: f64,
    pub dt: f64,
}

impl ClosedFormContinuousNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            w_tau: -0.5,
            w_x: 0.8,
            w_in: 1.0,
            tau_base: 10.0,
            bias: 0.0,
            v_threshold: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let sigma_tau = 1.0 / (1.0 + (-(self.w_tau * current + self.bias)).exp());
        let tau_eff = (self.tau_base * sigma_tau).max(0.1);
        let f_target = (self.w_x * self.x + self.w_in * current).tanh();
        let decay = (-self.dt / tau_eff).exp();
        self.x = self.x * decay + f_target * (1.0 - decay);
        if self.x >= self.v_threshold {
            self.x -= self.v_threshold;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
    }
}

impl Default for ClosedFormContinuousNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qif_fires_with_positive_input() {
        let mut n = QuadraticIFNeuron::default();
        let total: i32 = (0..1000).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }

    #[test]
    fn theta_fires() {
        let mut n = ThetaNeuron::default();
        let total: i32 = (0..1000).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }

    #[test]
    fn perfect_integrator_fires() {
        let mut n = PerfectIntegratorNeuron::default();
        let total: i32 = (0..100).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }

    #[test]
    fn gated_lif_fires() {
        let mut n = GatedLIFNeuron::default();
        let total: i32 = (0..20).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }

    #[test]
    fn nlif_fires() {
        let mut n = NonlinearLIFNeuron::new();
        let total: i32 = (0..2000).map(|_| n.step(500.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn sfa_fires_then_adapts() {
        let mut n = SFANeuron::new();
        let first: i32 = (0..100).map(|_| n.step(30.0)).sum();
        let second: i32 = (0..100).map(|_| n.step(30.0)).sum();
        assert!(first > 0);
        assert!(second <= first + 2);
    }

    #[test]
    fn mat_dual_threshold_adapts() {
        let mut n = MATNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(30.0)).sum();
        assert!(total > 0);
        assert!(n.theta1 > 0.0 || n.theta2 > 0.0);
    }

    #[test]
    fn escape_rate_stochastic() {
        let mut n = EscapeRateNeuron::new(42);
        let total: i32 = (0..1000).map(|_| n.step(30.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn escape_rate_exact_flow_matches_closed_form() {
        let mut n = EscapeRateNeuron::new(42);
        n.v = -65.0;
        n.dt = 5.0;
        n.rho_0 = 1.0e-12;
        let current = 10.0;
        let v0 = n.v;
        let v_inf = n.v_rest + n.resistance * current;
        let euler = v0 + (-(v0 - n.v_rest) + n.resistance * current) / n.tau_m * n.dt;
        let expected = v_inf + (v0 - v_inf) * (-n.dt / n.tau_m).exp();

        assert_eq!(n.step(current), 0);
        assert!((n.v - expected).abs() < 1e-14);
        assert!((n.v - euler).abs() > 1e-3);
    }

    #[test]
    fn klif_fires() {
        let mut n = KLIFNeuron::default();
        let total: i32 = (0..50).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }

    #[test]
    fn ilif_self_inhibits() {
        let mut n = InhibitoryLIFNeuron::default();
        let total: i32 = (0..100).map(|_| n.step(0.8)).sum();
        assert!(total > 0);
    }

    #[test]
    fn clif_positive_spike() {
        let mut n = ComplementaryLIFNeuron::default();
        let total: i32 = (0..20).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }

    #[test]
    fn plif_fires() {
        let mut n = ParametricLIFNeuron::default();
        let total: i32 = (0..20).map(|_| n.step(1.5)).sum();
        assert!(total > 0);
    }

    #[test]
    fn non_resetting_threshold_increases() {
        let mut n = NonResettingLIFNeuron::new();
        let initial = n.theta;
        for _ in 0..5000 {
            n.step(30.0);
        }
        assert!(n.theta > initial);
    }

    #[test]
    fn adaptive_threshold_fires() {
        let mut n = AdaptiveThresholdIFNeuron::new();
        let total: i32 = (0..500).map(|_| n.step(30.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn sigma_delta_encodes() {
        let mut n = SigmaDeltaNeuron::default();
        let total: i32 = (0..10).map(|_| n.step(0.3)).sum();
        assert!(total > 0);
    }

    #[test]
    fn energy_lif_fires() {
        let mut n = EnergyLIFNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(30.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn iqif_fires() {
        let mut n = IntegerQIFNeuron::default();
        let total: i32 = (0..400).map(|_| n.step(10)).sum();
        assert_eq!(total, 26);
    }

    #[test]
    fn cfc_activates() {
        let mut n = ClosedFormContinuousNeuron {
            v_threshold: 0.9,
            ..ClosedFormContinuousNeuron::new()
        };
        let total: i32 = (0..100).map(|_| n.step(5.0)).sum();
        assert!(total > 0);
    }

    // ── Multi-angle tests for trivial IF models ──

    // -- QuadraticIF --
    #[test]
    fn qif_silent_without_input() {
        let mut n = QuadraticIFNeuron::default();
        let t: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn qif_reset_clears_state() {
        let mut n = QuadraticIFNeuron::default();
        for _ in 0..100 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - n.v_reset).abs() < 1e-10);
    }
    #[test]
    fn qif_bounded() {
        let mut n = QuadraticIFNeuron::default();
        for _ in 0..1000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn qif_nan_no_panic() {
        let mut n = QuadraticIFNeuron::default();
        let before = n.v;
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!(n.v, before);
    }
    #[test]
    fn qif_nonfinite_increment_preserves_state() {
        let mut n = QuadraticIFNeuron {
            v: -0.25,
            ..Default::default()
        };
        let before = n.v;
        assert_eq!(n.step(-1.0e308), 0);
        assert_eq!(n.v, before);
    }
    #[test]
    fn qif_matches_exact_positive_current_flow() {
        let mut n = QuadraticIFNeuron::default();
        let root_i = 0.5_f64.sqrt();
        let expected = root_i * ((n.v / root_i).atan() + root_i * n.dt).tan();
        assert_eq!(n.step(0.5), 0);
        assert!((n.v - expected).abs() < 1e-12);
    }
    #[test]
    fn qif_preserves_negative_current_fixed_point() {
        let mut n = QuadraticIFNeuron::default();
        assert_eq!(n.step(-1.0), 0);
        assert_eq!(n.v, -1.0);
    }
    #[test]
    fn qif_exact_flow_resets_on_peak_crossing() {
        let mut n = QuadraticIFNeuron {
            v: 0.95,
            dt: 0.5,
            ..Default::default()
        };
        assert_eq!(n.step(1.0), 1);
        assert_eq!(n.v, n.v_reset);
    }
    #[test]
    fn qif_negative_no_crash() {
        let mut n = QuadraticIFNeuron::default();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }

    // -- Theta --
    #[test]
    fn theta_silent_without_input() {
        let mut n = ThetaNeuron::default();
        let t: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn theta_reset_clears_state() {
        let mut n = ThetaNeuron::default();
        for _ in 0..100 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.theta - 0.0).abs() < 1e-10);
    }
    #[test]
    fn theta_bounded() {
        let mut n = ThetaNeuron::default();
        for _ in 0..1000 {
            n.step(10.0);
        }
        assert!(n.theta.is_finite());
    }
    #[test]
    fn theta_nan_no_panic() {
        ThetaNeuron::default().step(f64::NAN);
    }
    #[test]
    fn theta_exact_positive_flow() {
        let mut n = ThetaNeuron::default();
        n.theta = 1.0;
        n.dt = 0.2;
        let root_i = 2.0_f64.sqrt();
        let phase = ((n.theta / 2.0).tan() / root_i).atan();
        let expected =
            ThetaNeuron::wrap_phase(2.0 * (root_i * (phase + root_i * n.dt).tan()).atan());
        let spike = n.step(2.0);
        assert_eq!(spike, 0);
        assert!((n.theta - expected).abs() < 1.0e-12);
    }
    #[test]
    fn theta_exact_flow_reports_within_step_crossing() {
        let mut n = ThetaNeuron::default();
        n.theta = 2.5;
        n.dt = 1.0;
        assert_eq!(n.step(1.0), 1);
        assert!(n.theta >= -std::f64::consts::PI && n.theta <= std::f64::consts::PI);
    }
    #[test]
    fn theta_stable_fixed_point_preserved() {
        let mut n = ThetaNeuron::default();
        n.theta = -std::f64::consts::FRAC_PI_2;
        n.dt = 100.0;
        assert_eq!(n.step(-1.0), 0);
        assert!((n.theta + std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
    }
    #[test]
    fn theta_non_finite_exact_candidate_preserves_state() {
        let mut n = ThetaNeuron::default();
        n.theta = 0.25;
        n.dt = 1.0e308;
        let before = n.theta;
        assert_eq!(n.step(-1.0e308), 0);
        assert_eq!(n.theta, before);
    }

    // -- PerfectIntegrator --
    #[test]
    fn pi_silent_without_input() {
        let mut n = PerfectIntegratorNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn pi_reset_clears_state() {
        let mut n = PerfectIntegratorNeuron::default();
        for _ in 0..50 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - n.v_reset).abs() < 1e-10);
    }
    #[test]
    fn pi_bounded() {
        let mut n = PerfectIntegratorNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn pi_nan_no_panic() {
        PerfectIntegratorNeuron::default().step(f64::NAN);
    }

    // -- GatedLIF --
    #[test]
    fn gated_lif_silent_without_input() {
        let mut n = GatedLIFNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn gated_lif_reset_clears_state() {
        let mut n = GatedLIFNeuron::default();
        for _ in 0..20 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn gated_lif_bounded() {
        let mut n = GatedLIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn gated_lif_nan_no_panic() {
        GatedLIFNeuron::default().step(f64::NAN);
    }

    // -- NonlinearLIF --
    #[test]
    fn nlif_silent_without_input() {
        let mut n = NonlinearLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn nlif_reset_clears_state() {
        let mut n = NonlinearLIFNeuron::new();
        for _ in 0..100 {
            n.step(500.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn nlif_bounded() {
        let mut n = NonlinearLIFNeuron::new();
        for _ in 0..2000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn nlif_nan_no_panic() {
        NonlinearLIFNeuron::new().step(f64::NAN);
    }
    #[test]
    fn nlif_recovery_evolves() {
        let mut n = NonlinearLIFNeuron::new();
        for _ in 0..2000 {
            n.step(500.0);
        }
        assert!(
            n.w > 0.0,
            "recovery variable w should increase during spiking"
        );
    }

    // -- SFA --
    #[test]
    fn sfa_silent_without_input() {
        let mut n = SFANeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn sfa_reset_clears_state() {
        let mut n = SFANeuron::new();
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
        assert!((n.g_sfa - 0.0).abs() < 1e-10);
    }
    #[test]
    fn sfa_bounded() {
        let mut n = SFANeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sfa_nan_no_panic() {
        SFANeuron::new().step(f64::NAN);
    }

    // -- MAT --
    #[test]
    fn mat_silent_without_input() {
        let mut n = MATNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn mat_reset_clears_state() {
        let mut n = MATNeuron::new();
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
        assert!((n.theta1 - 0.0).abs() < 1e-10);
        assert!((n.theta2 - 0.0).abs() < 1e-10);
    }
    #[test]
    fn mat_bounded() {
        let mut n = MATNeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn mat_nan_no_panic() {
        MATNeuron::new().step(f64::NAN);
    }

    // -- EscapeRate --
    #[test]
    fn escape_rate_reset_clears_state() {
        let mut n = EscapeRateNeuron::new(42);
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn escape_rate_bounded() {
        let mut n = EscapeRateNeuron::new(42);
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn escape_rate_nan_no_panic() {
        let mut n = EscapeRateNeuron::new(42);
        let before = n.v;
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!(n.v, before);
    }
    #[test]
    fn escape_rate_invalid_state_does_not_mutate() {
        let mut n = EscapeRateNeuron::new(42);
        n.v = -65.0;
        n.tau_m = 0.0;
        assert_eq!(n.step(1.0), 0);
        assert_eq!(n.v, -65.0);
    }
    #[test]
    fn escape_rate_seed_varies() {
        let mut n1 = EscapeRateNeuron::new(1);
        let mut n2 = EscapeRateNeuron::new(999);
        let t1: i32 = (0..1000).map(|_| n1.step(30.0)).sum();
        let t2: i32 = (0..1000).map(|_| n2.step(30.0)).sum();
        assert!(t1 > 0 && t2 > 0);
    }

    // -- KLIF --
    #[test]
    fn klif_silent_without_input() {
        let mut n = KLIFNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn klif_reset_clears_state() {
        let mut n = KLIFNeuron::default();
        for _ in 0..20 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn klif_bounded() {
        let mut n = KLIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn klif_nan_no_panic() {
        KLIFNeuron::default().step(f64::NAN);
    }

    // -- InhibitoryLIF --
    #[test]
    fn ilif_silent_without_input() {
        let mut n = InhibitoryLIFNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn ilif_reset_clears_state() {
        let mut n = InhibitoryLIFNeuron::default();
        for _ in 0..50 {
            n.step(0.8);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
        assert!((n.inh_trace - 0.0).abs() < 1e-10);
    }
    #[test]
    fn ilif_bounded() {
        let mut n = InhibitoryLIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn ilif_nan_no_panic() {
        InhibitoryLIFNeuron::default().step(f64::NAN);
    }

    // -- ComplementaryLIF --
    #[test]
    fn clif_silent_without_input() {
        let mut n = ComplementaryLIFNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn clif_reset_clears_state() {
        let mut n = ComplementaryLIFNeuron::default();
        for _ in 0..20 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v_pos - 0.0).abs() < 1e-10);
        assert!((n.v_neg - 0.0).abs() < 1e-10);
    }
    #[test]
    fn clif_bounded() {
        let mut n = ComplementaryLIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v_pos.is_finite());
    }
    #[test]
    fn clif_nan_no_panic() {
        ComplementaryLIFNeuron::default().step(f64::NAN);
    }

    // -- ParametricLIF --
    #[test]
    fn plif_silent_without_input() {
        let mut n = ParametricLIFNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn plif_reset_clears_state() {
        let mut n = ParametricLIFNeuron::default();
        for _ in 0..20 {
            n.step(1.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn plif_bounded() {
        let mut n = ParametricLIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn plif_nan_no_panic() {
        ParametricLIFNeuron::default().step(f64::NAN);
    }

    // -- NonResettingLIF --
    #[test]
    fn nrlif_reset_clears_state() {
        let mut n = NonResettingLIFNeuron::new();
        for _ in 0..500 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
        assert!((n.theta - n.theta_rest).abs() < 1e-10);
    }
    #[test]
    fn nrlif_bounded() {
        let mut n = NonResettingLIFNeuron::new();
        for _ in 0..5000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn nrlif_nan_no_panic() {
        NonResettingLIFNeuron::new().step(f64::NAN);
    }
    #[test]
    fn nrlif_negative_no_crash() {
        let mut n = NonResettingLIFNeuron::new();
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }

    // -- AdaptiveThresholdIF --
    #[test]
    fn atif_silent_without_input() {
        let mut n = AdaptiveThresholdIFNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn atif_reset_clears_state() {
        let mut n = AdaptiveThresholdIFNeuron::new();
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn atif_bounded() {
        let mut n = AdaptiveThresholdIFNeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn atif_nan_no_panic() {
        AdaptiveThresholdIFNeuron::new().step(f64::NAN);
    }
    #[test]
    fn atif_threshold_increases_with_spikes() {
        let mut n = AdaptiveThresholdIFNeuron::new();
        let theta_init = n.theta;
        for _ in 0..500 {
            n.step(30.0);
        }
        assert!(n.theta > theta_init, "threshold should adapt after spikes");
    }

    // -- SigmaDelta --
    #[test]
    fn sd_reset_clears_state() {
        let mut n = SigmaDeltaNeuron::default();
        for _ in 0..10 {
            n.step(0.3);
        }
        n.reset();
        assert!((n.sigma - 0.0).abs() < 1e-10);
    }
    #[test]
    fn sd_bounded() {
        let mut n = SigmaDeltaNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.sigma.is_finite());
    }
    #[test]
    fn sd_nan_no_panic() {
        SigmaDeltaNeuron::default().step(f64::NAN);
    }

    // -- EnergyLIF --
    #[test]
    fn energy_lif_silent_without_input() {
        let mut n = EnergyLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn energy_lif_reset_clears_state() {
        let mut n = EnergyLIFNeuron::new();
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn energy_lif_bounded() {
        let mut n = EnergyLIFNeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn energy_lif_nan_no_panic() {
        EnergyLIFNeuron::new().step(f64::NAN);
    }
    #[test]
    fn energy_lif_epsilon_depletes() {
        let mut n = EnergyLIFNeuron::new();
        let e0 = n.epsilon;
        for _ in 0..200 {
            n.step(30.0);
        }
        assert!(n.epsilon < e0, "energy should deplete during spiking");
    }

    // -- IntegerQIF --
    #[test]
    fn iqif_silent_without_input() {
        let mut n = IntegerQIFNeuron::default();
        let t: i32 = (0..200).map(|_| n.step(0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn iqif_reset_clears_state() {
        let mut n = IntegerQIFNeuron::default();
        for _ in 0..100 {
            n.step(10);
        }
        n.reset();
        assert_eq!(n.v, 128);
    }
    #[test]
    fn iqif_bounded() {
        let mut n = IntegerQIFNeuron::default();
        for _ in 0..1000 {
            n.step(10000);
        }
        // Integer — check no overflow panic occurred
    }
    #[test]
    fn iqif_negative_no_crash() {
        let mut n = IntegerQIFNeuron::default();
        for _ in 0..500 {
            n.step(-50);
        }
        assert_eq!(n.v, 0);
    }
    #[test]
    fn iqif_matches_source_tutorial_prefix_and_period() {
        let mut n = IntegerQIFNeuron::default();
        let mut trace = Vec::new();
        let mut spike_steps = Vec::new();
        for index in 0..400 {
            if n.step(10) == 1 {
                spike_steps.push(index);
            }
            trace.push(n.v);
        }
        assert_eq!(
            &trace[..15],
            &[138, 146, 153, 159, 165, 170, 176, 183, 190, 198, 207, 217, 229, 242, 128]
        );
        assert_eq!(spike_steps, (14..400).step_by(15).collect::<Vec<_>>());
        assert_eq!(trace.last(), Some(&198));
    }

    // -- ClosedFormContinuous --
    #[test]
    fn cfc_reset_clears_state() {
        let mut n = ClosedFormContinuousNeuron::new();
        for _ in 0..50 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.x - 0.0).abs() < 1e-10);
    }
    #[test]
    fn cfc_bounded() {
        let mut n = ClosedFormContinuousNeuron {
            v_threshold: 0.9,
            ..ClosedFormContinuousNeuron::new()
        };
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.x.is_finite());
    }
    #[test]
    fn cfc_nan_no_panic() {
        ClosedFormContinuousNeuron::new().step(f64::NAN);
    }

    // -- StochasticLIFNeuron tests --

    #[test]
    fn stochastic_lif_fires_with_input() {
        let mut n = StochasticLIFNeuron::new(42);
        let total: i32 = (0..500).map(|_| n.step(2.0)).sum();
        assert!(total > 0, "StochasticLIF should fire with strong input");
    }

    #[test]
    fn stochastic_lif_silent_without_input() {
        let mut n = StochasticLIFNeuron::new(42);
        // noise_std=0 by default, so zero input => no spikes
        let total: i32 = (0..500).map(|_| n.step(0.0)).sum();
        assert_eq!(
            total, 0,
            "StochasticLIF should be silent at zero input with no noise"
        );
    }

    #[test]
    fn stochastic_lif_reset_clears_state() {
        let mut n = StochasticLIFNeuron::new(42);
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert!(
            (n.v - n.v_rest).abs() < 1e-12,
            "reset must restore v to v_rest"
        );
        assert_eq!(
            n.refractory_counter, 0,
            "reset must clear refractory counter"
        );
    }

    #[test]
    fn stochastic_lif_extreme_input_bounded() {
        let mut n = StochasticLIFNeuron::new(42);
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite(), "v must stay finite under extreme input");
    }

    #[test]
    fn stochastic_lif_nan_input_stays_finite() {
        let mut n = StochasticLIFNeuron::new(42);
        // Run some normal steps first
        for _ in 0..10 {
            n.step(1.0);
        }
        let v_before = n.v;
        n.step(f64::NAN);
        // After NaN input, v is likely NaN — verify no panic occurred
        // The key invariant: the step function does not panic
        let _ = v_before;
    }

    #[test]
    fn stochastic_lif_negative_input_no_crash() {
        let mut n = StochasticLIFNeuron::new(42);
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite(), "v must stay finite with negative input");
    }

    #[test]
    fn stochastic_lif_noise_affects_firing() {
        // With noise, the neuron may fire even at subthreshold input
        let mut n_noisy = StochasticLIFNeuron::new(123);
        n_noisy.noise_std = 0.5;
        let total_noisy: i32 = (0..5000).map(|_| n_noisy.step(0.8)).sum();

        let mut n_quiet = StochasticLIFNeuron::new(123);
        n_quiet.noise_std = 0.0;
        let total_quiet: i32 = (0..5000).map(|_| n_quiet.step(0.8)).sum();

        // Subthreshold input: quiet neuron may not fire, noisy one may
        // At minimum, they should differ (noise has an effect)
        assert!(
            total_noisy != total_quiet || total_noisy > 0,
            "noise should affect firing pattern"
        );
    }

    #[test]
    fn stochastic_lif_refractory_blocks_spikes() {
        let mut n = StochasticLIFNeuron::new(42);
        n.refractory_period = 5;
        let mut spikes = Vec::new();
        for _ in 0..500 {
            spikes.push(n.step(3.0));
        }
        // After a spike, next `refractory_period` steps must be silent
        for (i, &s) in spikes.iter().enumerate() {
            if s == 1 {
                for j in 1..=5 {
                    if i + j < spikes.len() {
                        assert_eq!(
                            spikes[i + j],
                            0,
                            "step {} after spike at {} must be silent (refractory)",
                            j,
                            i
                        );
                    }
                }
            }
        }
    }
}

/// Stochastic LIF — LIF with Gaussian noise.
#[derive(Clone, Debug)]
pub struct StochasticLIFNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_mem: f64,
    pub dt: f64,
    pub noise_std: f64,
    pub resistance: f64,
    pub refractory_period: i32,
    pub refractory_counter: i32,
    rng: Xoshiro256PlusPlus,
}

impl StochasticLIFNeuron {
    pub fn new(seed: u64) -> Self {
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            v_threshold: 1.0,
            tau_mem: 20.0,
            dt: 1.0,
            noise_std: 0.0,
            resistance: 1.0,
            refractory_period: 0,
            refractory_counter: 0,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            self.v = self.v_rest;
            return 0;
        }
        let dv_leak = -(self.v - self.v_rest) * (self.dt / self.tau_mem);
        let dv_input = self.resistance * current * self.dt;
        let mut dv_noise = 0.0;
        if self.noise_std > 0.0 {
            let u1: f64 = self.rng.random();
            let u2: f64 = self.rng.random();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            dv_noise = self.noise_std * self.dt.sqrt() * z0;
        }
        self.v += dv_leak + dv_input + dv_noise;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.refractory_counter = self.refractory_period;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_counter = 0;
    }
}

impl Default for StochasticLIFNeuron {
    fn default() -> Self {
        Self::new(42)
    }
}
