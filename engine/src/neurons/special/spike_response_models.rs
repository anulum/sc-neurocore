// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — spike-response and GLM neuron models

//! Kernel-based SRM0 and point-process generalised linear models.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spike_response_fires() {
        let mut n = SpikeResponseNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(10.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn spike_response_reset_clears_state() {
        let mut n = SpikeResponseNeuron::new();
        for _ in 0..100 {
            n.step(10.0);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }

    #[test]
    fn spike_response_remains_finite_under_large_input() {
        let mut n = SpikeResponseNeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn spike_response_nan_does_not_panic() {
        SpikeResponseNeuron::new().step(f64::NAN);
    }

    #[test]
    fn spike_response_is_silent_without_input() {
        let mut n = SpikeResponseNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }

    #[test]
    fn glm_fires() {
        let mut n = GLMNeuron::new(5, 10, 42);
        let t: i32 = (0..5000).map(|_| n.step(20.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn glm_reset_clears_buffers() {
        let mut n = GLMNeuron::new(5, 10, 42);
        for _ in 0..100 {
            n.step(20.0);
        }
        n.reset();
        assert!(n.stim_buf.iter().all(|value| *value == 0.0));
        assert!(n.spike_buf.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn glm_nan_does_not_panic() {
        GLMNeuron::new(5, 10, 42).step(f64::NAN);
    }
}
