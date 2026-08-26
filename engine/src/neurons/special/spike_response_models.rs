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
        // Reference filters (Pillow 2008 discrete specialisation), matching
        // the Python reference implementation exactly.
        let k = (0..n_k).map(|i| (-(i as f64) / 3.0).exp() * 0.5).collect();
        let h = (0..n_h)
            .map(|t| -5.0 * (-(t as f64) / 2.0).exp() + 0.5 * (-(t as f64) / 10.0).exp())
            .collect();
        Self {
            mu: -3.0,
            dt_ms: 1.0,
            k,
            h,
            stim_buf: vec![0.0; n_k],
            spike_buf: vec![0.0; n_h],
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    fn valid(&self) -> bool {
        self.mu.is_finite()
            && self.dt_ms.is_finite()
            && self.dt_ms > 0.0
            && self.dt_ms <= 1000.0
            && self.k.len() == self.stim_buf.len()
            && self.h.len() == self.spike_buf.len()
            && self.k.iter().all(|value| value.is_finite())
            && self.h.iter().all(|value| value.is_finite())
            && self.stim_buf.iter().all(|value| value.is_finite())
            && self.spike_buf.iter().all(|value| value.is_finite())
    }

    pub fn try_step(&mut self, stimulus: f64, uniform: Option<f64>) -> Result<i32, &'static str> {
        if !stimulus.is_finite() {
            return Err("stimulus must be finite");
        }
        if let Some(sample) = uniform {
            if !sample.is_finite() || !(0.0..1.0).contains(&sample) {
                return Err("uniform must be finite and within [0, 1)");
            }
        }
        if !self.valid() {
            return Err("GLM state and parameters must satisfy the public bounds");
        }

        let nk = self.stim_buf.len();
        let nh = self.spike_buf.len();
        let mut stim_candidate = self.stim_buf.clone();
        for i in (1..nk).rev() {
            stim_candidate[i] = stim_candidate[i - 1];
        }
        if nk > 0 {
            stim_candidate[0] = stimulus;
        }
        let dot_k: f64 = self
            .k
            .iter()
            .zip(stim_candidate.iter())
            .map(|(a, b)| a * b)
            .sum();
        let dot_h: f64 = self
            .h
            .iter()
            .zip(self.spike_buf.iter())
            .map(|(a, b)| a * b)
            .sum();
        let log_rate = (dot_k + dot_h + self.mu).clamp(-20.0, 20.0);
        let lam = log_rate.exp();
        let p = lam * self.dt_ms / 1000.0;
        let draw = match uniform {
            Some(sample) => sample,
            None => self.rng.random::<f64>(),
        };
        let spike = if draw < p.min(1.0) { 1 } else { 0 };
        let mut spike_candidate = self.spike_buf.clone();
        for i in (1..nh).rev() {
            spike_candidate[i] = spike_candidate[i - 1];
        }
        if nh > 0 {
            spike_candidate[0] = spike as f64;
        }
        self.stim_buf = stim_candidate;
        self.spike_buf = spike_candidate;
        Ok(spike)
    }

    pub fn step(&mut self, stimulus: f64) -> i32 {
        self.try_step(stimulus, None).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.stim_buf.fill(0.0);
        self.spike_buf.fill(0.0);
    }

    pub fn stim_buf_view(&self) -> Vec<f64> {
        self.stim_buf.clone()
    }

    pub fn spike_buf_view(&self) -> Vec<f64> {
        self.spike_buf.clone()
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
    fn glm_nan_input_is_rejected_atomically() {
        let mut n = GLMNeuron::new(5, 10, 42);
        let stim_before = n.stim_buf.clone();
        let spike_before = n.spike_buf.clone();
        assert!(n.try_step(f64::NAN, None).is_err());
        assert!(n.try_step(f64::INFINITY, None).is_err());
        assert_eq!(n.stim_buf, stim_before);
        assert_eq!(n.spike_buf, spike_before);
    }

    #[test]
    fn glm_out_of_range_uniform_is_rejected() {
        let mut n = GLMNeuron::new(5, 10, 42);
        assert!(n.try_step(1.0, Some(1.0)).is_err());
        assert!(n.try_step(1.0, Some(f64::NAN)).is_err());
        assert!(n.try_step(1.0, Some(0.0)).is_ok());
    }

    #[test]
    fn glm_reference_filters_match_python_defaults() {
        let n = GLMNeuron::new(10, 20, 42);
        assert!((n.k[0] - 0.5).abs() < 1e-15);
        assert!((n.k[3] - 0.5 * (-1.0_f64).exp()).abs() < 1e-15);
        assert!((n.h[0] - -4.5).abs() < 1e-15);
        assert!(n.h[19] > n.h[0]);
    }

    #[test]
    fn glm_explicit_uniform_is_deterministic_across_seeds() {
        let mut a = GLMNeuron::new(10, 20, 1);
        let mut b = GLMNeuron::new(10, 20, 2);
        for index in 0..200 {
            let sample = (index as f64 % 97.0) / 97.0;
            let spike_a = a.try_step(5.0, Some(sample)).expect("finite drive");
            let spike_b = b.try_step(5.0, Some(sample)).expect("finite drive");
            assert_eq!(spike_a, spike_b);
        }
        assert_eq!(a.stim_buf, b.stim_buf);
        assert_eq!(a.spike_buf, b.spike_buf);
    }
}
