// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GIF Population Neuron Model

//! Seeded escape-rate generalized integrate-and-fire population dynamics.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// GIF population — escape-rate generalized IF. Mensi et al. 2012.
#[derive(Clone, Debug)]
pub struct GIFPopulationNeuron {
    pub v: f64,
    pub theta: f64,
    pub eta: f64,
    pub tau_m: f64,
    pub tau_eta: f64,
    pub delta_v: f64,
    pub lambda_0: f64,
    pub eta_increment: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub dt: f64,
    pub seed: u64,
    rng: Xoshiro256PlusPlus,
}

impl GIFPopulationNeuron {
    pub fn new(seed: u64) -> Self {
        Self {
            v: -65.0,
            theta: -50.0,
            eta: 0.0,
            tau_m: 20.0,
            tau_eta: 100.0,
            delta_v: 2.0,
            lambda_0: 0.001,
            eta_increment: 5.0,
            v_rest: -65.0,
            v_reset: -65.0,
            dt: 0.5,
            seed,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    fn finite_values(values: &[f64]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn valid_runtime(&self) -> bool {
        Self::finite_values(&[
            self.v,
            self.theta,
            self.eta,
            self.tau_m,
            self.tau_eta,
            self.delta_v,
            self.lambda_0,
            self.eta_increment,
            self.v_rest,
            self.v_reset,
            self.dt,
        ]) && self.tau_m > 0.0
            && self.tau_eta > 0.0
            && self.delta_v > 0.0
            && self.lambda_0 >= 0.0
            && self.dt > 0.0
    }

    fn advance_subthreshold(&self, current: f64) -> Option<(f64, f64)> {
        let eta_decay = (-self.dt / self.tau_eta).exp();
        let membrane_decay = (-self.dt / self.tau_m).exp();
        let x0 = self.v - self.v_rest - current;
        let eta_new = self.eta * eta_decay;
        let x_new = if (self.tau_m - self.tau_eta).abs() <= 1e-12 {
            membrane_decay * (x0 - self.eta * self.dt / self.tau_m)
        } else {
            let coupling = self.tau_eta / (self.tau_eta - self.tau_m);
            x0 * membrane_decay - self.eta * coupling * (eta_decay - membrane_decay)
        };
        let v_new = self.v_rest + current + x_new;
        if Self::finite_values(&[v_new, eta_new]) {
            Some((v_new, eta_new))
        } else {
            None
        }
    }

    fn spike_probability(&self, voltage: f64) -> f64 {
        if self.lambda_0 == 0.0 {
            return 0.0;
        }
        let exponent = ((voltage - self.theta) / self.delta_v).clamp(-745.0, 20.0);
        let hazard = self.lambda_0 * exponent.exp();
        (1.0 - (-hazard * self.dt).exp()).clamp(0.0, 1.0)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid_runtime() {
            return 0;
        }
        let Some((v_candidate, eta_candidate)) = self.advance_subthreshold(current) else {
            return 0;
        };
        self.v = v_candidate;
        self.eta = eta_candidate;
        if self.rng.random::<f64>() < self.spike_probability(self.v) {
            self.v = self.v_reset;
            self.eta += self.eta_increment;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.eta = 0.0;
        self.rng = Xoshiro256PlusPlus::seed_from_u64(self.seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_time_constants_use_limit_solution() {
        let mut n = GIFPopulationNeuron::new(42);
        n.tau_eta = n.tau_m;
        assert_eq!(n.step(0.0), 0);
        assert!(n.v.is_finite());
        assert!(n.eta.is_finite());
    }

    #[test]
    fn zero_escape_rate_remains_subthreshold() {
        let mut n = GIFPopulationNeuron::new(42);
        n.lambda_0 = 0.0;
        assert_eq!(n.step(30.0), 0);
        assert_ne!(n.v, n.v_reset);
    }

    #[test]
    fn nonfinite_subthreshold_candidate_preserves_state() {
        let mut n = GIFPopulationNeuron::new(42);
        n.v = f64::MAX;
        n.v_rest = -f64::MAX;
        let before = (n.v, n.eta);
        assert_eq!(n.step(f64::MAX), 0);
        assert_eq!((n.v, n.eta), before);
    }

    #[test]
    fn gif_pop_fires() {
        let mut n = GIFPopulationNeuron::new(42);
        let t: i32 = (0..1000).map(|_| n.step(30.0)).sum();
        assert!(t > 0);
    }

    // -- GIFPopulation --
    #[test]
    fn gif_pop_exact_subthreshold_reference_point() {
        let mut n = GIFPopulationNeuron::new(7);
        n.v = -68.0;
        n.eta = 0.4;
        assert_eq!(n.step(4.0), 0);
        assert!((n.v - (-67.8370206677805)).abs() < 1e-12);
        assert!((n.eta - 0.398004991677073).abs() < 1e-15);
    }
    #[test]
    fn gif_pop_forced_spike_adds_decayed_adaptation() {
        let mut n = GIFPopulationNeuron::new(42);
        n.v = -51.0;
        n.eta = 0.3;
        n.theta = -90.0;
        n.lambda_0 = 1.0e9;
        assert_eq!(n.step(0.0), 1);
        assert!((n.v - n.v_reset).abs() < 1e-12);
        assert!((n.eta - 5.298503743757805).abs() < 1e-15);
    }
    #[test]
    fn gif_pop_invalid_input_preserves_state() {
        let mut n = GIFPopulationNeuron::new(42);
        n.v = -62.0;
        n.eta = 0.75;
        let before = (n.v, n.eta);
        assert_eq!(n.step(f64::NAN), 0);
        n.tau_m = 0.0;
        assert_eq!(n.step(1.0), 0);
        assert_eq!((n.v, n.eta), before);
    }
    #[test]
    fn gif_pop_seeded_reset_replays() {
        let mut n = GIFPopulationNeuron::new(123);
        n.theta = -90.0;
        n.lambda_0 = 1.0e9;
        let first: Vec<i32> = (0..3).map(|_| n.step(0.0)).collect();
        n.reset();
        let replay: Vec<i32> = (0..3).map(|_| n.step(0.0)).collect();
        assert_eq!(first, replay);
        assert!(n.eta > n.eta_increment);
    }
    #[test]
    fn gif_pop_negative_drive_remains_finite() {
        let mut n = GIFPopulationNeuron::new(42);
        for _ in 0..200 {
            n.step(-30.0);
        }
        assert!(n.v.is_finite());
        assert!(n.eta.is_finite());
    }
}
