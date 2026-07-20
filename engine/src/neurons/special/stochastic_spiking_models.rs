// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — stochastic spiking neuron models

//! Membrane-state stochastic IF and Galves-Löcherbach point-process models.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stochastic_if_fires() {
        let mut n = StochasticIFNeuron::new(42);
        let t: i32 = (0..500).map(|_| n.step(30.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn stochastic_if_reset_restores_resting_voltage() {
        let mut n = StochasticIFNeuron::new(42);
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }

    #[test]
    fn stochastic_if_remains_finite_under_large_current() {
        let mut n = StochasticIFNeuron::new(42);
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn stochastic_if_nan_does_not_panic() {
        StochasticIFNeuron::new(42).step(f64::NAN);
    }

    #[test]
    fn stochastic_if_remains_finite_under_negative_current() {
        let mut n = StochasticIFNeuron::new(42);
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn galves_locherbach_fires() {
        let mut n = GalvesLocherbachNeuron::new(42);
        let t: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn galves_locherbach_reset_restores_resting_voltage() {
        let mut n = GalvesLocherbachNeuron::new(42);
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }

    #[test]
    fn galves_locherbach_remains_finite_under_large_input() {
        let mut n = GalvesLocherbachNeuron::new(42);
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn galves_locherbach_nan_does_not_panic() {
        GalvesLocherbachNeuron::new(42).step(f64::NAN);
    }
}
