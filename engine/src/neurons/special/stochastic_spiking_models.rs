// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — stochastic spiking neuron models

//! Membrane-state stochastic IF and Galves-Löcherbach point-process models.

#![deny(missing_docs)]

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Stochastic IF — Ornstein-Uhlenbeck noise on LIF membrane.
#[derive(Clone, Debug)]
pub struct StochasticIFNeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Resting / leak reversal potential (mV).
    pub v_rest: f64,
    /// Membrane potential reset to after a spike (mV).
    pub v_reset: f64,
    /// Voltage at which a spike is registered (mV).
    pub v_threshold: f64,
    /// Membrane time constant (ms).
    pub tau_m: f64,
    /// Noise amplitude; the per-step membrane noise has standard deviation sigma \* sqrt(dt / tau_m), and the free membrane's stationary one is close to sigma / sqrt(2) (mV).
    pub sigma: f64,
    /// Step (ms).
    pub dt: f64,
    rng: Xoshiro256PlusPlus,
}

impl StochasticIFNeuron {
    /// Create a neuron at rest (-70 mV) with the default parameters, `dt` = 1 ms and a
    /// Xoshiro256++ generator seeded with `seed`.
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
    /// Advance one Euler-Maruyama step and return 1 on a spike, else 0.
    ///
    /// `v` gains `(-(v - v_rest) + current) / tau_m * dt + sigma * sqrt(dt / tau_m) * z`,
    /// where `z` is a Box-Muller standard normal sample (the first uniform is floored at
    /// 1e-30). Reaching `v_threshold` sets `v` to `v_reset`. The input is not validated.
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
    /// Return `v` to `v_rest`; the parameters and the generator state are kept.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }
}

/// Galves-Löcherbach 2013 — stochastic point process with memory.
#[derive(Clone, Debug)]
pub struct GalvesLocherbachNeuron {
    /// Normalised membrane-potential variable (dimensionless).
    pub v: f64,
    /// Resting value of the normalised membrane-potential variable (dimensionless).
    pub v_rest: f64,
    /// Leak factor applied to the membrane potential each step (dimensionless).
    pub decay: f64,
    /// Centre of the sigmoidal per-step firing probability, where it is one half (dimensionless).
    pub threshold_rate: f64,
    /// Steepness of the sigmoidal firing intensity (dimensionless).
    pub steepness: f64,
    /// Step factor: the event probability is the sigmoid times `dt`.
    pub dt: f64,
    rng: Xoshiro256PlusPlus,
}

impl GalvesLocherbachNeuron {
    /// Create a neuron at rest (`v` = 0) with the default parameters, `dt` = 1 and a
    /// Xoshiro256++ generator seeded with `seed`.
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
    /// Advance one step with `weighted_input` and return 1 on an event, else 0.
    ///
    /// `v` becomes `decay * v + weighted_input`. An event occurs with probability
    /// `dt / (1 + exp(-steepness * (v - threshold_rate)))` and sets `v` to `v_rest`. Inputs
    /// and parameters are not validated; the Python model rejects a non-finite `v`, `v_rest`,
    /// `threshold_rate` or input, `decay` outside `[0, 1]`, a non-positive `steepness` and `dt`
    /// outside `(0, 1]`.
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
    /// Return `v` to `v_rest`; the parameters and the generator state are kept.
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
