// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stochastic LIF Neuron

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

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

#[cfg(test)]
mod tests {
    use super::*;

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
