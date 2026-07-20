// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — stochastic point-process neuron models

//! Homogeneous Poisson, inhomogeneous Poisson, and gamma-renewal processes.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Homogeneous Poisson binary-bin generator with a replayable LFSR16 stream.
#[derive(Clone, Debug)]
pub struct PoissonNeuron {
    pub rate_hz: f64,
    pub dt_ms: f64,
    pub rng_state: u16,
    pub initial_seed: u16,
}

impl PoissonNeuron {
    pub fn new(rate_hz: f64, dt_ms: f64, seed: u64) -> Self {
        let narrowed = (seed & u64::from(u16::MAX)) as u16;
        let initial_seed = if narrowed == 0 { 0xACE1 } else { narrowed };
        Self {
            rate_hz,
            dt_ms,
            rng_state: initial_seed,
            initial_seed,
        }
    }

    pub fn valid(&self) -> bool {
        self.rate_hz.is_finite()
            && self.rate_hz >= 0.0
            && self.dt_ms.is_finite()
            && self.dt_ms > 0.0
            && self.rng_state != 0
            && self.initial_seed != 0
    }

    pub fn try_step(&mut self, rate_override: f64) -> Result<i32, &'static str> {
        if !self.valid() || !rate_override.is_finite() {
            return Err("invalid Poisson state or rate override");
        }
        let r = if rate_override < 0.0 {
            self.rate_hz
        } else {
            rate_override
        };
        if !r.is_finite() || r < 0.0 {
            return Err("invalid active Poisson rate");
        }
        let hazard = r * self.dt_ms / 1000.0;
        if !hazard.is_finite() || hazard < 0.0 {
            return Err("non-finite Poisson interval hazard");
        }
        let probability = -(-hazard).exp_m1();
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err("invalid Poisson spike probability");
        }
        let mut sample = self.rng_state;
        for _ in 0..8 {
            let feedback = (sample ^ (sample >> 2) ^ (sample >> 3) ^ (sample >> 5)) & 1;
            sample = (sample >> 1) | (feedback << 15);
        }
        let threshold = if probability <= 0.0 {
            0_u32
        } else if probability >= 1.0 {
            65_536_u32
        } else {
            (probability * 65_535.0).floor() as u32 + 1
        };
        self.rng_state = sample;
        Ok(i32::from(u32::from(sample) < threshold))
    }

    /// Preserve the engine runner's infallible dispatch boundary.
    pub fn step(&mut self, rate_override: f64) -> i32 {
        self.try_step(rate_override).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.rng_state = self.initial_seed;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_fires() {
        let mut n = PoissonNeuron::new(200.0, 1.0, 42);
        let t: i32 = (0..1000)
            .map(|_| n.try_step(-1.0).expect("valid Poisson step"))
            .sum();
        assert!(t > 10);
    }

    #[test]
    fn poisson_reset_replays_stream() {
        let mut n = PoissonNeuron::new(200.0, 1.0, 42);
        let first: Vec<i32> = (0..100)
            .map(|_| n.try_step(-1.0).expect("valid Poisson step"))
            .collect();
        n.reset();
        let replay: Vec<i32> = (0..100)
            .map(|_| n.try_step(-1.0).expect("valid Poisson step"))
            .collect();
        assert_eq!(first, replay);
    }

    #[test]
    fn poisson_nan_fails_closed() {
        let mut n = PoissonNeuron::new(200.0, 1.0, 42);
        let before = n.rng_state;
        assert!(n.try_step(f64::NAN).is_err());
        assert_eq!(n.rng_state, before);
    }

    #[test]
    fn poisson_seed_varies() {
        let mut n1 = PoissonNeuron::new(200.0, 1.0, 1);
        let mut n2 = PoissonNeuron::new(200.0, 1.0, 999);
        let t1: Vec<i32> = (0..1000)
            .map(|_| n1.try_step(-1.0).expect("valid Poisson step"))
            .collect();
        let t2: Vec<i32> = (0..1000)
            .map(|_| n2.try_step(-1.0).expect("valid Poisson step"))
            .collect();
        assert_ne!(t1, t2);
    }

    #[test]
    fn poisson_full_period_matches_quantised_exact_hazard() {
        let mut n = PoissonNeuron::new(250.0, 1.0, 0xACE1);
        let spikes: i32 = (0..65_535)
            .map(|_| n.try_step(-1.0).expect("valid Poisson step"))
            .sum();
        assert_eq!(spikes, 14_496);
        assert_eq!(n.rng_state, 0xACE1);
    }

    #[test]
    fn inhomogeneous_poisson_fires() {
        let mut n = InhomogeneousPoissonNeuron::new(1.0, 42);
        let t: i32 = (0..1000).map(|_| n.step(200.0)).sum();
        assert!(t > 10);
    }

    #[test]
    fn inhomogeneous_poisson_zero_rate() {
        let mut n = InhomogeneousPoissonNeuron::new(1.0, 42);
        let t: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }

    #[test]
    fn inhomogeneous_poisson_nan_does_not_panic() {
        let mut n = InhomogeneousPoissonNeuron::new(1.0, 42);
        n.step(f64::NAN);
    }

    #[test]
    fn gamma_renewal_fires() {
        let mut n = GammaRenewalNeuron::new(100.0, 3, 42);
        let t: i32 = (0..2000).map(|_| n.step(-1.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn gamma_renewal_reset_clears_elapsed_time() {
        let mut n = GammaRenewalNeuron::new(100.0, 3, 42);
        for _ in 0..100 {
            n.step(-1.0);
        }
        n.reset();
        assert!((n.time_since_spike - 0.0).abs() < 1e-10);
    }

    #[test]
    fn gamma_renewal_nan_does_not_panic() {
        let mut n = GammaRenewalNeuron::new(100.0, 3, 42);
        n.step(f64::NAN);
    }
}
