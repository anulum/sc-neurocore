// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — project stochastic rate-adaptation recurrence

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Count-neutral SC logistic rate-adaptation model with hazard sampling.
#[derive(Clone, Debug)]
pub struct SCStochasticRateAdaptationNeuron {
    pub a: f64,
    pub f_max: f64,
    pub beta: f64,
    pub i_half: f64,
    pub tau_a: f64,
    pub delta_a: f64,
    pub dt: f64,
    rng: Xoshiro256PlusPlus,
}

impl SCStochasticRateAdaptationNeuron {
    #[must_use]
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

    fn rate(&self, current: f64, a: f64) -> f64 {
        let z = self.beta * (current - a - self.i_half);
        if z >= 0.0 {
            self.f_max / (1.0 + (-z).exp())
        } else {
            let exp_z = z.exp();
            self.f_max * exp_z / (1.0 + exp_z)
        }
    }

    fn rhs(&self, current: f64, a: f64) -> (f64, f64) {
        let rate = self.rate(current, a);
        (-a / self.tau_a + self.delta_a * rate, rate)
    }

    /// Advance with an explicit uniform variate; returns `-1` on invalid input.
    pub fn step_with_uniform(&mut self, current: f64, uniform: f64) -> i32 {
        if !self.valid()
            || !current.is_finite()
            || !uniform.is_finite()
            || !(0.0..1.0).contains(&uniform)
        {
            return -1;
        }
        let (k1, r1) = self.rhs(current, self.a);
        let (k2, r2) = self.rhs(current, self.a + 0.5 * self.dt * k1);
        let (k3, r3) = self.rhs(current, self.a + 0.5 * self.dt * k2);
        let (k4, r4) = self.rhs(current, self.a + self.dt * k3);
        let next_a = self.a + self.dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        let rate = (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0;
        let probability = 1.0 - (-rate * self.dt / 1000.0).exp();
        if !next_a.is_finite()
            || next_a < 0.0
            || !probability.is_finite()
            || !(0.0..=1.0).contains(&probability)
        {
            return -1;
        }
        self.a = next_a;
        i32::from(uniform < probability)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let uniform = self.rng.random::<f64>();
        self.step_with_uniform(current, uniform)
    }

    pub fn reset(&mut self) {
        self.a = 0.0;
    }

    #[must_use]
    pub fn valid(&self) -> bool {
        self.a.is_finite()
            && self.a >= 0.0
            && self.f_max.is_finite()
            && self.f_max > 0.0
            && self.beta.is_finite()
            && self.beta > 0.0
            && self.i_half.is_finite()
            && self.tau_a.is_finite()
            && self.tau_a > 0.0
            && self.delta_a.is_finite()
            && self.delta_a >= 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_uniform_is_reproducible() {
        let mut left = SCStochasticRateAdaptationNeuron::new(1);
        let mut right = SCStochasticRateAdaptationNeuron::new(2);
        assert_eq!(
            left.step_with_uniform(20.0, 0.01),
            right.step_with_uniform(20.0, 0.01)
        );
        assert_eq!(left.a, right.a);
    }
}
