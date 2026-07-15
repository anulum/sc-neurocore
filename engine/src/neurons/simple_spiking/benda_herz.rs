// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Benda-Herz Neuron Model

//! Benda-Herz firing-rate adaptation dynamics.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Benda-Herz 2003 — firing-rate adaptation via subtractive feedback.
#[derive(Clone, Debug)]
pub struct BendaHerzNeuron {
    pub a: f64,
    pub f_max: f64,
    pub beta: f64,
    pub i_half: f64,
    pub tau_a: f64,
    pub delta_a: f64,
    pub dt: f64,
    rng: Xoshiro256PlusPlus,
}

impl BendaHerzNeuron {
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
    pub fn step(&mut self, current: f64) -> i32 {
        let x = current - self.a;
        let rate = self.f_max / (1.0 + (-self.beta * (x - self.i_half)).exp());
        self.a += (-self.a / self.tau_a + self.delta_a * rate) * self.dt;
        let p = rate * self.dt / 1000.0;
        if self.rng.random::<f64>() < p {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.a = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_is_seed_reproducible() {
        let mut left = BendaHerzNeuron::new(42);
        let mut right = BendaHerzNeuron::new(42);
        let left_trace: Vec<i32> = (0..128).map(|_| left.step(20.0)).collect();
        let right_trace: Vec<i32> = (0..128).map(|_| right.step(20.0)).collect();
        assert_eq!(left_trace, right_trace);
    }

    #[test]
    fn benda_herz_fires() {
        let mut n = BendaHerzNeuron::new(42);
        let t: i32 = (0..10000).map(|_| n.step(20.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn benda_herz_reset_clears_state() {
        let mut n = BendaHerzNeuron::new(42);
        for _ in 0..100 {
            n.step(20.0);
        }
        n.reset();
        assert!((n.a - 0.0).abs() < 1e-10);
    }

    #[test]
    fn benda_herz_bounded() {
        let mut n = BendaHerzNeuron::new(42);
        for _ in 0..10000 {
            n.step(1e4);
        }
        assert!(n.a.is_finite());
    }

    #[test]
    fn benda_herz_adaptation() {
        let mut n = BendaHerzNeuron::new(42);
        for _ in 0..10000 {
            n.step(20.0);
        }
        assert!(n.a > 0.0, "adaptation variable a should increase: {}", n.a);
    }

    #[test]
    fn benda_herz_nan_no_panic() {
        BendaHerzNeuron::new(42).step(f64::NAN);
    }

    #[test]
    fn benda_herz_negative_no_crash() {
        let mut n = BendaHerzNeuron::new(42);
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.a.is_finite());
    }
}
