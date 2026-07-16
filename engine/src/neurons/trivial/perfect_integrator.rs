// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Perfect Integrator Neuron

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_integrator_fires() {
        let mut n = PerfectIntegratorNeuron::default();
        let total: i32 = (0..100).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }
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
}
