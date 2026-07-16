// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Multi-timescale Adaptive Threshold Neuron

/// Multi-timescale Adaptive Threshold. Kobayashi et al. 2009.
#[derive(Clone, Debug)]
pub struct MATNeuron {
    pub v: f64,
    pub theta1: f64,
    pub theta2: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold_base: f64,
    pub tau_m: f64,
    pub tau_1: f64,
    pub tau_2: f64,
    pub h1: f64,
    pub h2: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl MATNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            theta1: 0.0,
            theta2: 0.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold_base: -50.0,
            tau_m: 10.0,
            tau_1: 10.0,
            tau_2: 200.0,
            h1: 5.0,
            h2: 3.0,
            resistance: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) + self.resistance * current) / self.tau_m * self.dt;
        self.theta1 *= (-self.dt / self.tau_1).exp();
        self.theta2 *= (-self.dt / self.tau_2).exp();
        let threshold = self.v_threshold_base + self.theta1 + self.theta2;
        if self.v >= threshold {
            self.v = self.v_reset;
            self.theta1 += self.h1;
            self.theta2 += self.h2;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta1 = 0.0;
        self.theta2 = 0.0;
    }
}

impl Default for MATNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_dual_threshold_adapts() {
        let mut n = MATNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(30.0)).sum();
        assert!(total > 0);
        assert!(n.theta1 > 0.0 || n.theta2 > 0.0);
    }
    #[test]
    fn mat_silent_without_input() {
        let mut n = MATNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn mat_reset_clears_state() {
        let mut n = MATNeuron::new();
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
        assert!((n.theta1 - 0.0).abs() < 1e-10);
        assert!((n.theta2 - 0.0).abs() < 1e-10);
    }
    #[test]
    fn mat_bounded() {
        let mut n = MATNeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn mat_nan_no_panic() {
        MATNeuron::new().step(f64::NAN);
    }
}
