// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Balanced Resonate-and-Fire Neuron Model

//! Balanced Resonate-and-Fire oscillator dynamics.

/// Balanced Resonate-and-Fire — divergence-bound RF with smooth refractory
/// reset. Higuchi, Kairat, Bohte, and Otte 2024, Algorithm 1.
#[derive(Clone, Debug)]
pub struct BalancedResonateAndFireNeuron {
    pub x: f64,
    pub y: f64,
    pub q: f64,
    pub omega: f64,
    pub b_offset: f64,
    pub threshold: f64,
    pub gamma: f64,
    pub dt: f64,
}

pub fn brf_sustain_oscillation_boundary(omega: f64, dt: f64) -> f64 {
    let scaled = dt * omega;
    (-1.0 + (1.0 - scaled * scaled).max(0.0).sqrt()) / dt
}

impl BalancedResonateAndFireNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            q: 0.0,
            omega: 10.0,
            b_offset: 1.0,
            threshold: 1.0,
            gamma: 0.9,
            dt: 0.01,
        }
    }

    pub fn p_omega(&self) -> f64 {
        brf_sustain_oscillation_boundary(self.omega, self.dt)
    }

    pub fn damping(&self) -> f64 {
        self.p_omega() - self.b_offset - self.q
    }

    pub fn dynamic_threshold(&self) -> f64 {
        self.threshold + self.q
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !(self.dt.is_finite()
            && self.omega.is_finite()
            && self.dt > 0.0
            && self.omega > 0.0
            && self.dt * self.omega <= 1.0)
        {
            return 0;
        }
        let b_t = self.damping();
        let theta_t = self.dynamic_threshold();
        let x_prev = self.x;
        let y_prev = self.y;
        self.x = x_prev + self.dt * (b_t * x_prev - self.omega * y_prev + current);
        self.y = y_prev + self.dt * (self.omega * x_prev + b_t * y_prev);
        let spike = if self.x >= theta_t { 1 } else { 0 };
        self.q = self.gamma * self.q + spike as f64;
        spike
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
        self.q = 0.0;
    }
}
impl Default for BalancedResonateAndFireNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = BalancedResonateAndFireNeuron::default();
        let constructed = BalancedResonateAndFireNeuron::new();
        assert_eq!(default.x, constructed.x);
    }

    #[test]
    fn brf_boundary_matches_algorithm() {
        let p = brf_sustain_oscillation_boundary(10.0, 0.01);
        let expected = (-1.0 + (1.0_f64 - 0.1_f64 * 0.1_f64).sqrt()) / 0.01;
        assert!((p - expected).abs() < 1e-12);
    }

    #[test]
    fn brf_step_matches_algorithm_one_step() {
        let mut n = BalancedResonateAndFireNeuron {
            x: 0.2,
            y: -0.1,
            q: 0.3,
            omega: 12.0,
            b_offset: 0.75,
            threshold: 1.0,
            gamma: 0.9,
            dt: 0.01,
        };
        let p_omega = brf_sustain_oscillation_boundary(12.0, 0.01);
        let b_t = p_omega - 0.75 - 0.3;
        let expected_x = 0.2 + 0.01 * (b_t * 0.2 - 12.0 * -0.1 + 2.0);
        let expected_y = -0.1 + 0.01 * (12.0 * 0.2 + b_t * -0.1);
        let expected_spike = if expected_x >= 1.3 { 1 } else { 0 };
        let spike = n.step(2.0);
        assert_eq!(spike, expected_spike);
        assert!((n.x - expected_x).abs() < 1e-12);
        assert!((n.y - expected_y).abs() < 1e-12);
        assert!((n.q - (0.9 * 0.3 + expected_spike as f64)).abs() < 1e-12);
    }

    #[test]
    fn brf_reset_clears_membrane_and_refractory_state() {
        let mut n = BalancedResonateAndFireNeuron::new();
        assert_eq!(n.step(200.0), 1);
        n.reset();
        assert_eq!(n.x, 0.0);
        assert_eq!(n.y, 0.0);
        assert_eq!(n.q, 0.0);
    }
}
