// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for adaptive_threshold_if

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct AdaptiveThresholdIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub theta_rest: f64,
    pub delta_theta: f64,
    pub tau_m: f64,
    pub tau_theta: f64,
    pub dt: f64,
}

impl AdaptiveThresholdIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            theta: -50.0_f64,
            v_rest: -65.0_f64,
            v_reset: -65.0_f64,
            theta_rest: -50.0_f64,
            delta_theta: 5.0_f64,
            tau_m: 10.0_f64,
            tau_theta: 50.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_adaptive_threshold_if(self) || !i_ext.is_finite() {
            return 0;
        }

        self.v += (-(self.v - self.v_rest) + i_ext) / self.tau_m * self.dt;
        self.theta += -(self.theta - self.theta_rest) / self.tau_theta * self.dt;
        if self.v >= self.theta {
            self.v = self.v_reset;
            self.theta += self.delta_theta;
            return 1;
        }
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_rest;
    }
}

pub fn validate_adaptive_threshold_if(state: &AdaptiveThresholdIFNeuron) -> bool {
    state.v.is_finite()
        && state.theta.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.theta_rest.is_finite()
        && state.delta_theta.is_finite()
        && state.delta_theta >= 0.0
        && state.tau_m.is_finite()
        && state.tau_m > 0.0
        && state.tau_theta.is_finite()
        && state.tau_theta > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.dt <= state.tau_m
        && state.dt <= state.tau_theta
        && state.theta_rest > state.v_rest
        && state.theta_rest > state.v_reset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_threshold_if_new() {
        let state = AdaptiveThresholdIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_adaptive_threshold_if(&state));
    }

    #[test]
    fn test_adaptive_threshold_if_step() {
        let mut state = AdaptiveThresholdIFNeuron::new();
        let spike = state.step(100.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_adaptive_threshold_if_rejects_nonphysical_geometry() {
        let mut state = AdaptiveThresholdIFNeuron::new();
        state.theta_rest = -70.0;
        assert!(!validate_adaptive_threshold_if(&state));
    }
}
