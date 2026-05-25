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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveThresholdIFError {
    InvalidInput,
    InvalidState,
    NonFiniteUpdate,
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

    pub fn step(&mut self, i_ext: f64) -> Result<i32, AdaptiveThresholdIFError> {
        if !i_ext.is_finite() {
            return Err(AdaptiveThresholdIFError::InvalidInput);
        }
        if !validate_adaptive_threshold_if(self) {
            return Err(AdaptiveThresholdIFError::InvalidState);
        }

        let next_v = self.v + (-(self.v - self.v_rest) + i_ext) / self.tau_m * self.dt;
        let next_theta = self.theta + (-(self.theta - self.theta_rest)) / self.tau_theta * self.dt;
        if !next_v.is_finite() || !next_theta.is_finite() {
            return Err(AdaptiveThresholdIFError::NonFiniteUpdate);
        }
        if next_v >= next_theta {
            let spike_theta = next_theta + self.delta_theta;
            if !spike_theta.is_finite() {
                return Err(AdaptiveThresholdIFError::NonFiniteUpdate);
            }
            self.v = self.v_reset;
            self.theta = spike_theta;
            return Ok(1);
        }
        self.v = next_v;
        self.theta = next_theta;
        Ok(0)
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
        let spike = state.step(100.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_adaptive_threshold_if_rejects_nonphysical_geometry() {
        let mut state = AdaptiveThresholdIFNeuron::new();
        state.theta_rest = -70.0;
        assert!(!validate_adaptive_threshold_if(&state));
    }

    #[test]
    fn test_adaptive_threshold_if_rejects_invalid_input_without_mutation() {
        let mut state = AdaptiveThresholdIFNeuron::new();
        let before = (state.v, state.theta);
        assert_eq!(
            state.step(f64::INFINITY),
            Err(AdaptiveThresholdIFError::InvalidInput)
        );
        assert_eq!((state.v, state.theta), before);
    }

    #[test]
    fn test_adaptive_threshold_if_rejects_nonfinite_update_without_mutation() {
        let mut state = AdaptiveThresholdIFNeuron::new();
        state.tau_m = 1.0e-308;
        state.dt = 1.0e-308;
        let before = (state.v, state.theta);
        assert_eq!(
            state.step(1.0e308),
            Err(AdaptiveThresholdIFError::NonFiniteUpdate)
        );
        assert_eq!((state.v, state.theta), before);
    }
}
