// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for expif

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct ExpIFNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub v_rh: f64,
    pub delta_t: f64,
    pub tau: f64,
    pub dt: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpIFError {
    InvalidInput,
    InvalidState,
    NonFiniteUpdate,
}

impl ExpIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            v_rest: -65.0_f64,
            v_reset: -68.0_f64,
            v_threshold: -50.0_f64,
            v_rh: -55.0_f64,
            delta_t: 2.0_f64,
            tau: 20.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, ExpIFError> {
        if !i_ext.is_finite() {
            return Err(ExpIFError::InvalidInput);
        }
        if !validate_expif(self) {
            return Err(ExpIFError::InvalidState);
        }

        let k1 = self.rhs(self.v, i_ext);
        let k2 = self.rhs(self.v + 0.5 * self.dt * k1, i_ext);
        let k3 = self.rhs(self.v + 0.5 * self.dt * k2, i_ext);
        let k4 = self.rhs(self.v + self.dt * k3, i_ext);
        let next_v = self.v + self.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        if !k1.is_finite()
            || !k2.is_finite()
            || !k3.is_finite()
            || !k4.is_finite()
            || !next_v.is_finite()
        {
            return Err(ExpIFError::NonFiniteUpdate);
        }

        self.v = next_v;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            return Ok(1);
        }
        Ok(0)
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
    }

    fn rhs(&self, v: f64, i_ext: f64) -> f64 {
        let arg = ((v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * arg.exp();
        (-(v - self.v_rest) + exp_term + i_ext) / self.tau
    }
}

pub fn validate_expif(state: &ExpIFNeuron) -> bool {
    state.v.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold.is_finite()
        && state.v_rh.is_finite()
        && state.delta_t.is_finite()
        && state.tau.is_finite()
        && state.dt.is_finite()
        && state.delta_t > 0.0
        && state.tau > 0.0
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expif_new() {
        let state = ExpIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_expif(&state));
    }

    #[test]
    fn test_expif_step() {
        let mut state = ExpIFNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_expif_rejects_invalid_input_without_mutation() {
        let mut state = ExpIFNeuron::new();
        let before = state.v;
        assert_eq!(state.step(f64::INFINITY), Err(ExpIFError::InvalidInput));
        assert_eq!(state.v, before);
    }

    #[test]
    fn test_expif_rejects_nonfinite_update_without_mutation() {
        let mut state = ExpIFNeuron::new();
        state.dt = 1.0e308;
        let before = state.v;
        assert_eq!(state.step(1.0e308), Err(ExpIFError::NonFiniteUpdate));
        assert_eq!(state.v, before);
    }

    #[test]
    fn test_expif_matches_rk4_reference() {
        let mut state = ExpIFNeuron::new();
        state.v = -60.0;
        let current = 12.0;
        let k1 = state.rhs(state.v, current);
        let k2 = state.rhs(state.v + 0.5 * state.dt * k1, current);
        let k3 = state.rhs(state.v + 0.5 * state.dt * k2, current);
        let k4 = state.rhs(state.v + state.dt * k3, current);
        let expected = state.v + state.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

        assert_eq!(state.step(current), Ok(0));
        assert!((state.v - expected).abs() < 1.0e-12);
    }
}
