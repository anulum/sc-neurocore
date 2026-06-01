// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for plif

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct ParametricLIFNeuron {
    pub v: f64,
    pub a: f64,
    pub threshold: f64,
    pub dt: f64,
}

impl ParametricLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0_f64,
            a: 0.0_f64,
            threshold: 1.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn alpha(&self) -> f64 {
        if self.a >= 0.0 {
            let z = (-self.a).exp();
            1.0 / (1.0 + z)
        } else {
            let z = self.a.exp();
            z / (1.0 + z)
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_plif(self) || !i_ext.is_finite() {
            return 0;
        }

        let spike = if self.v >= self.threshold { 1.0 } else { 0.0 };
        let next_v = self.alpha() * self.v * (1.0 - spike) + i_ext;
        if !next_v.is_finite() {
            return 0;
        }
        self.v = next_v;
        if next_v >= self.threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0_f64;
    }
}

pub fn validate_plif(state: &ParametricLIFNeuron) -> bool {
    state.v.is_finite()
        && state.a.is_finite()
        && state.threshold.is_finite()
        && state.threshold > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plif_new() {
        let state = ParametricLIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_plif(&state));
    }

    #[test]
    fn test_plif_step() {
        let mut state = ParametricLIFNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_plif_alpha_is_stable_for_large_negative_parameter() {
        let mut state = ParametricLIFNeuron::new();
        state.a = -1000.0;
        assert_eq!(state.alpha(), 0.0);
    }

    #[test]
    fn test_plif_candidate_overflow_preserves_state() {
        let mut state = ParametricLIFNeuron::new();
        state.v = 1.0e308;
        state.a = 1000.0;
        state.threshold = 1.7e308;
        let before = state.v;
        let spike = state.step(1.0e308);
        assert_eq!(spike, 0);
        assert_eq!(state.v, before);
    }

    #[test]
    fn test_plif_invalid_runtime_state_preserves_state() {
        let mut state = ParametricLIFNeuron::new();
        state.v = 0.25;
        state.threshold = 0.0;
        let spike = state.step(0.1);
        assert_eq!(spike, 0);
        assert_eq!(state.v, 0.25);
    }
}
