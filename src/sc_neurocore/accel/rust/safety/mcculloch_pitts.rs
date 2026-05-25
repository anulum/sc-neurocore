// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for mcculloch_pitts

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct McCullochPittsNeuron {
    pub theta: f64,
}

impl McCullochPittsNeuron {
    pub fn new() -> Self {
        Self { theta: 1.0_f64 }
    }

    pub fn step(&mut self, weighted_input: f64) -> Result<i32, &'static str> {
        if !weighted_input.is_finite() {
            return Err("mcculloch pitts weighted input must be finite");
        }
        if !validate_mcculloch_pitts(self) {
            return Err("mcculloch pitts threshold must be finite");
        }
        Ok(if weighted_input >= self.theta { 1 } else { 0 })
    }

    pub fn reset(&mut self) {
        // Stateless model: reset is intentionally a no-op.
    }
}

pub fn validate_mcculloch_pitts(state: &McCullochPittsNeuron) -> bool {
    state.theta.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcculloch_pitts_new() {
        let state = McCullochPittsNeuron::new();
        assert!(validate_mcculloch_pitts(&state));
    }

    #[test]
    fn test_mcculloch_pitts_step() {
        let mut state = McCullochPittsNeuron::new();
        let spike = state.step(10.0).expect("valid step must succeed");
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_heaviside_boundary() {
        let mut state = McCullochPittsNeuron { theta: 2.0 };
        assert_eq!(state.step(1.999999999999999), Ok(0));
        assert_eq!(state.step(2.0), Ok(1));
    }

    #[test]
    fn test_invalid_runtime_threshold_fails() {
        let mut state = McCullochPittsNeuron { theta: f64::NAN };
        assert!(state.step(1.0).is_err());
    }

    #[test]
    fn test_non_finite_input_fails() {
        let mut state = McCullochPittsNeuron::new();
        assert!(state.step(f64::INFINITY).is_err());
    }
}
