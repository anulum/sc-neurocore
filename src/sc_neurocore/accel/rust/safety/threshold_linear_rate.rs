// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for threshold_linear_rate

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct ThresholdLinearRateNeuron {
    pub r: f64,
    pub theta: f64,
    pub gain: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdLinearRateError {
    InvalidInput,
    InvalidState,
    NonFiniteOutput,
}

impl ThresholdLinearRateNeuron {
    pub fn new() -> Self {
        Self {
            r: 0.0_f64,
            theta: 0.0_f64,
            gain: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<f64, ThresholdLinearRateError> {
        if !i_ext.is_finite() {
            return Err(ThresholdLinearRateError::InvalidInput);
        }
        if !validate_threshold_linear_rate(self) {
            return Err(ThresholdLinearRateError::InvalidState);
        }

        let drive = (i_ext - self.theta).max(0.0);
        let next_r = self.gain * drive;
        if !next_r.is_finite() || next_r < 0.0 {
            return Err(ThresholdLinearRateError::NonFiniteOutput);
        }
        self.r = next_r;
        Ok(next_r)
    }

    pub fn reset(&mut self) {
        self.r = 0.0_f64;
    }
}

pub fn validate_threshold_linear_rate(state: &ThresholdLinearRateNeuron) -> bool {
    state.r.is_finite()
        && state.r >= 0.0
        && state.theta.is_finite()
        && state.gain.is_finite()
        && state.gain >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_linear_rate_new() {
        let state = ThresholdLinearRateNeuron::new();
        assert!(validate_threshold_linear_rate(&state));
    }

    #[test]
    fn test_threshold_linear_rate_step() {
        let mut state = ThresholdLinearRateNeuron::new();
        assert_eq!(state.step(10.0), Ok(10.0));
    }

    #[test]
    fn test_threshold_linear_rate_rejects_invalid_input_without_mutation() {
        let mut state = ThresholdLinearRateNeuron::new();
        let before = state.r;
        assert_eq!(
            state.step(f64::INFINITY),
            Err(ThresholdLinearRateError::InvalidInput)
        );
        assert_eq!(state.r, before);
    }

    #[test]
    fn test_threshold_linear_rate_rejects_nonfinite_output_without_mutation() {
        let mut state = ThresholdLinearRateNeuron::new();
        state.gain = 1.0e308;
        state.r = 0.25;
        let before = state.r;
        assert_eq!(
            state.step(1.0e308),
            Err(ThresholdLinearRateError::NonFiniteOutput)
        );
        assert_eq!(state.r, before);
    }
}
