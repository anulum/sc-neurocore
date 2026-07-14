// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Independent Rust safety contract for threshold-linear rate

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
        Self::with_parameters(0.0, 0.0, 1.0)
            .expect("the default threshold-linear contract is valid")
    }

    pub fn with_parameters(
        r: f64,
        theta: f64,
        gain: f64,
    ) -> Result<Self, ThresholdLinearRateError> {
        let state = Self { r, theta, gain };
        if !validate_threshold_linear_rate(&state) {
            return Err(ThresholdLinearRateError::InvalidState);
        }
        Ok(state)
    }

    pub fn step(&mut self, current: f64) -> Result<f64, ThresholdLinearRateError> {
        if !current.is_finite() {
            return Err(ThresholdLinearRateError::InvalidInput);
        }
        if !validate_threshold_linear_rate(self) {
            return Err(ThresholdLinearRateError::InvalidState);
        }
        let next_r = self.gain * (current - self.theta).max(0.0);
        if !next_r.is_finite() || next_r < 0.0 {
            return Err(ThresholdLinearRateError::NonFiniteOutput);
        }
        self.r = next_r;
        Ok(next_r)
    }

    pub fn reset(&mut self) {
        self.r = 0.0;
    }
}

impl Default for ThresholdLinearRateNeuron {
    fn default() -> Self {
        Self::new()
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
    fn default_contract_is_valid() {
        assert!(validate_threshold_linear_rate(
            &ThresholdLinearRateNeuron::new()
        ));
    }

    #[test]
    fn configured_transfer_covers_all_threshold_branches() {
        let mut state = ThresholdLinearRateNeuron::with_parameters(0.25, 1.5, 2.0).unwrap();
        assert_eq!(state.step(1.0), Ok(0.0));
        assert_eq!(state.step(1.5), Ok(0.0));
        assert_eq!(state.step(3.0), Ok(3.0));
    }

    #[test]
    fn invalid_input_does_not_mutate_output_cache() {
        let mut state = ThresholdLinearRateNeuron::with_parameters(0.25, 1.5, 2.0).unwrap();
        assert_eq!(
            state.step(f64::INFINITY),
            Err(ThresholdLinearRateError::InvalidInput)
        );
        assert_eq!(state.r, 0.25);
    }

    #[test]
    fn nonfinite_output_does_not_mutate_output_cache() {
        let mut state = ThresholdLinearRateNeuron::with_parameters(0.25, 0.0, 1.0e308).unwrap();
        assert_eq!(
            state.step(1.0e308),
            Err(ThresholdLinearRateError::NonFiniteOutput)
        );
        assert_eq!(state.r, 0.25);
    }

    #[test]
    fn reset_preserves_configuration() {
        let mut state = ThresholdLinearRateNeuron::with_parameters(0.25, -0.4, 2.5).unwrap();
        state.step(3.0).unwrap();
        state.reset();
        assert_eq!((state.r, state.theta, state.gain), (0.0, -0.4, 2.5));
    }

    #[test]
    fn constructor_rejects_invalid_contracts() {
        assert!(ThresholdLinearRateNeuron::with_parameters(-0.1, 0.0, 1.0).is_err());
        assert!(ThresholdLinearRateNeuron::with_parameters(0.0, f64::NAN, 1.0).is_err());
        assert!(ThresholdLinearRateNeuron::with_parameters(0.0, 0.0, -1.0).is_err());
    }
}
