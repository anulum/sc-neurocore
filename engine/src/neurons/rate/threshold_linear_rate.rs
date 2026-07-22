// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Threshold-linear rate neuron model

/// Threshold-linear continuous-rate transfer with cached output.
#[derive(Clone, Debug)]
pub struct ThresholdLinearRateNeuron {
    pub r: f64,
    pub theta: f64,
    pub gain: f64,
}

impl ThresholdLinearRateNeuron {
    /// Construct the maintained factory-default transfer.
    pub fn new() -> Self {
        Self::with_parameters(0.0, 0.0, 1.0)
            .expect("the factory-default threshold-linear contract is valid")
    }

    /// Construct a fully configurable, validated transfer.
    pub fn with_parameters(r: f64, theta: f64, gain: f64) -> Result<Self, String> {
        let neuron = Self { r, theta, gain };
        neuron.validate()?;
        Ok(neuron)
    }

    /// Validate the complete mutable numeric contract.
    pub fn validate(&self) -> Result<(), String> {
        if !self.r.is_finite()
            || self.r < 0.0
            || !self.theta.is_finite()
            || !self.gain.is_finite()
            || self.gain < 0.0
        {
            return Err(
                "threshold-linear rate state and parameters must be finite, with non-negative r/gain"
                    .into(),
            );
        }
        Ok(())
    }

    /// Evaluate one input, preserving the cached output on any failure.
    pub fn try_step(&mut self, current: f64) -> Result<f64, String> {
        self.validate()?;
        if !current.is_finite() {
            return Err("threshold-linear rate current must be finite".into());
        }
        let candidate = self.gain * (current - self.theta).max(0.0);
        if !candidate.is_finite() || candidate < 0.0 {
            return Err("threshold-linear rate output must remain finite and non-negative".into());
        }
        self.r = candidate;
        Ok(candidate)
    }

    /// Evaluate one input through the legacy non-throwing engine boundary.
    pub fn step(&mut self, current: f64) -> f64 {
        self.try_step(current).unwrap_or(self.r)
    }

    /// Clear the cached output without changing threshold or gain.
    pub fn reset(&mut self) {
        self.r = 0.0;
    }
}
impl Default for ThresholdLinearRateNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tl_rate() {
        let mut n = ThresholdLinearRateNeuron::new();
        assert!(n.step(5.0) > 0.0);
        assert!(n.step(-1.0) == 0.0);
    }

    #[test]
    fn tl_rate_reset() {
        let mut n = ThresholdLinearRateNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.r - 0.0).abs() < 1e-10);
    }

    #[test]
    fn tl_rate_nan_no_panic() {
        let mut neuron = ThresholdLinearRateNeuron::with_parameters(0.25, 0.5, 2.0).unwrap();
        assert_eq!(neuron.step(f64::NAN), 0.25);
        assert_eq!(neuron.r, 0.25);
    }

    #[test]
    fn tl_rate_below_threshold() {
        let mut n = ThresholdLinearRateNeuron::new();
        assert!(n.step(-5.0) == 0.0, "below threshold → zero rate");
    }

    #[test]
    fn tl_rate_configured_transfer_and_reset() {
        let mut neuron = ThresholdLinearRateNeuron::with_parameters(0.25, 1.5, 2.0).unwrap();
        assert_eq!(neuron.try_step(3.0), Ok(3.0));
        neuron.reset();
        assert_eq!((neuron.r, neuron.theta, neuron.gain), (0.0, 1.5, 2.0));
    }

    #[test]
    fn tl_rate_rejects_invalid_contract_without_mutation() {
        assert!(ThresholdLinearRateNeuron::with_parameters(-1.0, 0.0, 1.0).is_err());
        assert!(ThresholdLinearRateNeuron::with_parameters(0.0, f64::NAN, 1.0).is_err());
        assert!(ThresholdLinearRateNeuron::with_parameters(0.0, 0.0, -1.0).is_err());
        let mut neuron = ThresholdLinearRateNeuron::with_parameters(0.25, 0.0, 1.0e308).unwrap();
        assert!(neuron.try_step(1.0e308).is_err());
        assert_eq!(neuron.r, 0.25);
    }
}
