// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Retained project bipolar sigma-delta accumulator.

/// Historical one-event-per-sample project state.
#[derive(Clone, Debug)]
pub struct SCSigmaDeltaAccumulatorNeuron {
    pub sigma: f64,
    pub v_threshold: f64,
}

impl SCSigmaDeltaAccumulatorNeuron {
    /// Construct the frozen project profile.
    pub fn new() -> Self {
        Self {
            sigma: 0.0,
            v_threshold: 1.0,
        }
    }
    /// Validate state and threshold.
    pub fn validate(&self) -> bool {
        self.sigma.is_finite() && self.v_threshold.is_finite() && self.v_threshold > 0.0
    }
    /// Advance one atomic bipolar accumulator transition.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !self.validate() {
            return Err("invalid SC SigmaDelta accumulator state or current");
        }
        let mut sigma = self.sigma + current;
        if !sigma.is_finite() {
            return Err("SC SigmaDelta accumulator candidate is non-finite");
        }
        let event = if sigma >= self.v_threshold {
            sigma -= self.v_threshold;
            1
        } else if sigma <= -self.v_threshold {
            sigma += self.v_threshold;
            -1
        } else {
            0
        };
        self.sigma = sigma;
        Ok(event)
    }
    /// Engine adapter: invalid state emits no event.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }
    /// Clear the accumulator.
    pub fn reset(&mut self) {
        self.sigma = 0.0;
    }
}
impl Default for SCSigmaDeltaAccumulatorNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frozen_signed_recurrence() {
        let mut neuron = SCSigmaDeltaAccumulatorNeuron::new();
        assert_eq!(neuron.try_step(3.25), Ok(1));
        assert_eq!(neuron.sigma, 2.25);
        assert_eq!(neuron.try_step(-4.5), Ok(-1));
        assert_eq!(neuron.sigma, -1.25);
    }
}
