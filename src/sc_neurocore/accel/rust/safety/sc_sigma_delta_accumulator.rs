// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Fail-closed retained project bipolar accumulator.

/// Frozen SC state and threshold.
#[derive(Debug, Clone)]
pub struct SCSigmaDeltaAccumulatorNeuron {
    pub sigma: f64,
    pub v_threshold: f64,
}
impl SCSigmaDeltaAccumulatorNeuron {
    /// Construct the historical default profile.
    pub fn new() -> Self {
        Self {
            sigma: 0.0,
            v_threshold: 1.0,
        }
    }
    /// Advance one atomic project transition.
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !validate_sc_sigma_delta_accumulator(self) {
            return Err("invalid SC SigmaDelta accumulator");
        }
        let mut sigma = self.sigma + current;
        if !sigma.is_finite() {
            return Err("SC SigmaDelta candidate is non-finite");
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
/// Validate retained project state.
pub fn validate_sc_sigma_delta_accumulator(s: &SCSigmaDeltaAccumulatorNeuron) -> bool {
    s.sigma.is_finite() && s.v_threshold.is_finite() && s.v_threshold > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_transition_matches_project() {
        let mut s = SCSigmaDeltaAccumulatorNeuron::new();
        assert_eq!(s.step(3.25), Ok(1));
        assert_eq!(s.sigma, 2.25);
        assert_eq!(s.step(-4.5), Ok(-1));
    }
}
