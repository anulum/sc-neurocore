// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — standalone project adaptive-threshold-map safety mirror

#![allow(dead_code)]
#![warn(missing_docs)]

/// Standalone safety mirror of the retained SC adaptive-threshold map.
#[derive(Debug, Clone)]
pub struct SCAdaptiveThresholdMapNeuron {
    /// Current fast state.
    pub x: f64,
    /// Current slow threshold state.
    pub theta: f64,
    /// Sigmoid gain.
    pub k: f64,
    /// Slow-state decay.
    pub beta: f64,
    /// Slow-state increment.
    pub gamma: f64,
    /// Level controlling slow-state adaptation.
    pub theta_spike: f64,
    /// Upward-crossing event threshold.
    pub x_threshold: f64,
}

impl SCAdaptiveThresholdMapNeuron {
    /// Construct the documented project reference configuration.
    pub fn new() -> Self {
        Self {
            x: 0.0,
            theta: 0.0,
            k: 1.5,
            beta: 0.95,
            gamma: 0.3,
            theta_spike: 0.8,
            x_threshold: 0.8,
        }
    }

    fn sigmoid(value: f64) -> f64 {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exponential = value.exp();
            exponential / (1.0 + exponential)
        }
    }

    /// Advance one simultaneous step atomically.
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !validate_sc_adaptive_threshold_map_neuron(self) {
            return Err("SC adaptive-map state and parameters must satisfy public bounds");
        }
        let previous_x = self.x;
        let activation = Self::sigmoid((self.x - self.theta) * 4.0);
        let next_x = -self.x + self.k * activation + current;
        let fired = if self.x >= self.theta_spike { 1.0 } else { 0.0 };
        let next_theta = self.beta * self.theta + self.gamma * fired;
        if !next_x.is_finite() || !next_theta.is_finite() {
            return Err("SC adaptive-map candidate state became non-finite");
        }
        self.x = next_x.clamp(-5.0, 5.0);
        self.theta = next_theta.clamp(-5.0, 5.0);
        Ok(i32::from(
            self.x >= self.x_threshold && previous_x < self.x_threshold,
        ))
    }

    /// Restore both states while preserving parameters.
    pub fn reset(&mut self) {
        self.x = 0.0;
        self.theta = 0.0;
    }
}

/// Return whether state and parameters satisfy the project contract.
pub fn validate_sc_adaptive_threshold_map_neuron(state: &SCAdaptiveThresholdMapNeuron) -> bool {
    state.x.is_finite()
        && (-5.0..=5.0).contains(&state.x)
        && state.theta.is_finite()
        && (-5.0..=5.0).contains(&state.theta)
        && state.k.is_finite()
        && (0.0..=5.0).contains(&state.k)
        && state.beta.is_finite()
        && (0.0..=1.0).contains(&state.beta)
        && state.gamma.is_finite()
        && (0.0..=2.0).contains(&state.gamma)
        && state.theta_spike.is_finite()
        && (0.0..=2.0).contains(&state.theta_spike)
        && state.x_threshold.is_finite()
        && (0.0..=2.0).contains(&state.x_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_step_and_atomic_failure() {
        let mut state = SCAdaptiveThresholdMapNeuron::new();
        assert_eq!(state.step(0.6), Ok(1));
        assert_eq!((state.x, state.theta), (1.35, 0.0));
        let before = (state.x, state.theta);
        assert!(state.step(f64::NAN).is_err());
        assert_eq!((state.x, state.theta), before);
    }
}
