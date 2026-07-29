// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained project adaptive-threshold map

//! Checked implementation of SC-NeuroCore's two-state sigmoid map.

#![warn(missing_docs)]

/// State and parameters of the retained SC adaptive-threshold map.
#[derive(Clone, Debug)]
pub struct SCAdaptiveThresholdMapNeuron {
    /// Current fast map state.
    pub x: f64,
    /// Current slow adaptive-threshold state.
    pub theta: f64,
    /// Sigmoid gain.
    pub k: f64,
    /// Slow-state decay.
    pub beta: f64,
    /// Slow-state increment after a level event.
    pub gamma: f64,
    /// Level controlling the slow-state increment.
    pub theta_spike: f64,
    /// Upward-crossing output threshold.
    pub x_threshold: f64,
}

impl Default for SCAdaptiveThresholdMapNeuron {
    fn default() -> Self {
        Self::new()
    }
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

    fn valid(&self) -> bool {
        self.x.is_finite()
            && (-5.0..=5.0).contains(&self.x)
            && self.theta.is_finite()
            && (-5.0..=5.0).contains(&self.theta)
            && self.k.is_finite()
            && (0.0..=5.0).contains(&self.k)
            && self.beta.is_finite()
            && (0.0..=1.0).contains(&self.beta)
            && self.gamma.is_finite()
            && (0.0..=2.0).contains(&self.gamma)
            && self.theta_spike.is_finite()
            && (0.0..=2.0).contains(&self.theta_spike)
            && self.x_threshold.is_finite()
            && (0.0..=2.0).contains(&self.x_threshold)
    }

    fn sigmoid(value: f64) -> f64 {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exponential = value.exp();
            exponential / (1.0 + exponential)
        }
    }

    /// Advance the simultaneous recurrence, leaving state unchanged on error.
    pub fn try_step(&mut self, current: f64) -> Result<i32, SCAdaptiveThresholdMapError> {
        if !self.valid() {
            return Err(SCAdaptiveThresholdMapError::InvalidConfiguration);
        }
        if !current.is_finite() {
            return Err(SCAdaptiveThresholdMapError::NonFiniteInput);
        }
        let previous_x = self.x;
        let activation = Self::sigmoid((self.x - self.theta) * 4.0);
        let next_x = -self.x + self.k * activation + current;
        let fired = if self.x >= self.theta_spike { 1.0 } else { 0.0 };
        let next_theta = self.beta * self.theta + self.gamma * fired;
        if !next_x.is_finite() || !next_theta.is_finite() {
            return Err(SCAdaptiveThresholdMapError::NonFiniteCandidate);
        }
        self.x = next_x.clamp(-5.0, 5.0);
        self.theta = next_theta.clamp(-5.0, 5.0);
        Ok(i32::from(
            self.x >= self.x_threshold && previous_x < self.x_threshold,
        ))
    }

    /// Advance one step and fail closed for the network-runner interface.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Restore both state variables while preserving parameters.
    pub fn reset(&mut self) {
        self.x = 0.0;
        self.theta = 0.0;
    }
}

/// Validation failures produced by the checked project map and batch runner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SCAdaptiveThresholdMapError {
    /// State or a parameter violates the project contract.
    InvalidConfiguration,
    /// A scalar or batch input is not finite.
    NonFiniteInput,
    /// An otherwise valid step produced a non-finite candidate.
    NonFiniteCandidate,
    /// A batch exceeds the signed 32-bit native ABI length.
    StepLimitExceeded,
}

impl std::fmt::Display for SCAdaptiveThresholdMapError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguration => "invalid SC adaptive-map state or parameters",
            Self::NonFiniteInput => "current must contain only finite values",
            Self::NonFiniteCandidate => "SC adaptive-map candidate must be finite",
            Self::StepLimitExceeded => "current exceeds the signed-32-bit step limit",
        })
    }
}

impl std::error::Error for SCAdaptiveThresholdMapError {}

/// Complete two-state trajectory and final receipts for one atomic batch.
#[derive(Clone, Debug)]
pub struct SCAdaptiveThresholdMapBatchResult {
    /// Fast state after every step.
    pub x: Vec<f64>,
    /// Slow threshold state after every step.
    pub theta: Vec<f64>,
    /// Upward-crossing output events.
    pub spikes: Vec<u8>,
    /// Final fast state, or initial state for an empty batch.
    pub x_final: f64,
    /// Final slow state, or initial state for an empty batch.
    pub theta_final: f64,
    /// Number of upward-crossing events in the batch.
    pub spike_count: usize,
}

#[allow(clippy::too_many_arguments)]
/// Run an atomically validated complete SC adaptive-threshold-map batch.
pub fn simulate_sc_adaptive_threshold_map(
    x: f64,
    theta: f64,
    k: f64,
    beta: f64,
    gamma: f64,
    theta_spike: f64,
    x_threshold: f64,
    current: &[f64],
) -> Result<SCAdaptiveThresholdMapBatchResult, SCAdaptiveThresholdMapError> {
    if current.len() > i32::MAX as usize {
        return Err(SCAdaptiveThresholdMapError::StepLimitExceeded);
    }
    let mut neuron = SCAdaptiveThresholdMapNeuron {
        x,
        theta,
        k,
        beta,
        gamma,
        theta_spike,
        x_threshold,
    };
    if !neuron.valid() {
        return Err(SCAdaptiveThresholdMapError::InvalidConfiguration);
    }
    if current.iter().any(|value| !value.is_finite()) {
        return Err(SCAdaptiveThresholdMapError::NonFiniteInput);
    }
    let mut x_trace = Vec::with_capacity(current.len());
    let mut theta_trace = Vec::with_capacity(current.len());
    let mut spikes = Vec::with_capacity(current.len());
    let mut spike_count = 0usize;
    for &drive in current {
        let event = neuron.try_step(drive)? as u8;
        x_trace.push(neuron.x);
        theta_trace.push(neuron.theta);
        spikes.push(event);
        spike_count += event as usize;
    }
    Ok(SCAdaptiveThresholdMapBatchResult {
        x: x_trace,
        theta: theta_trace,
        spikes,
        x_final: neuron.x,
        theta_final: neuron.theta,
        spike_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_matches_project_equation() {
        let mut neuron = SCAdaptiveThresholdMapNeuron::new();
        assert_eq!(neuron.try_step(0.6), Ok(1));
        assert_eq!((neuron.x, neuron.theta), (1.35, 0.0));
    }

    #[test]
    fn invalid_input_is_atomic() {
        let mut neuron = SCAdaptiveThresholdMapNeuron::new();
        let before = (neuron.x, neuron.theta);
        assert_eq!(
            neuron.try_step(f64::NAN),
            Err(SCAdaptiveThresholdMapError::NonFiniteInput)
        );
        assert_eq!((neuron.x, neuron.theta), before);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut neuron = SCAdaptiveThresholdMapNeuron::new();
        neuron.k = 2.0;
        neuron.try_step(1.0).unwrap();
        neuron.reset();
        assert_eq!((neuron.x, neuron.theta, neuron.k), (0.0, 0.0, 2.0));
    }
}
