// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Composite reduced adaptive-threshold leaky integrate-and-fire engine

//! Exact constant-input relaxation for the composite reduced adaptive-threshold
//! leaky integrate-and-fire neuron: leaky membrane relaxation, the a=0
//! Mihalas–Niebur threshold-decay limit, and the Platkiewicz–Brette fixed
//! post-spike threshold shift.

use std::error::Error;
use std::fmt;

/// Typed caller-contract and numerical failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveThresholdIFError {
    /// A state or parameter is not finite.
    NonFiniteConfiguration,
    /// A magnitude or ordering constraint is violated.
    InvalidScale,
    /// One external-current value is not finite.
    NonFiniteInput,
    /// The exact-relaxation candidate is not finite.
    NonFiniteCandidate,
}

impl fmt::Display for AdaptiveThresholdIFError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonFiniteConfiguration => {
                "adaptive-threshold state and parameters must be finite"
            }
            Self::InvalidScale => {
                "adaptive-threshold delta_theta must be non-negative, tau_m/tau_theta/dt must be positive, and theta_rest must exceed v_rest and v_reset"
            }
            Self::NonFiniteInput => "adaptive-threshold input must contain only finite values",
            Self::NonFiniteCandidate => {
                "adaptive-threshold exact-relaxation candidate must remain finite"
            }
        };
        formatter.write_str(message)
    }
}

impl Error for AdaptiveThresholdIFError {}

/// Composite reduced adaptive-threshold leaky integrate-and-fire neuron.
#[derive(Clone, Debug)]
pub struct AdaptiveThresholdIFNeuron {
    /// Membrane potential in millivolts.
    pub v: f64,
    /// Adaptive threshold in millivolts.
    pub theta: f64,
    /// Leak reversal potential in millivolts.
    pub v_rest: f64,
    /// Post-spike membrane reset in millivolts.
    pub v_reset: f64,
    /// Baseline threshold in millivolts.
    pub theta_rest: f64,
    /// Fixed non-negative post-spike threshold shift in millivolts.
    pub delta_theta: f64,
    /// Positive membrane time constant in milliseconds.
    pub tau_m: f64,
    /// Positive threshold relaxation time constant in milliseconds.
    pub tau_theta: f64,
    /// Positive piecewise-constant-input sampling interval in milliseconds.
    pub dt: f64,
}

impl AdaptiveThresholdIFNeuron {
    /// Construct the catalogue model-family defaults.
    pub fn new() -> Self {
        Self {
            v: -65.0,
            theta: -50.0,
            v_rest: -65.0,
            v_reset: -65.0,
            theta_rest: -50.0,
            delta_theta: 5.0,
            tau_m: 10.0,
            tau_theta: 50.0,
            dt: 0.1,
        }
    }

    /// Construct and validate a complete numerical configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_parameters(
        v: f64,
        theta: f64,
        v_rest: f64,
        v_reset: f64,
        theta_rest: f64,
        delta_theta: f64,
        tau_m: f64,
        tau_theta: f64,
        dt: f64,
    ) -> Result<Self, AdaptiveThresholdIFError> {
        let neuron = Self {
            v,
            theta,
            v_rest,
            v_reset,
            theta_rest,
            delta_theta,
            tau_m,
            tau_theta,
            dt,
        };
        neuron.validate()?;
        Ok(neuron)
    }

    fn validate(&self) -> Result<(), AdaptiveThresholdIFError> {
        if ![
            self.v,
            self.theta,
            self.v_rest,
            self.v_reset,
            self.theta_rest,
            self.delta_theta,
            self.tau_m,
            self.tau_theta,
            self.dt,
        ]
        .into_iter()
        .all(f64::is_finite)
        {
            return Err(AdaptiveThresholdIFError::NonFiniteConfiguration);
        }
        if self.delta_theta < 0.0
            || self.tau_m <= 0.0
            || self.tau_theta <= 0.0
            || self.dt <= 0.0
            || self.theta_rest <= self.v_rest
            || self.theta_rest <= self.v_reset
        {
            return Err(AdaptiveThresholdIFError::InvalidScale);
        }
        Ok(())
    }

    fn exact_relaxation(
        state: f64,
        steady_state: f64,
        tau: f64,
        dt: f64,
    ) -> Result<f64, AdaptiveThresholdIFError> {
        let decay = (-dt / tau).exp();
        let candidate = steady_state + (state - steady_state) * decay;
        if !candidate.is_finite() {
            return Err(AdaptiveThresholdIFError::NonFiniteCandidate);
        }
        Ok(candidate)
    }

    /// Advance one exact-relaxation interval with explicit error reporting.
    pub fn try_step(&mut self, current: f64) -> Result<i32, AdaptiveThresholdIFError> {
        self.validate()?;
        if !current.is_finite() {
            return Err(AdaptiveThresholdIFError::NonFiniteInput);
        }
        let next_v = Self::exact_relaxation(self.v, self.v_rest + current, self.tau_m, self.dt)?;
        let next_theta =
            Self::exact_relaxation(self.theta, self.theta_rest, self.tau_theta, self.dt)?;
        if next_v >= next_theta {
            let spike_theta = next_theta + self.delta_theta;
            if !spike_theta.is_finite() {
                return Err(AdaptiveThresholdIFError::NonFiniteCandidate);
            }
            self.v = self.v_reset;
            self.theta = spike_theta;
            return Ok(1);
        }
        self.v = next_v;
        self.theta = next_theta;
        Ok(0)
    }

    /// Preserve the legacy scalar API while failing closed on invalid input.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Restore the documented rest state while preserving configuration.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_rest;
    }
}

impl Default for AdaptiveThresholdIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete post-update trace and final-state receipt.
pub struct AdaptiveThresholdIFTrace {
    /// Membrane-potential state trace.
    pub v: Vec<f64>,
    /// Adaptive-threshold state trace.
    pub theta: Vec<f64>,
    /// Binary spike-event trace represented as floating point for ABI parity.
    pub spikes: Vec<f64>,
    /// Final ``[v, theta]`` state, including the initial state for an empty batch.
    pub final_state: [f64; 2],
    /// Number of candidate-crossing spikes.
    pub spike_count: usize,
}

/// Simulate a complete caller-owned piecewise-constant current vector atomically.
#[allow(clippy::too_many_arguments)]
pub fn simulate(
    v: f64,
    theta: f64,
    v_rest: f64,
    v_reset: f64,
    theta_rest: f64,
    delta_theta: f64,
    tau_m: f64,
    tau_theta: f64,
    dt: f64,
    current: &[f64],
) -> Result<AdaptiveThresholdIFTrace, AdaptiveThresholdIFError> {
    let mut neuron = AdaptiveThresholdIFNeuron::with_parameters(
        v,
        theta,
        v_rest,
        v_reset,
        theta_rest,
        delta_theta,
        tau_m,
        tau_theta,
        dt,
    )?;
    if !current.iter().all(|value| value.is_finite()) {
        return Err(AdaptiveThresholdIFError::NonFiniteInput);
    }
    let mut v_trace = Vec::with_capacity(current.len());
    let mut theta_trace = Vec::with_capacity(current.len());
    let mut spikes = Vec::with_capacity(current.len());
    let mut spike_count = 0usize;
    for drive in current {
        let spike = neuron.try_step(*drive)?;
        spike_count += spike as usize;
        v_trace.push(neuron.v);
        theta_trace.push(neuron.theta);
        spikes.push(f64::from(spike));
    }
    Ok(AdaptiveThresholdIFTrace {
        v: v_trace,
        theta: theta_trace,
        spikes,
        final_state: [neuron.v, neuron.theta],
        spike_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_catalogue_model_family() {
        let neuron = AdaptiveThresholdIFNeuron::new();
        assert_eq!(
            (
                neuron.v,
                neuron.theta,
                neuron.v_rest,
                neuron.v_reset,
                neuron.theta_rest,
                neuron.delta_theta,
                neuron.tau_m,
                neuron.tau_theta,
                neuron.dt,
            ),
            (-65.0, -50.0, -65.0, -65.0, -50.0, 5.0, 10.0, 50.0, 0.1)
        );
    }

    #[test]
    fn one_step_matches_exact_relaxation_closed_form() {
        let mut neuron = AdaptiveThresholdIFNeuron::with_parameters(
            -60.0, -52.0, -70.0, -68.0, -48.0, 3.0, 8.0, 40.0, 0.05,
        )
        .unwrap();
        let current = 12.5;
        let decay_v = (-neuron.dt / neuron.tau_m).exp();
        let decay_theta = (-neuron.dt / neuron.tau_theta).exp();
        let expected_v =
            (neuron.v_rest + current) + (neuron.v - (neuron.v_rest + current)) * decay_v;
        let expected_theta = neuron.theta_rest + (neuron.theta - neuron.theta_rest) * decay_theta;
        assert_eq!(neuron.try_step(current), Ok(0));
        assert!((neuron.v - expected_v).abs() < 1.0e-12);
        assert!((neuron.theta - expected_theta).abs() < 1.0e-12);
    }

    #[test]
    fn spike_resets_voltage_and_shifts_threshold() {
        let mut neuron = AdaptiveThresholdIFNeuron::with_parameters(
            -50.5, -51.0, -65.0, -65.0, -50.0, 5.0, 10.0, 50.0, 0.1,
        )
        .unwrap();
        assert_eq!(neuron.try_step(0.0), Ok(1));
        assert_eq!(neuron.v, -65.0);
        let decay_theta = (-0.1_f64 / 50.0).exp();
        let relaxed = -50.0 + (-51.0 + 50.0) * decay_theta;
        assert!((neuron.theta - (relaxed + 5.0)).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_step_is_atomic() {
        let mut neuron = AdaptiveThresholdIFNeuron::new();
        neuron.v = -60.0;
        neuron.theta = -55.0;
        let before = (neuron.v, neuron.theta);
        assert_eq!(
            neuron.try_step(f64::NAN),
            Err(AdaptiveThresholdIFError::NonFiniteInput)
        );
        assert_eq!((neuron.v, neuron.theta), before);
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        assert!(matches!(
            AdaptiveThresholdIFNeuron::with_parameters(
                -65.0, -50.0, -65.0, -65.0, -70.0, 5.0, 10.0, 50.0, 0.1
            ),
            Err(AdaptiveThresholdIFError::InvalidScale)
        ));
        assert!(matches!(
            AdaptiveThresholdIFNeuron::with_parameters(
                -65.0, -50.0, -65.0, -65.0, -50.0, -1.0, 10.0, 50.0, 0.1
            ),
            Err(AdaptiveThresholdIFError::InvalidScale)
        ));
        assert!(matches!(
            AdaptiveThresholdIFNeuron::with_parameters(
                -65.0, -50.0, -65.0, -65.0, -50.0, 5.0, 0.0, 50.0, 0.1
            ),
            Err(AdaptiveThresholdIFError::InvalidScale)
        ));
    }

    #[test]
    fn batch_matches_scalar_and_empty_preserves_initial_state() {
        let empty = simulate(-60.0, -55.0, -65.0, -65.0, -50.0, 5.0, 10.0, 50.0, 0.1, &[]).unwrap();
        assert!(empty.v.is_empty() && empty.theta.is_empty() && empty.spikes.is_empty());
        assert_eq!(empty.final_state, [-60.0, -55.0]);

        let drive = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0];
        let batch = simulate(
            -65.0, -50.0, -65.0, -65.0, -50.0, 5.0, 10.0, 50.0, 0.1, &drive,
        )
        .unwrap();
        let mut scalar = AdaptiveThresholdIFNeuron::new();
        let mut count = 0;
        for value in drive {
            count += scalar.try_step(value).unwrap() as usize;
        }
        assert_eq!(batch.final_state, [scalar.v, scalar.theta]);
        assert_eq!(batch.spike_count, count);
    }

    #[test]
    fn reset_restores_documented_rest_state_not_configuration() {
        let mut neuron = AdaptiveThresholdIFNeuron::new();
        neuron.v = -55.0;
        neuron.theta = -40.0;
        neuron.reset();
        assert_eq!((neuron.v, neuron.theta), (-65.0, -50.0));
        assert_eq!(
            (
                neuron.delta_theta,
                neuron.tau_m,
                neuron.tau_theta,
                neuron.dt
            ),
            (5.0, 10.0, 50.0, 0.1)
        );
    }
}
