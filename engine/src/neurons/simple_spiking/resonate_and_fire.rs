// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Izhikevich resonate-and-fire engine

//! Exact constant-real-input flow for the Izhikevich (2001) complex resonator.

use std::error::Error;
use std::fmt;

/// Typed caller-contract and numerical failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResonateAndFireError {
    /// A state or parameter is not finite.
    NonFiniteConfiguration,
    /// The angular frequency, threshold, or timestep is not positive.
    NonPositiveScale,
    /// One external-current value is not finite.
    NonFiniteInput,
    /// The exact-flow candidate is not finite.
    NonFiniteCandidate,
}

impl fmt::Display for ResonateAndFireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonFiniteConfiguration => "resonate-and-fire state and parameters must be finite",
            Self::NonPositiveScale => "resonate-and-fire omega, threshold, and dt must be positive",
            Self::NonFiniteInput => "resonate-and-fire input must contain only finite values",
            Self::NonFiniteCandidate => "resonate-and-fire exact-flow candidate must remain finite",
        };
        formatter.write_str(message)
    }
}

impl Error for ResonateAndFireError {}

/// Damped complex resonator with voltage-coordinate threshold.
#[derive(Clone, Debug)]
pub struct ResonateAndFireNeuron {
    /// Current-like real coordinate.
    pub x: f64,
    /// Voltage-like imaginary coordinate.
    pub y: f64,
    /// Radial damping/growth coefficient.
    pub b: f64,
    /// Positive angular frequency.
    pub omega: f64,
    /// Positive voltage-coordinate spike threshold.
    pub threshold: f64,
    /// Positive piecewise-constant-input sampling interval.
    pub dt: f64,
}

impl ResonateAndFireNeuron {
    /// Construct the source paper's commonly illustrated parameter set.
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            b: -1.0,
            omega: 10.0,
            threshold: 1.0,
            dt: 0.01,
        }
    }

    /// Construct and validate a complete numerical configuration.
    pub fn with_parameters(
        x: f64,
        y: f64,
        b: f64,
        omega: f64,
        threshold: f64,
        dt: f64,
    ) -> Result<Self, ResonateAndFireError> {
        let neuron = Self {
            x,
            y,
            b,
            omega,
            threshold,
            dt,
        };
        neuron.validate()?;
        Ok(neuron)
    }

    fn validate(&self) -> Result<(), ResonateAndFireError> {
        if ![self.x, self.y, self.b, self.omega, self.threshold, self.dt]
            .into_iter()
            .all(f64::is_finite)
        {
            return Err(ResonateAndFireError::NonFiniteConfiguration);
        }
        if self.omega <= 0.0 || self.threshold <= 0.0 || self.dt <= 0.0 {
            return Err(ResonateAndFireError::NonPositiveScale);
        }
        Ok(())
    }

    fn exact_flow(&self, current: f64) -> Result<(f64, f64), ResonateAndFireError> {
        let denominator = self.b * self.b + self.omega * self.omega;
        let damping_argument = self.b * self.dt;
        let angle = self.omega * self.dt;
        let x_ss = -self.b * current / denominator;
        let y_ss = self.omega * current / denominator;
        let decay = damping_argument.exp();
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        if ![
            denominator,
            damping_argument,
            angle,
            x_ss,
            y_ss,
            decay,
            cos_angle,
            sin_angle,
        ]
        .into_iter()
        .all(f64::is_finite)
            || denominator <= 0.0
        {
            return Err(ResonateAndFireError::NonFiniteCandidate);
        }
        let dx = self.x - x_ss;
        let dy = self.y - y_ss;
        let next_x = x_ss + decay * (dx * cos_angle - dy * sin_angle);
        let next_y = y_ss + decay * (dx * sin_angle + dy * cos_angle);
        if !next_x.is_finite() || !next_y.is_finite() {
            return Err(ResonateAndFireError::NonFiniteCandidate);
        }
        Ok((next_x, next_y))
    }

    /// Advance one exact-flow interval with explicit error reporting.
    pub fn try_step(&mut self, current: f64) -> Result<i32, ResonateAndFireError> {
        self.validate()?;
        if !current.is_finite() {
            return Err(ResonateAndFireError::NonFiniteInput);
        }
        let old_y = self.y;
        let (next_x, next_y) = self.exact_flow(current)?;
        if old_y < self.threshold && next_y >= self.threshold {
            self.x = 0.0;
            self.y = self.threshold;
            return Ok(1);
        }
        self.x = next_x;
        self.y = next_y;
        Ok(0)
    }

    /// Preserve the legacy scalar API while failing closed on invalid input.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Restore the quiescent initial state while preserving parameters.
    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}

impl Default for ResonateAndFireNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete post-update trace and final-state receipt.
pub struct ResonateAndFireTrace {
    /// Current-like state trace.
    pub x: Vec<f64>,
    /// Voltage-like state trace.
    pub y: Vec<f64>,
    /// Binary spike-event trace represented as floating point for ABI parity.
    pub spikes: Vec<f64>,
    /// Final ``[x, y]`` state, including the initial state for an empty batch.
    pub final_state: [f64; 2],
    /// Number of sampled upward threshold crossings.
    pub spike_count: usize,
}

/// Simulate a complete caller-owned piecewise-constant current vector atomically.
pub fn simulate(
    x: f64,
    y: f64,
    b: f64,
    omega: f64,
    threshold: f64,
    dt: f64,
    current: &[f64],
) -> Result<ResonateAndFireTrace, ResonateAndFireError> {
    let mut neuron = ResonateAndFireNeuron::with_parameters(x, y, b, omega, threshold, dt)?;
    if !current.iter().all(|value| value.is_finite()) {
        return Err(ResonateAndFireError::NonFiniteInput);
    }
    let mut x_trace = Vec::with_capacity(current.len());
    let mut y_trace = Vec::with_capacity(current.len());
    let mut spikes = Vec::with_capacity(current.len());
    let mut spike_count = 0usize;
    for drive in current {
        let spike = neuron.try_step(*drive)?;
        spike_count += spike as usize;
        x_trace.push(neuron.x);
        y_trace.push(neuron.y);
        spikes.push(f64::from(spike));
    }
    Ok(ResonateAndFireTrace {
        x: x_trace,
        y: y_trace,
        spikes,
        final_state: [neuron.x, neuron.y],
        spike_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_source_illustration_parameters() {
        let neuron = ResonateAndFireNeuron::new();
        assert_eq!(
            (neuron.b, neuron.omega, neuron.threshold, neuron.dt),
            (-1.0, 10.0, 1.0, 0.01)
        );
    }

    #[test]
    fn one_step_matches_closed_form_rotation() {
        let mut neuron =
            ResonateAndFireNeuron::with_parameters(0.3, -0.2, -0.2, 1.7, 100.0, 1.25).unwrap();
        let current = 0.8;
        let denominator = neuron.b * neuron.b + neuron.omega * neuron.omega;
        let x_ss = -neuron.b * current / denominator;
        let y_ss = neuron.omega * current / denominator;
        let decay = (neuron.b * neuron.dt).exp();
        let angle = neuron.omega * neuron.dt;
        let expected_x =
            x_ss + decay * ((neuron.x - x_ss) * angle.cos() - (neuron.y - y_ss) * angle.sin());
        let expected_y =
            y_ss + decay * ((neuron.x - x_ss) * angle.sin() + (neuron.y - y_ss) * angle.cos());
        assert_eq!(neuron.try_step(current), Ok(0));
        assert!((neuron.x - expected_x).abs() < 1.0e-12);
        assert!((neuron.y - expected_y).abs() < 1.0e-12);
    }

    #[test]
    fn spike_uses_voltage_crossing_and_source_reset() {
        let mut neuron =
            ResonateAndFireNeuron::with_parameters(0.0, 0.99, 0.0, 1.0, 1.0, 0.1).unwrap();
        assert_eq!(neuron.try_step(10.0), Ok(1));
        assert_eq!((neuron.x, neuron.y), (0.0, 1.0));
        assert_eq!(neuron.try_step(0.0), Ok(0));
    }

    #[test]
    fn radius_alone_does_not_spike() {
        let mut neuron =
            ResonateAndFireNeuron::with_parameters(2.0, 0.0, 0.0, 1.0, 1.0, 0.01).unwrap();
        assert_eq!(neuron.try_step(0.0), Ok(0));
    }

    #[test]
    fn invalid_step_is_atomic() {
        let mut neuron = ResonateAndFireNeuron::new();
        neuron.x = 0.25;
        neuron.y = -0.5;
        let before = (neuron.x, neuron.y);
        assert_eq!(
            neuron.try_step(f64::NAN),
            Err(ResonateAndFireError::NonFiniteInput)
        );
        assert_eq!((neuron.x, neuron.y), before);
    }

    #[test]
    fn batch_matches_scalar_and_empty_preserves_initial_state() {
        let empty = simulate(0.2, -0.1, -1.0, 10.0, 1.0, 0.01, &[]).unwrap();
        assert!(empty.x.is_empty() && empty.y.is_empty() && empty.spikes.is_empty());
        assert_eq!(empty.final_state, [0.2, -0.1]);

        let drive = [0.0, 0.25, -0.1, 1.0];
        let batch = simulate(0.0, 0.0, -1.0, 10.0, 1.0, 0.01, &drive).unwrap();
        let mut scalar = ResonateAndFireNeuron::new();
        let mut count = 0;
        for value in drive {
            count += scalar.try_step(value).unwrap() as usize;
        }
        assert_eq!(batch.final_state, [scalar.x, scalar.y]);
        assert_eq!(batch.spike_count, count);
    }

    #[test]
    fn reset_restores_quiescent_state_not_post_spike_state() {
        let mut neuron = ResonateAndFireNeuron::new();
        neuron.x = 0.4;
        neuron.y = 0.8;
        neuron.reset();
        assert_eq!((neuron.x, neuron.y), (0.0, 0.0));
        assert_eq!(
            (neuron.b, neuron.omega, neuron.threshold, neuron.dt),
            (-1.0, 10.0, 1.0, 0.01)
        );
    }
}
