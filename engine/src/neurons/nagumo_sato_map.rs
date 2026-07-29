// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — source-faithful Nagumo–Sato refractory map

//! Checked implementation of the Nagumo–Sato reduction (Aihara 1989, Eqs. 1-7).

#![warn(missing_docs)]

/// Source-faithful internal state and parameters of the Nagumo–Sato map.
#[derive(Clone, Debug)]
pub struct NagumoSatoMapNeuron {
    /// Current internal state `y(t)`.
    pub y: f64,
    /// Refractory-memory damping factor in `[0, 1)`.
    pub k: f64,
    /// Positive refractory decrement.
    pub alpha: f64,
    /// Constant transformed stimulus `a`.
    pub bias: f64,
}

impl Default for NagumoSatoMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl NagumoSatoMapNeuron {
    /// Construct the documented source operating configuration.
    pub fn new() -> Self {
        Self {
            y: 0.1,
            k: 0.6,
            alpha: 1.0,
            bias: 0.2,
        }
    }

    fn heaviside(value: f64) -> i32 {
        i32::from(value >= 0.0)
    }

    fn valid(&self) -> bool {
        [self.y, self.k, self.alpha, self.bias]
            .iter()
            .all(|value| value.is_finite())
            && (0.0..1.0).contains(&self.k)
            && self.alpha > 0.0
    }

    /// Return the all-or-none source output `H(y)`, with `H(0)=1`.
    pub fn output(&self) -> i32 {
        Self::heaviside(self.y)
    }

    /// Advance one source-equation step, leaving state unchanged on error.
    pub fn try_step(&mut self, current: f64) -> Result<i32, NagumoSatoMapError> {
        if !self.valid() {
            return Err(NagumoSatoMapError::InvalidConfiguration);
        }
        if !current.is_finite() {
            return Err(NagumoSatoMapError::NonFiniteInput);
        }
        let next_y = self.k * self.y - self.alpha * f64::from(self.output()) + self.bias + current;
        if !next_y.is_finite() {
            return Err(NagumoSatoMapError::NonFiniteCandidate);
        }
        let event = Self::heaviside(next_y);
        self.y = next_y;
        Ok(event)
    }

    /// Advance one step and fail closed for the network-runner interface.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Restore `y` to the source initial condition while preserving parameters.
    pub fn reset(&mut self) {
        self.y = 0.1;
    }
}

/// Validation failures produced by the checked source map and batch runner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NagumoSatoMapError {
    /// State or a parameter violates the source contract.
    InvalidConfiguration,
    /// A scalar or batch input is not finite.
    NonFiniteInput,
    /// An otherwise valid step produced a non-finite candidate.
    NonFiniteCandidate,
    /// A batch exceeds the signed 32-bit native ABI length.
    StepLimitExceeded,
}

impl std::fmt::Display for NagumoSatoMapError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguration => "invalid Nagumo-Sato state or parameters",
            Self::NonFiniteInput => "current must contain only finite values",
            Self::NonFiniteCandidate => "Nagumo-Sato map candidate must be finite",
            Self::StepLimitExceeded => "current exceeds the signed-32-bit step limit",
        })
    }
}

impl std::error::Error for NagumoSatoMapError {}

/// Complete state/output trajectory and final receipts for one atomic batch.
#[derive(Clone, Debug)]
pub struct NagumoSatoMapBatchResult {
    /// Internal state after every step.
    pub y: Vec<f64>,
    /// All-or-none output after every step.
    pub x: Vec<u8>,
    /// Event alias of the all-or-none output.
    pub spikes: Vec<u8>,
    /// Final internal state, or the initial state for an empty batch.
    pub y_final: f64,
    /// Final all-or-none output.
    pub x_final: u8,
    /// Number of firing outputs in the batch.
    pub spike_count: usize,
}

/// Run an atomically validated complete Nagumo–Sato batch.
pub fn simulate_nagumo_sato_map(
    y: f64,
    k: f64,
    alpha: f64,
    bias: f64,
    current: &[f64],
) -> Result<NagumoSatoMapBatchResult, NagumoSatoMapError> {
    if current.len() > i32::MAX as usize {
        return Err(NagumoSatoMapError::StepLimitExceeded);
    }
    let mut neuron = NagumoSatoMapNeuron { y, k, alpha, bias };
    if !neuron.valid() {
        return Err(NagumoSatoMapError::InvalidConfiguration);
    }
    if current.iter().any(|value| !value.is_finite()) {
        return Err(NagumoSatoMapError::NonFiniteInput);
    }
    let mut y_trace = Vec::with_capacity(current.len());
    let mut output = Vec::with_capacity(current.len());
    let mut spike_count = 0usize;
    for &drive in current {
        let event = neuron.try_step(drive)? as u8;
        y_trace.push(neuron.y);
        output.push(event);
        spike_count += event as usize;
    }
    Ok(NagumoSatoMapBatchResult {
        y: y_trace,
        x: output.clone(),
        spikes: output,
        y_final: neuron.y,
        x_final: neuron.output() as u8,
        spike_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_steps_match_source_equation() {
        let mut neuron = NagumoSatoMapNeuron::new();
        assert_eq!(neuron.try_step(0.0), Ok(0));
        assert!((neuron.y - (-0.74)).abs() < 1.0e-15);
        assert_eq!(neuron.try_step(0.0), Ok(0));
        assert!((neuron.y - (-0.244)).abs() < 1.0e-15);
        assert_eq!(neuron.try_step(0.0), Ok(1));
        assert!((neuron.y - 0.0536).abs() < 1.0e-15);
    }

    #[test]
    fn invalid_batch_is_atomic() {
        assert_eq!(
            simulate_nagumo_sato_map(0.1, 0.6, 1.0, 0.2, &[0.0, f64::NAN]).unwrap_err(),
            NagumoSatoMapError::NonFiniteInput
        );
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut neuron = NagumoSatoMapNeuron {
            y: -2.0,
            k: 0.5,
            alpha: 2.0,
            bias: 0.7,
        };
        neuron.reset();
        assert_eq!(
            (neuron.y, neuron.k, neuron.alpha, neuron.bias),
            (0.1, 0.5, 2.0, 0.7)
        );
    }
}
