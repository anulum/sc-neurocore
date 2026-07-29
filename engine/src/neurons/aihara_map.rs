// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — source-faithful Aihara chaotic neuron

//! Checked implementation of Aihara's reduced one-state map (1989, Eqs. 10-12).

/// Aihara's internal-state map and graded logistic output.
#[derive(Clone, Debug)]
pub struct AiharaMapNeuron {
    pub y: f64,
    pub k: f64,
    pub alpha: f64,
    pub bias: f64,
    pub epsilon: f64,
}

impl Default for AiharaMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl AiharaMapNeuron {
    pub fn new() -> Self {
        Self {
            y: 0.1,
            k: 0.7,
            alpha: 1.0,
            bias: 0.3968,
            epsilon: 0.01,
        }
    }

    fn logistic(value: f64, epsilon: f64) -> f64 {
        let argument = value / epsilon;
        if argument >= 0.0 {
            1.0 / (1.0 + (-argument).exp())
        } else {
            let exponential = argument.exp();
            exponential / (1.0 + exponential)
        }
    }

    fn valid(&self) -> bool {
        [self.y, self.k, self.alpha, self.bias, self.epsilon]
            .iter()
            .all(|value| value.is_finite())
            && (0.0..1.0).contains(&self.k)
            && self.alpha > 0.0
            && self.epsilon > 0.0
    }

    /// Current graded output `x(t)=f(y(t))` from Eq. 11.
    pub fn output(&self) -> f64 {
        Self::logistic(self.y, self.epsilon)
    }

    /// Checked Eq. 10 update; failures never mutate state.
    pub fn try_step(&mut self, current: f64) -> Result<i32, AiharaMapError> {
        if !self.valid() {
            return Err(AiharaMapError::InvalidConfiguration);
        }
        if !current.is_finite() {
            return Err(AiharaMapError::NonFiniteInput);
        }
        let next_y = self.k * self.y - self.alpha * self.output() + self.bias + current;
        if !next_y.is_finite() {
            return Err(AiharaMapError::NonFiniteCandidate);
        }
        let event = i32::from(Self::logistic(next_y, self.epsilon) >= 0.5);
        self.y = next_y;
        Ok(event)
    }

    /// Compatibility update for the engine network runner.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.y = 0.1;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AiharaMapError {
    InvalidConfiguration,
    NonFiniteInput,
    NonFiniteCandidate,
    StepLimitExceeded,
}

impl std::fmt::Display for AiharaMapError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguration => "invalid Aihara state or parameters",
            Self::NonFiniteInput => "current must contain only finite values",
            Self::NonFiniteCandidate => "Aihara map candidate must be finite",
            Self::StepLimitExceeded => "current exceeds the signed-32-bit step limit",
        })
    }
}

impl std::error::Error for AiharaMapError {}

#[derive(Clone, Debug)]
pub struct AiharaMapBatchResult {
    pub y: Vec<f64>,
    pub x: Vec<f64>,
    pub spikes: Vec<u8>,
    pub y_final: f64,
    pub x_final: f64,
    pub spike_count: usize,
}

/// Run an atomically validated piecewise-stimulus batch.
pub fn simulate_aihara_map(
    y: f64,
    k: f64,
    alpha: f64,
    bias: f64,
    epsilon: f64,
    current: &[f64],
) -> Result<AiharaMapBatchResult, AiharaMapError> {
    if current.len() > i32::MAX as usize {
        return Err(AiharaMapError::StepLimitExceeded);
    }
    let mut neuron = AiharaMapNeuron {
        y,
        k,
        alpha,
        bias,
        epsilon,
    };
    if !neuron.valid() {
        return Err(AiharaMapError::InvalidConfiguration);
    }
    if current.iter().any(|value| !value.is_finite()) {
        return Err(AiharaMapError::NonFiniteInput);
    }

    let mut y_trace = Vec::with_capacity(current.len());
    let mut x_trace = Vec::with_capacity(current.len());
    let mut spikes = Vec::with_capacity(current.len());
    let mut spike_count = 0usize;
    for &drive in current {
        let event = neuron.try_step(drive)?;
        y_trace.push(neuron.y);
        x_trace.push(neuron.output());
        spikes.push(event as u8);
        spike_count += event as usize;
    }
    Ok(AiharaMapBatchResult {
        y: y_trace,
        x: x_trace,
        spikes,
        y_final: neuron.y,
        x_final: neuron.output(),
        spike_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_matches_primary_equation() {
        let mut neuron = AiharaMapNeuron::new();
        let expected = 0.7 * 0.1 - 1.0 / (1.0 + (-10.0_f64).exp()) + 0.3968;
        assert_eq!(neuron.try_step(0.0), Ok(0));
        assert!((neuron.y - expected).abs() < 1.0e-15);
    }

    #[test]
    fn waveform_shaper_is_a_level_observable() {
        let mut neuron = AiharaMapNeuron::new();
        neuron.y = -0.1;
        neuron.k = 0.0;
        neuron.alpha = 1.0;
        neuron.bias = 0.2;
        assert_eq!(neuron.try_step(0.0), Ok(1));
        neuron.alpha = 0.01;
        assert_eq!(neuron.try_step(0.0), Ok(1));
    }

    #[test]
    fn invalid_batch_is_atomic() {
        let result = simulate_aihara_map(0.1, 0.7, 1.0, 0.3968, 0.01, &[0.0, f64::NAN]);
        assert_eq!(result.unwrap_err(), AiharaMapError::NonFiniteInput);
    }

    #[test]
    fn source_defaults_are_bounded_and_nontrivial() {
        let drive = vec![0.0; 4096];
        let result = simulate_aihara_map(0.1, 0.7, 1.0, 0.3968, 0.01, &drive).unwrap();
        assert!(result.y.iter().all(|value| value.is_finite()));
        assert!(result.spike_count > 0 && result.spike_count < drive.len());
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut neuron = AiharaMapNeuron {
            y: -2.0,
            k: 0.6,
            alpha: 2.0,
            bias: 0.5,
            epsilon: 0.015,
        };
        neuron.reset();
        assert_eq!(neuron.y, 0.1);
        assert_eq!(
            (neuron.k, neuron.alpha, neuron.bias, neuron.epsilon),
            (0.6, 2.0, 0.5, 0.015)
        );
    }
}
