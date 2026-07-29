// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for Aihara Eqs. 10-12

#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct AiharaMapNeuron {
    pub y: f64,
    pub k: f64,
    pub alpha: f64,
    pub bias: f64,
    pub epsilon: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AiharaMapError {
    InvalidInput,
    InvalidState,
    NonFiniteUpdate,
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

    pub fn output(&self) -> f64 {
        logistic(self.y / self.epsilon)
    }

    pub fn step(&mut self, current: f64) -> Result<i32, AiharaMapError> {
        if !current.is_finite() {
            return Err(AiharaMapError::InvalidInput);
        }
        if !validate_aihara_map_neuron(self) {
            return Err(AiharaMapError::InvalidState);
        }
        let next_y = self.k * self.y - self.alpha * self.output() + self.bias + current;
        if !next_y.is_finite() {
            return Err(AiharaMapError::NonFiniteUpdate);
        }
        let event = i32::from(logistic(next_y / self.epsilon) >= 0.5);
        self.y = next_y;
        Ok(event)
    }

    pub fn reset(&mut self) {
        self.y = 0.1;
    }
}

pub fn validate_aihara_map_neuron(state: &AiharaMapNeuron) -> bool {
    [state.y, state.k, state.alpha, state.bias, state.epsilon]
        .iter()
        .all(|value| value.is_finite())
        && (0.0..1.0).contains(&state.k)
        && state.alpha > 0.0
        && state.epsilon > 0.0
}

fn logistic(argument: f64) -> f64 {
    if argument >= 0.0 {
        1.0 / (1.0 + (-argument).exp())
    } else {
        let exponential = argument.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_step_and_level_event() {
        let mut state = AiharaMapNeuron::new();
        assert_eq!(state.step(0.0), Ok(0));
        let expected = 0.07 - 1.0 / (1.0 + (-10.0_f64).exp()) + 0.3968;
        assert!((state.y - expected).abs() < 1.0e-15);
        state.y = -0.1;
        state.k = 0.0;
        state.alpha = 0.01;
        state.bias = 0.2;
        assert_eq!(state.step(0.0), Ok(1));
        assert_eq!(state.step(0.0), Ok(1));
    }

    #[test]
    fn errors_leave_state_unchanged() {
        let mut state = AiharaMapNeuron::new();
        let before = state.y;
        assert_eq!(state.step(f64::NAN), Err(AiharaMapError::InvalidInput));
        assert_eq!(state.y, before);
        state.epsilon = 0.0;
        assert_eq!(state.step(0.0), Err(AiharaMapError::InvalidState));
        assert_eq!(state.y, before);
    }
}
