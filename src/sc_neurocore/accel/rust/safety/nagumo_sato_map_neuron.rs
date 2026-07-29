// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — standalone Nagumo–Sato safety mirror

#![allow(dead_code)]
#![warn(missing_docs)]

/// Standalone safety mirror of the source-faithful Nagumo–Sato map.
#[derive(Debug, Clone)]
pub struct NagumoSatoMapNeuron {
    /// Current internal state.
    pub y: f64,
    /// Refractory-memory damping factor.
    pub k: f64,
    /// Positive refractory decrement.
    pub alpha: f64,
    /// Constant transformed stimulus.
    pub bias: f64,
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

    /// Return `H(y)`, with `H(0)=1`.
    pub fn output(&self) -> i32 {
        i32::from(self.y >= 0.0)
    }

    /// Advance one step atomically or return a static safety error.
    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !validate_nagumo_sato_map_neuron(self) {
            return Err("Nagumo-Sato state and parameters must satisfy source bounds");
        }
        let next_y = self.k * self.y - self.alpha * f64::from(self.output()) + self.bias + current;
        if !next_y.is_finite() {
            return Err("Nagumo-Sato candidate state became non-finite");
        }
        self.y = next_y;
        Ok(self.output())
    }

    /// Restore the source initial state while preserving parameters.
    pub fn reset(&mut self) {
        self.y = 0.1;
    }
}

/// Return whether state and parameters satisfy the source contract.
pub fn validate_nagumo_sato_map_neuron(state: &NagumoSatoMapNeuron) -> bool {
    [state.y, state.k, state.alpha, state.bias]
        .iter()
        .all(|value| value.is_finite())
        && (0.0..1.0).contains(&state.k)
        && state.alpha > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_step_and_atomic_failure() {
        let mut state = NagumoSatoMapNeuron::new();
        assert_eq!(state.step(0.0), Ok(0));
        assert!((state.y + 0.74).abs() < 1e-15);
        let before = state.y;
        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state.y, before);
    }
}
