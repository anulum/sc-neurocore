// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — XOR-nonlinearity dendritic neuron

#![deny(missing_docs)]

/// XOR-nonlinearity dendritic neuron.
///
/// Koch, Biophysics of Computation, 1999, Ch. 12.
/// Output = 1 if `(d1 + d2 - 2*d1*d2) > threshold`.
#[derive(Clone, Debug)]
pub struct DendriticNeuron {
    /// Output threshold on the XOR combination (dimensionless); the comparison is strict.
    pub threshold: f64,
    last_current: f64,
}

impl DendriticNeuron {
    /// Create a neuron with the given threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            last_current: 0.0,
        }
    }

    /// Create a neuron with threshold 0.5.
    pub fn with_defaults() -> Self {
        Self::new(0.5)
    }

    /// Return 1 when `input_a + input_b - 2 * input_a * input_b` exceeds `threshold`, else
    /// 0. The combined value is kept internally; the inputs are not validated.
    pub fn step(&mut self, input_a: f64, input_b: f64) -> i32 {
        self.last_current = input_a + input_b - 2.0 * input_a * input_b;
        if self.last_current > self.threshold {
            1
        } else {
            0
        }
    }

    /// Clear the stored combined value; the threshold is kept.
    pub fn reset(&mut self) {
        self.last_current = 0.0;
    }
}

#[cfg(test)]
#[path = "dendritic_neuron_tests.rs"]
mod tests;
