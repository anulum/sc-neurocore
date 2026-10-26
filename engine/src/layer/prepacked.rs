// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense inference from packed input rows

//! Shape admission and accumulation for owned and borrowed packed input rows.

use super::allocation::product;
use super::{DenseLayer, DenseLayerError};
use crate::simd;

/// Packed rows admitted against the same immutably borrowed layer.
pub(crate) struct PrepackedInputs<'a> {
    layer: &'a DenseLayer,
    inputs: &'a [u64],
    words: usize,
}

impl DenseLayer {
    /// Compute activations from independently owned packed input rows.
    pub fn forward_prepacked(&self, inputs: &[Vec<u64>]) -> Result<Vec<f64>, String> {
        self.try_forward_prepacked(inputs)
            .map_err(|error| error.to_string())
    }

    /// Validate every packed row before reserving the activation vector.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` for incorrect row or word counts, or `Allocation`
    /// when activation storage cannot be reserved.
    pub fn try_forward_prepacked(&self, inputs: &[Vec<u64>]) -> Result<Vec<f64>, DenseLayerError> {
        if inputs.len() != self.n_inputs {
            return Err(DenseLayerError::Invalid(format!(
                "Expected {} packed inputs, got {}.",
                self.n_inputs,
                inputs.len()
            )));
        }
        let words = self.length.div_ceil(64);
        for (index, row) in inputs.iter().enumerate() {
            if row.len() != words {
                return Err(DenseLayerError::Invalid(format!(
                    "Packed input {} has {} words, expected {}.",
                    index,
                    row.len(),
                    words
                )));
            }
        }
        self.compute_outputs(true, |neuron| {
            let total: u64 = inputs
                .iter()
                .enumerate()
                .map(|(input, words)| {
                    simd::fused_and_popcount_dispatch(self.weight_slice(neuron, input), words)
                })
                .sum();
            total as f64 * self.inv_length
        })
    }

    /// Compute activations from contiguous `[input][word]` packed storage.
    pub fn forward_prepacked_2d(
        &self,
        inputs: &[u64],
        n_inputs: usize,
        words: usize,
    ) -> Result<Vec<f64>, String> {
        self.try_forward_prepacked_2d(inputs, n_inputs, words)
            .map_err(|error| error.to_string())
    }

    /// Validate packed dimensions with checked multiplication before inference.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` for mismatched dimensions or buffer length, or
    /// `Allocation` for an overflowing size or unavailable activation storage.
    pub fn try_forward_prepacked_2d(
        &self,
        inputs: &[u64],
        n_inputs: usize,
        words: usize,
    ) -> Result<Vec<f64>, DenseLayerError> {
        let admitted = self.validate_prepacked_2d(inputs, n_inputs, words)?;
        self.compute_outputs(true, |neuron| {
            admitted
                .layer
                .prepacked_activation(admitted.inputs, admitted.words, neuron)
        })
    }

    /// Validate borrowed packed dimensions before Python output allocation.
    pub(crate) fn validate_prepacked_2d<'a>(
        &'a self,
        inputs: &'a [u64],
        n_inputs: usize,
        words: usize,
    ) -> Result<PrepackedInputs<'a>, DenseLayerError> {
        if n_inputs != self.n_inputs {
            return Err(DenseLayerError::Invalid(format!(
                "Expected {} packed inputs, got {}.",
                self.n_inputs, n_inputs
            )));
        }
        let expected_words = self.length.div_ceil(64);
        if words != expected_words {
            return Err(DenseLayerError::Invalid(format!(
                "Expected {} words per input, got {}.",
                expected_words, words
            )));
        }
        if inputs.len() != product(n_inputs, words)? {
            return Err(DenseLayerError::Invalid(format!(
                "Flat buffer length {} != n_inputs({}) * words({}).",
                inputs.len(),
                n_inputs,
                words
            )));
        }
        Ok(PrepackedInputs {
            layer: self,
            inputs,
            words,
        })
    }

    /// Accumulate one neuron's borrowed packed rows with unchanged summation order.
    fn prepacked_activation(&self, inputs: &[u64], words: usize, neuron: usize) -> f64 {
        let total: u64 = (0..self.n_inputs)
            .map(|input| {
                let start = input * words;
                simd::fused_and_popcount_dispatch(
                    self.weight_slice(neuron, input),
                    &inputs[start..start + words],
                )
            })
            .sum();
        total as f64 * self.inv_length
    }
}

impl PrepackedInputs<'_> {
    /// Fill owned output using the previously admitted packed shape.
    pub(crate) fn try_forward_into(&self, output: &mut [f64]) -> Result<(), DenseLayerError> {
        self.layer.validate_output(output)?;
        self.layer.fill_outputs(true, output, |neuron| {
            self.layer
                .prepacked_activation(self.inputs, self.words, neuron)
        });
        Ok(())
    }
}
