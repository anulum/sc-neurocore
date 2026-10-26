// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Failure-atomic dense batch inference

//! Dense sample seeds, checked batch shapes and unpublished result ownership.

use super::allocation::zeroed;
use super::forward::ForwardInputs;
use super::{DenseLayer, DenseLayerError};
use rayon::prelude::*;

impl DenseLayer {
    /// Fill an existing batch output only after the entire inference succeeds.
    pub fn forward_batch_into(
        &self,
        inputs: &[f64],
        samples: usize,
        seed: u64,
        output: &mut [f64],
    ) -> Result<(), String> {
        self.try_forward_batch_into(inputs, samples, seed, output)
            .map_err(|error| error.to_string())
    }

    /// Preserve caller-owned output on shape or allocation refusal.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` for incorrect input/output dimensions, or `Allocation`
    /// when any sample or the private complete result cannot be reserved.
    pub fn try_forward_batch_into(
        &self,
        inputs: &[f64],
        samples: usize,
        seed: u64,
        output: &mut [f64],
    ) -> Result<(), DenseLayerError> {
        self.validate_batch_buffers(inputs, samples, output)?;
        let completed = self.try_forward_batch(inputs, samples, seed)?;
        output.copy_from_slice(&completed);
        Ok(())
    }

    /// Return flat row-major `[sample][neuron]` batch activations.
    pub fn forward_batch(
        &self,
        inputs: &[f64],
        samples: usize,
        seed: u64,
    ) -> Result<Vec<f64>, String> {
        self.try_forward_batch(inputs, samples, seed)
            .map_err(|error| error.to_string())
    }

    /// Retain per-sample wrapping seeds and propagate every sample refusal.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` for an overflowing output count or wrong input shape,
    /// or `Allocation` when private batch or sample storage cannot be reserved.
    pub fn try_forward_batch(
        &self,
        inputs: &[f64],
        samples: usize,
        seed: u64,
    ) -> Result<Vec<f64>, DenseLayerError> {
        let count = samples.checked_mul(self.n_neurons).ok_or_else(|| {
            DenseLayerError::Invalid(
                "Output size overflow when allocating n_samples * n_neurons.".into(),
            )
        })?;
        let mut output = zeroed(count)?;
        self.try_forward_batch_unpublished_into(inputs, samples, seed, &mut output)?;
        Ok(output)
    }

    /// Fill exclusively owned batch storage, propagating every sample refusal.
    ///
    /// This buffer must remain unpublished until success. On refusal a caller
    /// must discard it because earlier sample rows may have completed. Public
    /// caller-owned output uses the atomic `try_forward_batch_into` instead.
    pub(crate) fn try_forward_batch_unpublished_into(
        &self,
        inputs: &[f64],
        samples: usize,
        seed: u64,
        output: &mut [f64],
    ) -> Result<(), DenseLayerError> {
        self.validate_batch_buffers(inputs, samples, output)?;
        output
            .par_chunks_mut(self.n_neurons)
            .enumerate()
            .try_for_each(|(sample, row)| {
                let start = sample * self.n_inputs;
                let sample_seed = seed.wrapping_add((sample as u64).wrapping_mul(1_000_000));
                let inputs = ForwardInputs {
                    layer: self,
                    inputs: &inputs[start..start + self.n_inputs],
                };
                inputs.try_forward_fast_into(sample_seed, row)
            })?;
        Ok(())
    }

    /// Validate row-major batch input counts without multiplication overflow.
    fn validate_batch_inputs(&self, inputs: &[f64], samples: usize) -> Result<(), DenseLayerError> {
        let expected = samples.checked_mul(self.n_inputs).ok_or_else(|| {
            DenseLayerError::Invalid(
                "Input size overflow when validating n_samples * n_inputs.".into(),
            )
        })?;
        if inputs.len() != expected {
            return Err(DenseLayerError::Invalid(format!(
                "Expected {} values ({}×{}), got {}.",
                expected,
                samples,
                self.n_inputs,
                inputs.len()
            )));
        }
        Ok(())
    }

    /// Validate both batch shapes before a caller-owned buffer can change.
    fn validate_batch_buffers(
        &self,
        inputs: &[f64],
        samples: usize,
        output: &[f64],
    ) -> Result<(), DenseLayerError> {
        self.validate_batch_inputs(inputs, samples)?;
        let expected = samples.checked_mul(self.n_neurons).ok_or_else(|| {
            DenseLayerError::Invalid(
                "Output size overflow when validating n_samples * n_neurons.".into(),
            )
        })?;
        if output.len() != expected {
            return Err(DenseLayerError::Invalid(format!(
                "Expected output length {} ({}×{}), got {}.",
                expected,
                samples,
                self.n_neurons,
                output.len(),
            )));
        }
        Ok(())
    }
}
