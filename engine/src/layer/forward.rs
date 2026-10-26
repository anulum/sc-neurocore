// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fallible dense forward kernels

//! Dense inference with reserved outputs and unchanged Bernoulli protocols.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use super::allocation::{product, zeroed};
use super::{DenseLayer, DenseLayerError, RAYON_ENCODE_THRESHOLD, RAYON_NEURON_THRESHOLD};
use crate::{bitstream, simd};

/// Feature values admitted against the same immutably borrowed layer.
pub(crate) struct ForwardInputs<'a> {
    pub(super) layer: &'a DenseLayer,
    pub(super) inputs: &'a [f64],
}

impl DenseLayer {
    /// Compute stochastic activations using one sequential scalar RNG.
    pub fn forward(&self, inputs: &[f64], seed: u64) -> Result<Vec<f64>, String> {
        self.try_forward(inputs, seed)
            .map_err(|error| error.to_string())
    }

    /// Reserve packed inputs and outputs before scalar Bernoulli inference.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` for an incorrect input count or `Allocation` when
    /// packed input or activation storage cannot be reserved.
    pub fn try_forward(&self, inputs: &[f64], seed: u64) -> Result<Vec<f64>, DenseLayerError> {
        self.validate_inputs(inputs)?;
        let words = self.words_per_input;
        let mut packed = zeroed(product(self.n_inputs, words)?)?;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        for (&probability, destination) in inputs.iter().zip(packed.chunks_mut(words)) {
            bitstream::bernoulli_packed_into(probability, self.length, &mut rng, destination)
                .map_err(|message| DenseLayerError::Invalid(message.into()))?;
        }
        self.compute_packed(&packed)
    }

    /// Compute activations using independent wrapping input-index seeds.
    pub fn forward_fast(&self, inputs: &[f64], seed: u64) -> Result<Vec<f64>, String> {
        self.try_forward_fast(inputs, seed)
            .map_err(|error| error.to_string())
    }

    /// Encode SIMD input streams directly into fallibly reserved storage.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` for an incorrect input count or `Allocation` when
    /// packed input or activation storage cannot be reserved.
    pub fn try_forward_fast(&self, inputs: &[f64], seed: u64) -> Result<Vec<f64>, DenseLayerError> {
        let packed = self.validate_inputs(inputs)?.encode_fast(seed)?;
        self.compute_packed(&packed)
    }

    /// Fuse independent input encoding with AND/popcount accumulation.
    pub fn forward_fused(&self, inputs: &[f64], seed: u64) -> Result<Vec<f64>, String> {
        self.try_forward_fused(inputs, seed)
            .map_err(|error| error.to_string())
    }

    /// Fuse encoding into reserved output without materializing input streams.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` for an incorrect input count or `Allocation` when
    /// the activation vector cannot be reserved.
    pub fn try_forward_fused(
        &self,
        inputs: &[f64],
        seed: u64,
    ) -> Result<Vec<f64>, DenseLayerError> {
        self.validate_inputs(inputs)?;
        self.compute_outputs(self.n_neurons >= RAYON_NEURON_THRESHOLD, |neuron| {
            let total: u64 = inputs
                .iter()
                .enumerate()
                .map(|(input, &probability)| {
                    let mut rng =
                        Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(input as u64));
                    bitstream::encode_and_popcount(
                        self.weight_slice(neuron, input),
                        probability,
                        self.length,
                        &mut rng,
                    )
                })
                .sum();
            total as f64 * self.inv_length
        })
    }

    /// Preserve the fast-forward kernel used by NumPy bindings.
    pub fn forward_numpy_inner(&self, inputs: &[f64], seed: u64) -> Result<Vec<f64>, String> {
        self.forward_fast(inputs, seed)
    }

    /// Borrow admitted features without allowing layer dimensions to change.
    pub(crate) fn validate_inputs<'a>(
        &'a self,
        inputs: &'a [f64],
    ) -> Result<ForwardInputs<'a>, DenseLayerError> {
        if inputs.len() != self.n_inputs {
            return Err(DenseLayerError::Invalid(format!(
                "Expected input of length {}, got {}.",
                self.n_inputs,
                inputs.len()
            )));
        }
        Ok(ForwardInputs {
            layer: self,
            inputs,
        })
    }

    /// Check one activation slot per neuron before any output write.
    pub(super) fn validate_output(&self, output: &[f64]) -> Result<(), DenseLayerError> {
        if output.len() != self.n_neurons {
            return Err(DenseLayerError::Invalid(format!(
                "Expected output length {}, got {}.",
                self.n_neurons,
                output.len(),
            )));
        }
        Ok(())
    }

    /// Accumulate one activation per packed weight row.
    fn compute_packed(&self, inputs: &[u64]) -> Result<Vec<f64>, DenseLayerError> {
        let row_words = inputs.len();
        self.compute_outputs(self.n_neurons >= RAYON_NEURON_THRESHOLD, |neuron| {
            let start = neuron * row_words;
            let total = simd::fused_and_popcount_dispatch(
                &self.packed_weights_flat[start..start + row_words],
                inputs,
            );
            total as f64 * self.inv_length
        })
    }

    /// Write into a fully reserved activation vector without Rayon collection growth.
    pub(super) fn compute_outputs(
        &self,
        parallel: bool,
        compute: impl Fn(usize) -> f64 + Sync + Send,
    ) -> Result<Vec<f64>, DenseLayerError> {
        let mut output = zeroed(self.n_neurons)?;
        self.fill_outputs(parallel, &mut output, compute);
        Ok(output)
    }

    /// Fill validated activation storage without reserving a second output vector.
    pub(super) fn fill_outputs(
        &self,
        parallel: bool,
        output: &mut [f64],
        compute: impl Fn(usize) -> f64 + Sync + Send,
    ) {
        if parallel {
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, value)| *value = compute(index));
        } else {
            for (index, value) in output.iter_mut().enumerate() {
                *value = compute(index);
            }
        }
    }
}

impl ForwardInputs<'_> {
    /// Fill owned output after admission, without reserving a second output.
    pub(crate) fn try_forward_fast_into(
        &self,
        seed: u64,
        output: &mut [f64],
    ) -> Result<(), DenseLayerError> {
        self.layer.validate_output(output)?;
        let packed = self.encode_fast(seed)?;
        let row_words = packed.len();
        self.layer.fill_outputs(
            self.layer.n_neurons >= RAYON_NEURON_THRESHOLD,
            output,
            |neuron| {
                let start = neuron * row_words;
                let total = simd::fused_and_popcount_dispatch(
                    &self.layer.packed_weights_flat[start..start + row_words],
                    &packed,
                );
                total as f64 * self.layer.inv_length
            },
        );
        Ok(())
    }

    /// Preserve independent SIMD RNG draws in a fallibly reserved packed buffer.
    fn encode_fast(&self, seed: u64) -> Result<Vec<u64>, DenseLayerError> {
        let layer = self.layer;
        let mut packed = zeroed(product(layer.n_inputs, layer.words_per_input)?)?;
        let encode = |(index, destination): (usize, &mut [u64])| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(index as u64));
            bitstream::bernoulli_packed_simd_into(
                self.inputs[index],
                layer.length,
                &mut rng,
                destination,
            )
            .map_err(|message| DenseLayerError::Invalid(message.into()))
        };
        if layer.n_inputs >= RAYON_ENCODE_THRESHOLD {
            packed
                .par_chunks_mut(layer.words_per_input)
                .enumerate()
                .try_for_each(encode)?;
        } else {
            packed
                .chunks_mut(layer.words_per_input)
                .enumerate()
                .try_for_each(encode)?;
        }
        Ok(packed)
    }
}
