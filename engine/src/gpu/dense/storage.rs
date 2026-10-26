// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GPU dense storage admission

//! Checked shader dimensions and fallible host/device buffer ownership.

use super::{GpuDenseError, GpuDenseLayer};
use crate::gpu::{buffers, context};
use crate::layer::{DenseLayer, DenseLayerError};
use std::sync::Mutex;

/// Multiply byte counts before requesting host or device storage.
fn bytes(factors: &[usize], element_size: usize) -> Result<u64, GpuDenseError> {
    let count = factors.iter().try_fold(element_size, |count, &factor| {
        count.checked_mul(factor).ok_or(DenseLayerError::Allocation)
    })?;
    Ok(count as u64)
}

impl GpuDenseLayer {
    /// Create checked GPU storage while retaining ordinary optional availability.
    ///
    /// # Errors
    /// Rejects zero dimensions, overflowing storage, unsupported shader/device
    /// dimensions and host/device allocation refusal. Returns `Ok(None)` only
    /// when no GPU context is available after dimension validation.
    pub fn try_new_checked(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        max_batch: usize,
    ) -> Result<Option<Self>, GpuDenseError> {
        for (value, message) in [
            (length, "bitstream length must be > 0"),
            (n_inputs, "n_inputs must be > 0"),
            (n_neurons, "n_neurons must be > 0"),
            (max_batch, "max_batch must be > 0"),
        ] {
            if value == 0 {
                return Err(DenseLayerError::Invalid(message.into()).into());
            }
        }
        let words = length.div_ceil(64);
        let weight_size = bytes(&[n_neurons, n_inputs, words], 8)?;
        let input_prob_size = bytes(&[max_batch, n_inputs], 4)?;
        let packed_size = bytes(&[max_batch, n_inputs, words], 8)?;
        let output_size = bytes(&[max_batch, n_neurons], 4)?;
        for count in [
            length as u64,
            weight_size / 8,
            packed_size / 8,
            output_size / 4,
        ] {
            if count > u32::MAX as u64 {
                return Err(DenseLayerError::Invalid(
                    "GPU dense dimensions exceed u32 shader indexing".into(),
                )
                .into());
            }
        }
        let Some(ctx) = context::get_context() else {
            return Ok(None);
        };
        let dev = &ctx.device;
        let limits = dev.limits();
        for size in [weight_size, input_prob_size, packed_size, output_size] {
            if size > limits.max_buffer_size || size > limits.max_storage_buffer_binding_size {
                return Err(DenseLayerError::Invalid(
                    "GPU dense storage exceeds device buffer limits".into(),
                )
                .into());
            }
        }
        let groups = (n_inputs * words).div_ceil(256);
        if groups > limits.max_compute_workgroups_per_dimension as usize
            || n_neurons > limits.max_compute_workgroups_per_dimension as usize
            || max_batch > limits.max_compute_workgroups_per_dimension as usize
        {
            return Err(DenseLayerError::Invalid(
                "GPU dense dimensions exceed device dispatch limits".into(),
            )
            .into());
        }
        let cpu = DenseLayer::try_new(n_inputs, n_neurons, length, seed)?;
        let allocation_scope = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation_scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let weight_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weights"),
            size: weight_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(
            &weight_buf,
            0,
            bytemuck::cast_slice(cpu.packed_weights_flat()),
        );
        let layer = Self {
            cpu,
            weight_buf,
            input_prob_buf: buffers::storage_buffer(dev, "input_probs", input_prob_size, true),
            packed_input_buf: buffers::storage_buffer(dev, "packed_inputs", packed_size, false),
            output_buf: buffers::storage_buffer(dev, "output", output_size, false),
            output_staging: buffers::staging_buffer(dev, "output_staging", output_size),
            encode_uniform_buf: buffers::uniform_buffer(dev, "encode_params", 32),
            accum_uniform_buf: buffers::uniform_buffer(dev, "accum_params", 32),
            max_batch_size: max_batch,
            call_lock: Mutex::new(()),
            dimensions: (n_inputs, n_neurons, length, words),
            ctx,
        };
        let validation = pollster::block_on(validation_scope.pop());
        let allocation = pollster::block_on(allocation_scope.pop());
        if allocation.is_some() {
            return Err(DenseLayerError::Allocation.into());
        }
        if let Some(error) = validation {
            return Err(GpuDenseError::Backend(error.to_string()));
        }
        Ok(Some(layer))
    }
}
