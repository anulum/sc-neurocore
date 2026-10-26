// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fallible GPU dense inference

//! Shape admission, owned host output and serialized GPU readback.

use super::{AccumParams, EncodeParams, GpuDenseError, GpuDenseLayer};
use crate::layer::DenseLayerError;

impl GpuDenseLayer {
    /// Return one Philox-encoded sample without publishing partial results.
    ///
    /// # Errors
    /// Returns typed input/storage errors or a backend submission/readback error.
    pub fn try_forward_gpu(&self, inputs: &[f64], seed: u64) -> Result<Vec<f64>, GpuDenseError> {
        self.try_forward_batch_gpu(inputs, 1, seed)
    }

    /// Compute row-major samples after shape and fallible host-storage admission.
    ///
    /// # Errors
    /// Rejects shape overflow/mismatch, batches exceeding construction capacity,
    /// host/device allocation refusal, or failed GPU readback. Packed weights
    /// remain unchanged and every mapped staging buffer is released on refusal.
    pub fn try_forward_batch_gpu(
        &self,
        inputs_flat: &[f64],
        n_samples: usize,
        seed: u64,
    ) -> Result<Vec<f64>, GpuDenseError> {
        let n_inputs = self.cpu.n_inputs;
        let n_neurons = self.cpu.n_neurons;
        let words = self.cpu.words_per_input;
        let length = self.cpu.length;
        if (n_inputs, n_neurons, length, words) != self.dimensions {
            return Err(DenseLayerError::Invalid(
                "GPU layer dimensions changed after construction".into(),
            )
            .into());
        }
        let expected = n_samples.checked_mul(n_inputs).ok_or_else(|| {
            DenseLayerError::Invalid(
                "Input size overflow when validating n_samples * n_inputs.".into(),
            )
        })?;
        if inputs_flat.len() != expected {
            return Err(DenseLayerError::Invalid(format!(
                "Expected {} values ({}×{}), got {}.",
                expected,
                n_samples,
                n_inputs,
                inputs_flat.len()
            ))
            .into());
        }
        if n_samples > self.max_batch_size {
            return Err(DenseLayerError::Invalid(format!(
                "Batch size {} exceeds max {}",
                n_samples, self.max_batch_size
            ))
            .into());
        }
        if n_samples == 0 {
            return Ok(Vec::new());
        }
        // Construction bounds max_batch * neurons; the unchanged geometry and
        // validated sample capacity preserve that bound for this result.
        let count = n_samples * n_neurons;
        let mut inputs_f32 = Vec::new();
        inputs_f32.try_reserve_exact(inputs_flat.len())?;
        inputs_f32.extend(inputs_flat.iter().map(|&value| value as f32));
        let mut result = Vec::new();
        result.try_reserve_exact(count)?;
        let _guard = self
            .call_lock
            .lock()
            .map_err(|error| GpuDenseError::Backend(format!("GPU dense call lock: {error}")))?;
        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;
        let allocation_scope = dev.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation_scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let completion = (|| {
            queue.write_buffer(&self.input_prob_buf, 0, bytemuck::cast_slice(&inputs_f32));
            // Write encode uniform params.
            let encode_params = EncodeParams {
                n_inputs: n_inputs as u32,
                words_per_input: words as u32,
                seed_lo: seed as u32,
                seed_hi: (seed >> 32) as u32,
                n_samples: n_samples as u32,
                length: length as u32,
                _pad0: 0,
                _pad1: 0,
            };
            queue.write_buffer(
                &self.encode_uniform_buf,
                0,
                bytemuck::bytes_of(&encode_params),
            );

            // Write accumulate uniform params.
            let accum_params = AccumParams {
                n_inputs: n_inputs as u32,
                n_neurons: n_neurons as u32,
                words_per_input: words as u32,
                inv_length: self.cpu.inv_length as f32,
                n_samples: n_samples as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            queue.write_buffer(
                &self.accum_uniform_buf,
                0,
                bytemuck::bytes_of(&accum_params),
            );

            // Create bind groups.
            let encode_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("encode_bg"),
                layout: &self.ctx.encode_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.input_prob_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.packed_input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.encode_uniform_buf.as_entire_binding(),
                    },
                ],
            });

            let accum_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("accum_bg"),
                layout: &self.ctx.accumulate_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.weight_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.packed_input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.accum_uniform_buf.as_entire_binding(),
                    },
                ],
            });

            // Encode + accumulate in a single command buffer.
            let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dense_forward"),
            });

            // Dispatch encode kernel.
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("encode"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.ctx.encode_pipeline);
                pass.set_bind_group(0, &encode_bg, &[]);
                let x_groups = ((n_inputs * words) as u32).div_ceil(256);
                pass.dispatch_workgroups(x_groups, n_samples as u32, 1);
            }

            // Dispatch accumulate kernel.
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("accumulate"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.ctx.accumulate_pipeline);
                pass.set_bind_group(0, &accum_bg, &[]);
                pass.dispatch_workgroups(n_neurons as u32, n_samples as u32, 1);
            }

            // Copy output to staging buffer for readback.
            let out_bytes = (n_samples * n_neurons * 4) as u64;
            encoder.copy_buffer_to_buffer(&self.output_buf, 0, &self.output_staging, 0, out_bytes);

            queue.submit(std::iter::once(encoder.finish()));
            let slice = self.output_staging.slice(..out_bytes);
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            slice.map_async(wgpu::MapMode::Read, move |outcome| {
                let _ = sender.send(outcome);
            });
            dev.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|error| GpuDenseError::Backend(format!("GPU dense poll: {error}")))?;
            receiver
                .recv()
                .map_err(|error| {
                    GpuDenseError::Backend(format!("GPU dense mapping callback: {error}"))
                })?
                .map_err(|error| GpuDenseError::Backend(format!("GPU dense mapping: {error}")))?;
            let data = slice.get_mapped_range().map_err(|error| {
                GpuDenseError::Backend(format!("GPU dense mapped range: {error}"))
            })?;
            let output_f32: &[f32] = bytemuck::cast_slice(&data);
            result.extend(output_f32[..count].iter().map(|&value| value as f64));
            Ok(result)
        })();
        self.output_staging.unmap();
        let validation = pollster::block_on(validation_scope.pop());
        let allocation = pollster::block_on(allocation_scope.pop());
        if allocation.is_some() {
            return Err(DenseLayerError::Allocation.into());
        }
        if let Some(error) = validation {
            return Err(GpuDenseError::Backend(error.to_string()));
        }
        completion
    }
}
