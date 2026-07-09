// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! GPU-accelerated DenseLayer using wgpu compute shaders.
//!
//! Wraps the CPU [`crate::layer::DenseLayer`] and provides GPU-accelerated
//! forward passes via Bernoulli-encode and AND+popcount compute kernels.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu;

use super::buffers;
use super::context::GpuContext;
use crate::layer::DenseLayer;

// ---- Uniform structs (must match WGSL layout exactly) ----

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EncodeParams {
    n_inputs: u32,
    words_per_input: u32,
    seed_lo: u32,
    seed_hi: u32,
    n_samples: u32,
    length: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AccumParams {
    n_inputs: u32,
    n_neurons: u32,
    words_per_input: u32,
    inv_length: f32,
    n_samples: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-accelerated stochastic computing dense layer.
pub struct GpuDenseLayer {
    /// CPU fallback — always available.
    pub cpu: DenseLayer,
    ctx: Arc<GpuContext>,

    // Persistent GPU buffers.
    weight_buf: wgpu::Buffer,

    // Per-call buffers (pre-allocated for max_batch_size).
    input_prob_buf: wgpu::Buffer,
    packed_input_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    output_staging: wgpu::Buffer,
    encode_uniform_buf: wgpu::Buffer,
    accum_uniform_buf: wgpu::Buffer,

    max_batch_size: usize,
}

impl GpuDenseLayer {
    /// Create a GPU-accelerated dense layer.
    ///
    /// Returns `None` if no GPU is available.
    pub fn try_new(
        n_inputs: usize,
        n_neurons: usize,
        length: usize,
        seed: u64,
        max_batch: usize,
    ) -> Option<Self> {
        let ctx = super::context::get_context()?;
        let cpu = DenseLayer::new(n_inputs, n_neurons, length, seed);
        let words = length.div_ceil(64);
        let dev = &ctx.device;

        // Upload weights as vec2<u32> (reinterpret u64 → 2×u32).
        let weight_bytes: &[u8] = bytemuck::cast_slice(cpu.packed_weights_flat());
        let weight_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weights"),
            size: weight_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&weight_buf, 0, weight_bytes);

        // Pre-allocate per-call buffers for max_batch.
        let input_prob_size = (max_batch * n_inputs * 4) as u64; // f32
        let packed_size = (max_batch * n_inputs * words * 8) as u64; // vec2<u32>
        let output_size = (max_batch * n_neurons * 4) as u64; // f32

        let input_prob_buf = buffers::storage_buffer(dev, "input_probs", input_prob_size, true);
        let packed_input_buf = buffers::storage_buffer(dev, "packed_inputs", packed_size, false);
        let output_buf = buffers::storage_buffer(dev, "output", output_size, false);
        let output_staging = buffers::staging_buffer(dev, "output_staging", output_size);
        let encode_uniform_buf = buffers::uniform_buffer(dev, "encode_params", 32);
        let accum_uniform_buf = buffers::uniform_buffer(dev, "accum_params", 32);

        Some(GpuDenseLayer {
            cpu,
            ctx,
            weight_buf,
            input_prob_buf,
            packed_input_buf,
            output_buf,
            output_staging,
            encode_uniform_buf,
            accum_uniform_buf,
            max_batch_size: max_batch,
        })
    }

    /// GPU forward pass for a single sample.
    pub fn forward_gpu(&self, inputs: &[f64], seed: u64) -> Vec<f64> {
        self.forward_batch_gpu(inputs, 1, seed)
    }

    /// GPU forward pass for a batch of samples.
    ///
    /// `inputs_flat` is row-major `[n_samples × n_inputs]`.
    /// Returns `[n_samples × n_neurons]` as `Vec<f64>`.
    pub fn forward_batch_gpu(&self, inputs_flat: &[f64], n_samples: usize, seed: u64) -> Vec<f64> {
        let n_inputs = self.cpu.n_inputs;
        let n_neurons = self.cpu.n_neurons;
        let words = self.cpu.words_per_input;
        let length = self.cpu.length;
        assert_eq!(inputs_flat.len(), n_samples * n_inputs);
        assert!(
            n_samples <= self.max_batch_size,
            "Batch size {} exceeds max {}",
            n_samples,
            self.max_batch_size
        );

        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;

        // Upload input probabilities as f32.
        let inputs_f32: Vec<f32> = inputs_flat.iter().map(|&x| x as f32).collect();
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

        // Submit and wait.
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.output_staging.slice(..out_bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = dev.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let data = slice
            .get_mapped_range()
            .expect("GPU output staging buffer is mapped for readback");
        let output_f32: &[f32] = bytemuck::cast_slice(&data);
        let result: Vec<f64> = output_f32[..n_samples * n_neurons]
            .iter()
            .map(|&x| x as f64)
            .collect();
        drop(data);
        self.output_staging.unmap();

        result
    }

    /// Name of the GPU adapter (e.g. "AMD Radeon RX 6600 XT").
    pub fn gpu_name(&self) -> &str {
        &self.ctx.adapter_name
    }
}
