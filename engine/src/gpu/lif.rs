// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! GPU-accelerated fixed-point LIF batch runner using a wgpu compute shader.
//!
//! Mirrors the CPU [`crate::lib_batch_lif_run_multi`] contract: `n_neurons`
//! independent fixed-point LIF neurons, each driven by its own constant current,
//! advanced for `n_steps` with zero noise. One GPU thread owns one neuron and
//! loops the steps internally (no per-step dispatch overhead), producing the same
//! `[n_neurons × n_steps]` spike (i32) and voltage (i16-range) layout as the CPU.
//! The integer arithmetic is bit-exact with `FixedPointLif::step`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu;

use super::buffers;
use super::context::GpuContext;

/// Uniform parameter block — layout must match `LifParams` in `lif_step.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LifParams {
    n_neurons: u32,
    n_steps: u32,
    data_width: u32,
    fraction: u32,
    leak_k: i32,
    gain_k: i32,
    v_rest: i32,
    v_reset: i32,
    v_threshold: i32,
    refractory_period: i32,
    noise: i32,
    _pad: u32,
}

/// One batch run's result: row-major `[n_neurons × n_steps]` spikes and voltages.
pub struct LifBatchResult {
    pub spikes: Vec<i32>,
    pub voltages: Vec<i32>,
}

/// GPU-accelerated fixed-point LIF batch runner.
pub struct GpuLifBatch {
    ctx: Arc<GpuContext>,
}

impl GpuLifBatch {
    /// Acquire the shared GPU context, or `None` if no GPU is available.
    pub fn try_new() -> Option<Self> {
        let ctx = super::context::get_context()?;
        Some(GpuLifBatch { ctx })
    }

    /// Run `n_neurons` LIF neurons for `n_steps` with per-neuron constant currents.
    ///
    /// `currents` has length `n_neurons`. Returns row-major `[n_neurons × n_steps]`
    /// spikes and voltages, bit-exact with the CPU reference.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        n_neurons: usize,
        n_steps: usize,
        leak_k: i16,
        gain_k: i16,
        currents: &[i32],
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        noise: i16,
    ) -> LifBatchResult {
        assert_eq!(
            currents.len(),
            n_neurons,
            "currents length {} does not match n_neurons {}",
            currents.len(),
            n_neurons
        );

        let total = n_neurons * n_steps;
        if n_neurons == 0 || n_steps == 0 {
            return LifBatchResult {
                spikes: vec![0; total],
                voltages: vec![0; total],
            };
        }

        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;

        let currents_bytes: &[u8] = bytemuck::cast_slice(currents);
        let currents_buf =
            buffers::storage_buffer(dev, "lif_currents", currents_bytes.len() as u64, true);
        queue.write_buffer(&currents_buf, 0, currents_bytes);

        let out_size = (total * 4) as u64; // i32 per element
        let spikes_buf = buffers::storage_buffer(dev, "lif_spikes", out_size, false);
        let voltages_buf = buffers::storage_buffer(dev, "lif_voltages", out_size, false);
        let spikes_staging = buffers::staging_buffer(dev, "lif_spikes_staging", out_size);
        let voltages_staging = buffers::staging_buffer(dev, "lif_voltages_staging", out_size);
        let uniform_buf = buffers::uniform_buffer(dev, "lif_params", 48);

        let params = LifParams {
            n_neurons: n_neurons as u32,
            n_steps: n_steps as u32,
            data_width,
            fraction,
            leak_k: leak_k as i32,
            gain_k: gain_k as i32,
            v_rest: v_rest as i32,
            v_reset: v_reset as i32,
            v_threshold: v_threshold as i32,
            refractory_period,
            noise: noise as i32,
            _pad: 0,
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&params));

        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lif_bg"),
            layout: &self.ctx.lif_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: currents_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spikes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: voltages_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lif_batch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lif_step"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ctx.lif_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let x_groups = (n_neurons as u32).div_ceil(64);
            pass.dispatch_workgroups(x_groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&spikes_buf, 0, &spikes_staging, 0, out_size);
        encoder.copy_buffer_to_buffer(&voltages_buf, 0, &voltages_staging, 0, out_size);

        queue.submit(std::iter::once(encoder.finish()));

        let spikes = read_i32(dev, &spikes_staging, out_size, total);
        let voltages = read_i32(dev, &voltages_staging, out_size, total);

        LifBatchResult { spikes, voltages }
    }

    /// Name of the GPU adapter (e.g. "AMD Radeon RX 6600 XT").
    pub fn gpu_name(&self) -> &str {
        &self.ctx.adapter_name
    }
}

/// Map a staging buffer, copy out `count` i32 values, and unmap.
fn read_i32(dev: &wgpu::Device, staging: &wgpu::Buffer, size: u64, count: usize) -> Vec<i32> {
    let slice = staging.slice(..size);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = dev.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let data = slice.get_mapped_range();
    let values: &[i32] = bytemuck::cast_slice(&data);
    let out = values[..count].to_vec();
    drop(data);
    staging.unmap();
    out
}
