// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! GPU-accelerated Izhikevich neuron batch runner using a wgpu compute shader.
//!
//! Mirrors the floating-point CPU [`crate::neuron::Izhikevich`]: `n_neurons`
//! independent neurons sharing the `(a, b, c, d, dt)` parameters, each driven by
//! its own constant current, advanced for `n_steps`. One GPU thread owns one
//! neuron and loops the steps internally (no per-step dispatch overhead),
//! producing the same `[n_neurons × n_steps]` spike (i32) and voltage (f32)
//! layout as the CPU model. Each step applies the two half-steps of the Euler
//! update (for stability on the `0.04·v²` term) then the `v >= v_peak` threshold
//! with reset `v ← c`, `u ← u + d`. WGSL has no f64, so the math is f32 and
//! agreement with the f64 CPU oracle is tolerance-based, not bit-exact — unlike
//! the fixed-point LIF kernel.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu;

use super::buffers;
use super::context::GpuContext;

/// Uniform parameter block — layout must match `IzhParams` in `izhikevich_step.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct IzhParams {
    n_neurons: u32,
    n_steps: u32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    dt: f32,
    v_peak: f32,
}

/// One batch run's result: row-major `[n_neurons × n_steps]` spikes and voltages.
pub struct IzhBatchResult {
    pub spikes: Vec<i32>,
    pub voltages: Vec<f32>,
}

/// GPU-accelerated Izhikevich neuron batch runner.
pub struct GpuIzhikevichBatch {
    ctx: Arc<GpuContext>,
}

impl GpuIzhikevichBatch {
    /// Acquire the shared GPU context, or `None` if no GPU is available.
    pub fn try_new() -> Option<Self> {
        let ctx = super::context::get_context()?;
        Some(GpuIzhikevichBatch { ctx })
    }

    /// Run `n_neurons` Izhikevich neurons for `n_steps` with per-neuron constant
    /// currents.
    ///
    /// `currents` has length `n_neurons`; every neuron shares `(a, b, c, d, dt)`
    /// and starts at `v = c`, `u = b·c` (the CPU model's initial state). Returns
    /// row-major `[n_neurons × n_steps]` spikes and voltages, agreeing with the
    /// CPU reference within f32 tolerance.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        n_neurons: usize,
        n_steps: usize,
        currents: &[f32],
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        dt: f32,
        v_peak: f32,
    ) -> IzhBatchResult {
        assert_eq!(
            currents.len(),
            n_neurons,
            "currents length {} does not match n_neurons {}",
            currents.len(),
            n_neurons
        );

        let total = n_neurons * n_steps;
        if n_neurons == 0 || n_steps == 0 {
            return IzhBatchResult {
                spikes: vec![0; total],
                voltages: vec![0.0; total],
            };
        }

        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;

        let currents_bytes: &[u8] = bytemuck::cast_slice(currents);
        let currents_buf =
            buffers::storage_buffer(dev, "izh_currents", currents_bytes.len() as u64, true);
        queue.write_buffer(&currents_buf, 0, currents_bytes);

        let out_size = (total * 4) as u64; // i32 / f32 per element
        let spikes_buf = buffers::storage_buffer(dev, "izh_spikes", out_size, false);
        let voltages_buf = buffers::storage_buffer(dev, "izh_voltages", out_size, false);
        let spikes_staging = buffers::staging_buffer(dev, "izh_spikes_staging", out_size);
        let voltages_staging = buffers::staging_buffer(dev, "izh_voltages_staging", out_size);
        let uniform_buf = buffers::uniform_buffer(dev, "izh_params", 32);

        let params = IzhParams {
            n_neurons: n_neurons as u32,
            n_steps: n_steps as u32,
            a,
            b,
            c,
            d,
            dt,
            v_peak,
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&params));

        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("izh_bg"),
            layout: &self.ctx.izhikevich_bind_group_layout,
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
            label: Some("izh_batch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("izhikevich_step"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ctx.izhikevich_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let x_groups = (n_neurons as u32).div_ceil(64);
            pass.dispatch_workgroups(x_groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&spikes_buf, 0, &spikes_staging, 0, out_size);
        encoder.copy_buffer_to_buffer(&voltages_buf, 0, &voltages_staging, 0, out_size);

        queue.submit(std::iter::once(encoder.finish()));

        let spikes = read_i32(dev, &spikes_staging, out_size, total);
        let voltages = read_f32(dev, &voltages_staging, out_size, total);

        IzhBatchResult { spikes, voltages }
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

/// Map a staging buffer, copy out `count` f32 values, and unmap.
fn read_f32(dev: &wgpu::Device, staging: &wgpu::Buffer, size: u64, count: usize) -> Vec<f32> {
    let slice = staging.slice(..size);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = dev.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let data = slice.get_mapped_range();
    let values: &[f32] = bytemuck::cast_slice(&data);
    let out = values[..count].to_vec();
    drop(data);
    staging.unmap();
    out
}
