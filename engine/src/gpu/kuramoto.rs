// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! GPU-accelerated Kuramoto oscillator integrator using a wgpu compute shader.
//!
//! Mirrors the baseline update of [`crate::scpn::kuramoto::KuramotoSolver`] with
//! zero noise: one GPU thread owns one oscillator and sums its coupling row
//! `K_nm · sin(θ_m − θ_n)` over all `m`, so the O(N²) all-to-all coupling runs
//! fully in parallel. Each dispatch advances one Euler step; the host ping-pongs
//! two phase buffers across `n_steps` dispatches recorded in a single command
//! encoder (wgpu inserts the inter-pass barriers), then reads back the final
//! phase vector. WGSL has no f64, so the math is f32 and agreement with the f64
//! CPU oracle is tolerance-based, not bit-exact.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu;

use super::buffers;
use super::context::GpuContext;

/// Uniform parameter block — layout must match `KuramotoParams` in `kuramoto_step.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct KuramotoParams {
    n_osc: u32,
    _pad: u32,
    dt: f32,
    n_inv: f32,
}

/// GPU-accelerated Kuramoto oscillator integrator.
pub struct GpuKuramoto {
    ctx: Arc<GpuContext>,
}

impl GpuKuramoto {
    /// Acquire the shared GPU context, or `None` if no GPU is available.
    pub fn try_new() -> Option<Self> {
        let ctx = super::context::get_context()?;
        Some(GpuKuramoto { ctx })
    }

    /// Integrate `n_osc` Kuramoto oscillators for `n_steps` Euler steps.
    ///
    /// `omega` (length `n_osc`), `coupling` (row-major `n_osc * n_osc`) and
    /// `initial_phases` (length `n_osc`) describe the system; noise is zero (the
    /// noise-free baseline the CPU oracle also runs for parity). Returns the final
    /// phase vector wrapped to `[0, 2π)`.
    pub fn run(
        &self,
        n_osc: usize,
        omega: &[f32],
        coupling: &[f32],
        initial_phases: &[f32],
        n_steps: usize,
        dt: f32,
    ) -> Vec<f32> {
        assert_eq!(
            omega.len(),
            n_osc,
            "omega length {} != n_osc {}",
            omega.len(),
            n_osc
        );
        assert_eq!(
            initial_phases.len(),
            n_osc,
            "initial_phases length {} != n_osc {}",
            initial_phases.len(),
            n_osc
        );
        assert_eq!(
            coupling.len(),
            n_osc * n_osc,
            "coupling length {} != n_osc^2 {}",
            coupling.len(),
            n_osc * n_osc
        );

        if n_osc == 0 {
            return Vec::new();
        }
        if n_steps == 0 {
            return initial_phases.to_vec();
        }

        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;
        let vec_size = (n_osc * 4) as u64;

        let omega_buf = buffers::storage_buffer(dev, "kuramoto_omega", vec_size, true);
        queue.write_buffer(&omega_buf, 0, bytemuck::cast_slice(omega));
        let coupling_buf =
            buffers::storage_buffer(dev, "kuramoto_coupling", (n_osc * n_osc * 4) as u64, true);
        queue.write_buffer(&coupling_buf, 0, bytemuck::cast_slice(coupling));

        // Ping-pong phase buffers: A starts with the initial phases, B is scratch.
        let phase_a = buffers::storage_buffer(dev, "kuramoto_phase_a", vec_size, false);
        queue.write_buffer(&phase_a, 0, bytemuck::cast_slice(initial_phases));
        let phase_b = buffers::storage_buffer(dev, "kuramoto_phase_b", vec_size, false);
        let staging = buffers::staging_buffer(dev, "kuramoto_phase_staging", vec_size);
        let uniform_buf = buffers::uniform_buffer(dev, "kuramoto_params", 16);

        let params = KuramotoParams {
            n_osc: n_osc as u32,
            _pad: 0,
            dt,
            n_inv: 1.0 / n_osc as f32,
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&params));

        // Two bind groups, one per ping-pong direction (A→B and B→A).
        let bind_group = |src: &wgpu::Buffer, dst: &wgpu::Buffer, label: &str| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.ctx.kuramoto_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: omega_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: coupling_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let bg_ab = bind_group(&phase_a, &phase_b, "kuramoto_bg_ab");
        let bg_ba = bind_group(&phase_b, &phase_a, "kuramoto_bg_ba");

        let x_groups = (n_osc as u32).div_ceil(64);
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("kuramoto_run"),
        });
        for step in 0..n_steps {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("kuramoto_step"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ctx.kuramoto_pipeline);
            // Even steps read A→write B; odd steps read B→write A.
            pass.set_bind_group(0, if step % 2 == 0 { &bg_ab } else { &bg_ba }, &[]);
            pass.dispatch_workgroups(x_groups, 1, 1);
        }
        // The last write landed in B if n_steps is odd, else A.
        let final_buf = if n_steps % 2 == 1 { &phase_b } else { &phase_a };
        encoder.copy_buffer_to_buffer(final_buf, 0, &staging, 0, vec_size);
        queue.submit(std::iter::once(encoder.finish()));

        read_f32(dev, &staging, vec_size, n_osc)
    }

    /// Name of the GPU adapter (e.g. "AMD Radeon RX 6600 XT").
    pub fn gpu_name(&self) -> &str {
        &self.ctx.adapter_name
    }
}

/// Map a staging buffer, copy out `count` f32 values, and unmap.
fn read_f32(dev: &wgpu::Device, staging: &wgpu::Buffer, size: u64, count: usize) -> Vec<f32> {
    let slice = staging.slice(..size);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = dev.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let data = slice
        .get_mapped_range()
        .expect("GPU phase staging buffer is mapped for readback");
    let values: &[f32] = bytemuck::cast_slice(&data);
    let out = values[..count].to_vec();
    drop(data);
    staging.unmap();
    out
}
