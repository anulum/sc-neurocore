// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Autonomous learning WGPU backend

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

struct WgpuContext {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
}

static WGPU_CONTEXT: OnceLock<Option<WgpuContext>> = OnceLock::new();

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct WgpuRuleParams {
    pub rule_type: u32,
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub dt: f32,
    pub count: u32,
    pub seed: u32,

    pub param_c: f32, // tau_e (R-STDP / ELIGENT)
    pub param_d: f32, // target_sum_weights (ELIGENT)
    pub _pad0: u32,
    pub _pad1: u32,
}

pub struct WgpuRuleLayer {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    weights_buf: wgpu::Buffer,
    pre_trace_buf: wgpu::Buffer,
    post_trace_buf: wgpu::Buffer,
    pre_probs_buf: wgpu::Buffer,
    post_probs_buf: wgpu::Buffer,
    param_extra_buf: wgpu::Buffer,
    param_extra2_buf: wgpu::Buffer,
    param_extra3_buf: wgpu::Buffer,
    rewards_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,

    pub count: u32,
    pub rule_type: u32,
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub param_c: f32,
    pub param_d: f32,
    seed_offset: u32,
}

impl WgpuRuleLayer {
    // The constructor mirrors the fixed WGSL parameter block; collapsing these
    // independent rule constants would obscure the C-ABI-to-shader mapping.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        count: usize,
        rule_type: u32,
        initial_weight: f32,
        a_plus: f32,
        a_minus: f32,
        tau_plus: f32,
        tau_minus: f32,
        param_c: f32,
        param_d: f32,
    ) -> Option<Self> {
        let parameters = [
            initial_weight,
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            param_c,
            param_d,
        ];
        let storage_bytes = count.checked_mul(std::mem::size_of::<f32>())?;
        if count == 0
            || count > u32::MAX as usize
            || storage_bytes > 1024 * 1024 * 1024
            || rule_type > 3
            || parameters.iter().any(|value| !value.is_finite())
            || !(0.0..=1.0).contains(&initial_weight)
            || a_plus < 0.0
            || a_minus < 0.0
            || tau_plus <= 0.0
            || tau_minus <= 0.0
            || param_c <= 0.0
            || param_d < 0.0
        {
            return None;
        }
        // Global Singleton context mapping replacing costly per-layer adapter querying over PCI-e
        let ctx_opt = WGPU_CONTEXT.get_or_init(|| {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))?;

            let mut limits = wgpu::Limits::downlevel_defaults();
            limits.max_storage_buffers_per_shader_stage = 16;
            limits.max_compute_workgroups_per_dimension = 65535;
            limits.max_storage_buffer_binding_size = 1024 * 1024 * 1024;
            limits.max_buffer_size = 1024 * 1024 * 1024;

            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Zenith WGPU Singleton Context"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            ))
            .ok()?;

            Some(WgpuContext {
                device: std::sync::Arc::new(device),
                queue: std::sync::Arc::new(queue),
            })
        });

        let ctx = ctx_opt.as_ref()?;
        // Natively, wgpu bindings (Device/Queue) are internally Arc-wrapped handles so cloning is virtually free
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();

        let wgsl_source = include_str!("shaders/plasticity.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Plasticity WGSL Kernel"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl_source)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Plasticity BindGroup"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Plasticity Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Plasticity Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        let create_storage_buf = |device: &wgpu::Device, contents: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(contents),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };

        // Initialize state arrays accurately out of box
        let zeros = vec![0.0f32; count];
        let mut param_extra2_init = vec![0.0f32; count];

        // Setup BCM theta_m (Starts at 0.5 default) and ELIGENT sum_weights (Starts at 1.0)
        let param_extra2_val = if rule_type == 0 {
            1.0f32
        } else if rule_type == 3 {
            0.5f32
        } else {
            0.0f32
        };
        param_extra2_init.fill(param_extra2_val);

        Some(Self {
            weights_buf: create_storage_buf(&device, &vec![initial_weight; count]),
            pre_trace_buf: create_storage_buf(&device, &zeros),
            post_trace_buf: create_storage_buf(&device, &zeros),
            pre_probs_buf: create_storage_buf(&device, &zeros),
            post_probs_buf: create_storage_buf(&device, &zeros),
            param_extra_buf: create_storage_buf(&device, &zeros),
            param_extra2_buf: create_storage_buf(&device, &param_extra2_init),
            param_extra3_buf: create_storage_buf(&device, &zeros),
            rewards_buf: create_storage_buf(&device, &zeros),
            params_buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<WgpuRuleParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),

            device,
            queue,
            compute_pipeline,
            bind_group_layout,
            count: count as u32,
            rule_type,
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            param_c,
            param_d,
            seed_offset: 42,
        })
    }

    pub fn set_deterministic_mode(&mut self, seed: u32) {
        self.seed_offset = seed;
    }

    pub fn step(&mut self, pre_probs: &[f32], post_probs: &[f32], rewards: &[f32], dt: f32) {
        self.queue
            .write_buffer(&self.pre_probs_buf, 0, bytemuck::cast_slice(pre_probs));
        self.queue
            .write_buffer(&self.post_probs_buf, 0, bytemuck::cast_slice(post_probs));
        if !rewards.is_empty() {
            self.queue
                .write_buffer(&self.rewards_buf, 0, bytemuck::cast_slice(rewards));
        }

        let params = WgpuRuleParams {
            rule_type: self.rule_type,
            a_plus: self.a_plus,
            a_minus: self.a_minus,
            tau_plus: self.tau_plus,
            tau_minus: self.tau_minus,
            dt,
            count: self.count,
            seed: self.seed_offset,
            param_c: self.param_c,
            param_d: self.param_d,
            _pad0: 0,
            _pad1: 0,
        };
        self.seed_offset = self.seed_offset.wrapping_add(1);

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Runtime BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pre_trace_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.post_trace_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.pre_probs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.post_probs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.param_extra_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.param_extra2_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.param_extra3_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.rewards_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let mut workgroups_x = (self.count as f32 / 256.0).ceil() as u32;
            let mut workgroups_y = 1;
            if workgroups_x > 65535 {
                workgroups_y = (workgroups_x as f32 / 65535.0).ceil() as u32;
                workgroups_x = 65535;
            }
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        // Wait for execution to finish
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Zero the plasticity traces without changing the learned weights.
    ///
    /// Mirrors the `PlasticityRule::reset` contract used by the CPU backend:
    /// pre/post traces, eligibility (param_extra) and threshold (param_extra3)
    /// are cleared; the rule-dependent accumulator (param_extra2 = BCM θ_m
    /// or ELIGENT sum_weights) is re-initialised to the same value used at
    /// construction (0.5 for BCM, 1.0 for ELIGENT, 0.0 otherwise); the
    /// weights buffer is untouched.
    pub fn reset(&mut self) {
        let count = self.count as usize;
        let zeros = vec![0.0f32; count];
        self.queue
            .write_buffer(&self.pre_trace_buf, 0, bytemuck::cast_slice(&zeros));
        self.queue
            .write_buffer(&self.post_trace_buf, 0, bytemuck::cast_slice(&zeros));
        self.queue
            .write_buffer(&self.param_extra_buf, 0, bytemuck::cast_slice(&zeros));
        self.queue
            .write_buffer(&self.param_extra3_buf, 0, bytemuck::cast_slice(&zeros));

        let param_extra2_val = if self.rule_type == 0 {
            1.0f32
        } else if self.rule_type == 3 {
            0.5f32
        } else {
            0.0f32
        };
        let extra2 = vec![param_extra2_val; count];
        self.queue
            .write_buffer(&self.param_extra2_buf, 0, bytemuck::cast_slice(&extra2));

        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Replace all weights after the caller has validated length and domain.
    pub fn set_weights(&mut self, weights: &[f32]) -> bool {
        if weights.len() != self.count as usize
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || !(0.0..=1.0).contains(weight))
        {
            return false;
        }
        self.queue
            .write_buffer(&self.weights_buf, 0, bytemuck::cast_slice(weights));
        self.device.poll(wgpu::Maintain::Wait);
        true
    }

    pub fn get_weights(&self) -> Option<Vec<f32>> {
        let size = (self.count as usize * std::mem::size_of::<f32>()) as u64;
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.weights_buf, 0, &staging_buf, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().ok()?.ok()?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::WgpuRuleLayer;

    #[test]
    fn rejects_oversized_buffer_before_requesting_an_adapter() {
        let count = 1024 * 1024 * 1024 / std::mem::size_of::<f32>() + 1;
        assert!(WgpuRuleLayer::new(count, 1, 0.5, 0.01, 0.005, 20.0, 20.0, 20.0, 1.0).is_none());
    }

    #[test]
    fn rejects_invalid_configuration_before_requesting_an_adapter() {
        assert!(WgpuRuleLayer::new(1, 4, 0.5, 0.01, 0.005, 20.0, 20.0, 20.0, 1.0).is_none());
        assert!(WgpuRuleLayer::new(1, 1, f32::NAN, 0.01, 0.005, 20.0, 20.0, 20.0, 1.0).is_none());
    }
}
