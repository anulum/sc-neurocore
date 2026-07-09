// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! GPU context management for wgpu compute.
//!
//! Provides a lazily-initialised singleton [`GpuContext`] holding the
//! wgpu device, queue, and pre-compiled compute pipelines.

use std::sync::{Arc, OnceLock};
use wgpu;

/// Shared GPU resources: device, queue, and compiled pipelines.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub encode_pipeline: wgpu::ComputePipeline,
    pub accumulate_pipeline: wgpu::ComputePipeline,
    pub lif_pipeline: wgpu::ComputePipeline,
    pub kuramoto_pipeline: wgpu::ComputePipeline,
    pub izhikevich_pipeline: wgpu::ComputePipeline,
    pub encode_bind_group_layout: wgpu::BindGroupLayout,
    pub accumulate_bind_group_layout: wgpu::BindGroupLayout,
    pub lif_bind_group_layout: wgpu::BindGroupLayout,
    pub kuramoto_bind_group_layout: wgpu::BindGroupLayout,
    pub izhikevich_bind_group_layout: wgpu::BindGroupLayout,
    pub adapter_name: String,
}

/// Global singleton — initialised once, shared across all GpuDenseLayers.
static GPU_CTX: OnceLock<Option<Arc<GpuContext>>> = OnceLock::new();

/// Returns a shared GPU context, or `None` if no GPU is available.
pub fn get_context() -> Option<Arc<GpuContext>> {
    GPU_CTX
        .get_or_init(|| GpuContext::try_new().map(Arc::new))
        .clone()
}

/// Returns `true` if a GPU is available for compute.
pub fn is_available() -> bool {
    get_context().is_some()
}

impl GpuContext {
    fn try_new() -> Option<Self> {
        let backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = backends;
        let instance = wgpu::Instance::new(descriptor);

        // Allow adapter selection via SC_GPU_ADAPTER env var (substring match).
        let adapter = if let Ok(name_filter) = std::env::var("SC_GPU_ADAPTER") {
            let filter_lower = name_filter.to_lowercase();
            pollster::block_on(instance.enumerate_adapters(backends))
                .into_iter()
                .find(|a| a.get_info().name.to_lowercase().contains(&filter_lower))
        } else {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            }))
            .ok()
        }?;

        let adapter_name = adapter.get_info().name.clone();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("sc-neurocore-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()?;

        // Compile encode shader.
        let encode_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("encode"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/encode.wgsl").into()),
        });

        // Compile accumulate shader.
        let accum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("accumulate"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/accumulate.wgsl").into()),
        });

        // Compile fixed-point LIF batch-step shader.
        let lif_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lif_step"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lif_step.wgsl").into()),
        });

        // Compile Kuramoto oscillator step shader.
        let kuramoto_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kuramoto_step"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/kuramoto_step.wgsl").into()),
        });

        // Compile Izhikevich batch-step shader.
        let izhikevich_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("izhikevich_step"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/izhikevich_step.wgsl").into()),
        });

        // Bind group layouts.
        let encode_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("encode_bgl"),
            entries: &[
                bgl_storage_ro(0), // input_probs
                bgl_storage_rw(1), // packed_inputs
                bgl_uniform(2),    // params
            ],
        });

        let accum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("accum_bgl"),
            entries: &[
                bgl_storage_ro(0), // packed_weights
                bgl_storage_ro(1), // packed_inputs
                bgl_storage_rw(2), // output
                bgl_uniform(3),    // params
            ],
        });

        let lif_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lif_bgl"),
            entries: &[
                bgl_storage_ro(0), // currents
                bgl_storage_rw(1), // spikes_out
                bgl_storage_rw(2), // voltages_out
                bgl_uniform(3),    // params
            ],
        });

        let kuramoto_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("kuramoto_bgl"),
            entries: &[
                bgl_storage_ro(0), // omega
                bgl_storage_ro(1), // coupling
                bgl_storage_ro(2), // phase_in
                bgl_storage_rw(3), // phase_out
                bgl_uniform(4),    // params
            ],
        });

        let izhikevich_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("izhikevich_bgl"),
            entries: &[
                bgl_storage_ro(0), // currents
                bgl_storage_rw(1), // spikes_out
                bgl_storage_rw(2), // voltages_out
                bgl_uniform(3),    // params
            ],
        });

        // Pipelines.
        let encode_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("encode_pl"),
            bind_group_layouts: &[Some(&encode_bgl)],
            immediate_size: 0,
        });
        let encode_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("encode_pipeline"),
            layout: Some(&encode_pl),
            module: &encode_shader,
            entry_point: Some("encode_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let accum_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("accum_pl"),
            bind_group_layouts: &[Some(&accum_bgl)],
            immediate_size: 0,
        });
        let accumulate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("accum_pipeline"),
                layout: Some(&accum_pl),
                module: &accum_shader,
                entry_point: Some("accumulate_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let lif_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lif_pl"),
            bind_group_layouts: &[Some(&lif_bgl)],
            immediate_size: 0,
        });
        let lif_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lif_pipeline"),
            layout: Some(&lif_pl),
            module: &lif_shader,
            entry_point: Some("lif_step_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let kuramoto_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("kuramoto_pl"),
            bind_group_layouts: &[Some(&kuramoto_bgl)],
            immediate_size: 0,
        });
        let kuramoto_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kuramoto_pipeline"),
            layout: Some(&kuramoto_pl),
            module: &kuramoto_shader,
            entry_point: Some("kuramoto_step_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let izhikevich_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("izhikevich_pl"),
            bind_group_layouts: &[Some(&izhikevich_bgl)],
            immediate_size: 0,
        });
        let izhikevich_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("izhikevich_pipeline"),
                layout: Some(&izhikevich_pl),
                module: &izhikevich_shader,
                entry_point: Some("izhikevich_step_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Some(GpuContext {
            device,
            queue,
            encode_pipeline,
            accumulate_pipeline,
            lif_pipeline,
            kuramoto_pipeline,
            izhikevich_pipeline,
            encode_bind_group_layout: encode_bgl,
            accumulate_bind_group_layout: accum_bgl,
            lif_bind_group_layout: lif_bgl,
            kuramoto_bind_group_layout: kuramoto_bgl,
            izhikevich_bind_group_layout: izhikevich_bgl,
            adapter_name,
        })
    }
}

// ---- Bind group layout entry helpers ----

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
