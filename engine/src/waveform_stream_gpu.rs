// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GPU streaming lossless encoder (wgpu compute, one range coder per channel)

//! GPU encoder for the streaming lossless coder with one range coder per channel.
//!
//! Its packets are byte-identical to [`super::waveform_stream::StreamEncoder`] with
//! `groups == channels`, so the CPU [`super::waveform_stream::StreamDecoder`] decodes them.
//! Each window runs three compute passes: validation (a contradicting sample refuses the whole
//! window before any state changes), encoding (one invocation per channel), and compaction of
//! the channel outputs to word-aligned offsets so only coded bytes cross the bus.

use super::waveform_stream::{level_of, nominal_value};
use wgpu::util::DeviceExt;

const STATE_WORDS: usize = 306;
const UNSET: i32 = i32::MIN;
const PROB_INIT: i32 = 2048;
const WORKGROUP: u32 = 64;

/// GPU encoder of one lossless stream (`groups == channels`).
pub struct GpuStreamEncoder {
    device: wgpu::Device,
    queue: wgpu::Queue,
    validate: wgpu::ComputePipeline,
    encode: wgpu::ComputePipeline,
    compact: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    samples: wgpu::Buffer,
    levels: wgpu::Buffer,
    nominal: wgpu::Buffer,
    state: wgpu::Buffer,
    table: wgpu::Buffer,
    out_words: wgpu::Buffer,
    out_len: wgpu::Buffer,
    conflict: wgpu::Buffer,
    offsets: wgpu::Buffer,
    compact_buf: wgpu::Buffer,
    pending: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    small_staging: wgpu::Buffer,
    big_staging: wgpu::Buffer,
    channels: usize,
    bits: u32,
    max_samples: usize,
    max_words: usize,
    /// Name of the adapter in use.
    pub adapter_name: String,
}

fn layout_entry(binding: u32, kind: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: kind,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage(device: &wgpu::Device, label: &str, size: usize, map: bool) -> wgpu::Buffer {
    let usage = if map {
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
    } else {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    };
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4) as u64,
        usage,
        mapped_at_creation: false,
    })
}

impl GpuStreamEncoder {
    /// New GPU encoder for `channels` channels of a `bits`-bit ADC and windows of at most
    /// `max_samples` samples. The adapter whose name contains `SC_GPU_ADAPTER` (if set) is used,
    /// otherwise the high-performance default.
    ///
    /// Channels must fit the CPU packet contract (`1..=65535`), ADC widths must be `1..=16`,
    /// and window lengths must fit a nonzero `u32`. Buffer sizes must fit both the device's
    /// allocation and storage-binding limits and the shader's 32-bit indices.
    pub fn new(channels: usize, bits: u32, max_samples: usize) -> Result<Self, String> {
        if channels == 0 || !(1..=16).contains(&bits) || max_samples == 0 {
            return Err(
                "GPU stream encoder needs channels >= 1, adc_bits 1..16 and max_samples >= 1"
                    .into(),
            );
        }
        if channels > u16::MAX as usize || max_samples > u32::MAX as usize {
            return Err("GPU stream dimensions exceed the CPU channel or u32 shader limits".into());
        }
        // Per channel and level: exact value, count and dense rank (three adjacent words,
        // channels * 2^bits * 3 in all), which must fit one storage binding.
        let table_bytes = 3 * ((channels as u64) << (bits + 2));
        let backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = backends;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = if let Ok(filter) = std::env::var("SC_GPU_ADAPTER") {
            let filter = filter.to_lowercase();
            pollster::block_on(instance.enumerate_adapters(backends))
                .into_iter()
                .find(|a| a.get_info().name.to_lowercase().contains(&filter))
        } else {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            }))
            .ok()
        }
        .ok_or_else(|| "no GPU adapter available".to_string())?;
        let adapter_name = adapter.get_info().name.clone();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.max(11);
        limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter.limits().max_buffer_size;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("waveform-stream-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|error| format!("GPU device: {error}"))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("waveform_stream_gpu"),
            source: wgpu::ShaderSource::Wgsl(include_str!("waveform_stream_gpu.wgsl").into()),
        });
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("waveform_stream_gpu_layout"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, ro),
                layout_entry(2, ro),
                layout_entry(3, ro),
                layout_entry(4, rw),
                layout_entry(5, rw),
                layout_entry(6, rw),
                layout_entry(7, rw),
                layout_entry(8, rw),
                layout_entry(9, ro),
                layout_entry(10, rw),
                layout_entry(11, rw),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("waveform_stream_gpu_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (validate, encode, compact) = (
            pipeline("validate"),
            pipeline("encode"),
            pipeline("compact_out"),
        );
        let max_words = (12 * max_samples as u128 + 8).div_ceil(4);
        // Output buffers bound sample and per-channel staging sizes; the level lookup also
        // bounds the nominal lookup, and the three-word table bounds pending level values.
        // Check state and both lookup tables as well as output before allocating any of them.
        let largest = (4 * channels as u128 * max_words)
            .max(4 * channels as u128 * STATE_WORDS as u128)
            .max(u128::from(table_bytes))
            .max(4 * 65536);
        let buffer_limit = u128::from(limits.max_buffer_size)
            .min(u128::from(limits.max_storage_buffer_binding_size))
            .min(u128::from(u32::MAX));
        if largest > buffer_limit {
            return Err(format!(
                "{channels} channels with windows of {max_samples} samples need {largest}-byte buffers, above the GPU buffer, binding or index limit"
            ));
        }
        let max_words = max_words as usize;
        let level_table: Vec<u32> = (-32768_i64..=32767)
            .map(|v| level_of(v, bits) as u32)
            .collect();
        let nominal_table: Vec<i32> = (0..(1_i64 << bits))
            .map(|l| nominal_value(l, bits) as i32)
            .collect();
        let init = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let levels = init("levels", bytemuck::cast_slice(&level_table));
        let nominal = init("nominal", bytemuck::cast_slice(&nominal_table));
        // Structure of arrays: word `w` of channel `c` lives at `w * channels + c`.
        let mut state_init = vec![0_i32; channels * STATE_WORDS];
        state_init[..channels].fill(1 << (bits - 1));
        state_init[82 * channels..].fill(PROB_INIT);
        let state = init("state", bytemuck::cast_slice(&state_init));
        let unset = vec![UNSET; channels << bits];
        let table_init: Vec<i32> = (0..channels)
            .flat_map(|_| (0..1_i32 << bits).flat_map(|level| [UNSET, 0, level]))
            .collect();
        let table = init("table", bytemuck::cast_slice(&table_init));
        let pending = init("pending", bytemuck::cast_slice(&unset));
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let words_total = channels * max_words;
        let mut encoder = GpuStreamEncoder {
            samples: storage(&device, "samples", 4 * max_samples * channels, false),
            out_words: storage(&device, "out_words", 4 * words_total, false),
            out_len: storage(&device, "out_len", 4 * channels, false),
            conflict: storage(&device, "conflict", 4, false),
            offsets: storage(&device, "offsets", 4 * channels, false),
            compact_buf: storage(&device, "compact", 4 * words_total, false),
            small_staging: storage(&device, "small_staging", 4 * (channels + 1), true),
            big_staging: storage(&device, "big_staging", 4 * words_total, true),
            device,
            queue,
            validate,
            encode,
            compact,
            layout,
            params,
            levels,
            nominal,
            state,
            table,
            channels,
            bits,
            max_samples,
            max_words,
            adapter_name,
            pending,
            bind_group: None,
        };
        encoder.bind_group = Some(encoder.make_bind_group());
        Ok(encoder)
    }

    fn make_bind_group(&self) -> wgpu::BindGroup {
        let buffers = [
            &self.params,
            &self.samples,
            &self.levels,
            &self.nominal,
            &self.state,
            &self.table,
            &self.out_words,
            &self.out_len,
            &self.conflict,
            &self.offsets,
            &self.compact_buf,
            &self.pending,
        ];
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("waveform_stream_gpu"),
            layout: &self.layout,
            entries: &entries,
        })
    }

    fn read(&self, staging: &wgpu::Buffer, bytes: u64) -> Vec<u32> {
        let slice = staging.slice(0..bytes.max(4));
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let data = slice
            .get_mapped_range()
            .expect("staging buffer is mapped for readback");
        let words: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        words
    }

    /// Encode one row-major `(samples, channels)` window into a packet (CPU packet format with
    /// one group per channel). A contradicting sample refuses the window with the encoder
    /// state unchanged.
    pub fn encode(&mut self, window: &[i16], samples: usize) -> Result<Vec<u8>, String> {
        if samples.checked_mul(self.channels) != Some(window.len()) {
            return Err("window size does not match samples x channels".into());
        }
        if samples > self.max_samples {
            return Err(format!(
                "window of {samples} samples exceeds max_samples {}",
                self.max_samples
            ));
        }
        let channels = self.channels;
        let values: Vec<i32> = window.iter().map(|&v| i32::from(v)).collect();
        let params = [
            channels as u32,
            samples as u32,
            self.bits,
            self.max_words as u32,
        ];
        self.queue
            .write_buffer(&self.params, 0, bytemuck::cast_slice(&params));
        self.queue
            .write_buffer(&self.samples, 0, bytemuck::cast_slice(&values));
        self.queue
            .write_buffer(&self.conflict, 0, bytemuck::cast_slice(&[0_u32]));
        let bind_group = self.bind_group.as_ref().expect("bind group created in new");
        let groups = (channels as u32).div_ceil(WORKGROUP);
        let mut commands = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encode"),
            });
        for pipeline in [&self.validate, &self.encode] {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        commands.copy_buffer_to_buffer(
            &self.out_len,
            0,
            &self.small_staging,
            0,
            4 * channels as u64,
        );
        commands.copy_buffer_to_buffer(
            &self.conflict,
            0,
            &self.small_staging,
            4 * channels as u64,
            4,
        );
        self.queue.submit(std::iter::once(commands.finish()));
        let small = self.read(&self.small_staging, 4 * (channels as u64 + 1));
        if small[channels] != 0 {
            let channel = small[..channels]
                .iter()
                .position(|&len| len == u32::MAX)
                .unwrap_or(0);
            return Err(format!(
                "a sample in channel {channel} contradicts its {}-bit ADC level table",
                self.bits
            ));
        }
        let lengths = &small[..channels];
        let mut offsets = Vec::with_capacity(channels);
        let mut total = 0_u32;
        for &length in lengths {
            offsets.push(total);
            total += length.div_ceil(4);
        }
        self.queue
            .write_buffer(&self.offsets, 0, bytemuck::cast_slice(&offsets));
        let mut commands = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compact"),
            });
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compact);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        commands.copy_buffer_to_buffer(
            &self.compact_buf,
            0,
            &self.big_staging,
            0,
            4 * u64::from(total.max(1)),
        );
        self.queue.submit(std::iter::once(commands.finish()));
        let words = self.read(&self.big_staging, 4 * u64::from(total.max(1)));
        let bytes: &[u8] = bytemuck::cast_slice(&words);
        let mut packet = Vec::with_capacity(bytes.len() + 4 * channels);
        for (c, &length) in lengths.iter().enumerate() {
            let start = 4 * offsets[c] as usize;
            packet.extend_from_slice(&length.to_be_bytes());
            packet.extend_from_slice(&bytes[start..start + length as usize]);
        }
        Ok(packet)
    }
}
