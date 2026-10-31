// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GPU lossless stream size, refusal and retry contracts

//! Public GPU waveform contracts; device cases require a physical GPU.
//!
//! Run with `--no-default-features --features gpu --test waveform_stream_gpu`
//! and `--test-threads=1`. These mathematical ADC vectors test codec contracts;
//! they are not recording data or performance measurements.

#![cfg(feature = "gpu")]

use sc_neurocore_engine::codecs::waveform_stream::{StreamDecoder, StreamEncoder, COUNT_LIMIT};
use sc_neurocore_engine::codecs::waveform_stream_gpu::GpuStreamEncoder;

/// Require the public constructor to refuse a request without panicking.
fn refused(channels: usize, bits: u32, samples: usize) -> String {
    match GpuStreamEncoder::new(channels, bits, samples) {
        Err(message) => message,
        Ok(_) => panic!("GPU encoder admitted {channels} channels, {bits} bits, {samples} rows"),
    }
}

/// Inspect the same selected physical adapter's published limits.
fn physical_limits() -> wgpu::Limits {
    let backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = if let Ok(filter) = std::env::var("SC_GPU_ADAPTER") {
        let filter = filter.to_lowercase();
        pollster::block_on(instance.enumerate_adapters(backends))
            .into_iter()
            .find(|adapter| adapter.get_info().name.to_lowercase().contains(&filter))
            .expect("selected physical GPU must be present")
    } else {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("physical GPU must be present")
    };
    assert!(matches!(
        adapter.get_info().device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
    ));
    let limits = adapter.limits();
    eprintln!(
        "GPU {}: buffer limit {}, storage binding limit {}",
        adapter.get_info().name,
        limits.max_buffer_size,
        limits.max_storage_buffer_binding_size
    );
    limits
}

/// Exercise real GPU coding, CPU byte parity and lossless decoding together.
fn equal_packet(
    gpu: &mut GpuStreamEncoder,
    cpu: &mut StreamEncoder,
    decoder: &mut StreamDecoder,
    window: &[i16],
    rows: usize,
) {
    let packet = gpu.encode(window, rows).expect("GPU encoding must succeed");
    assert_eq!(packet, cpu.encode(window, rows).unwrap());
    assert_eq!(decoder.decode(&packet, rows).unwrap(), window);
}

#[test]
fn constructor_invalid_configuration_is_refused() {
    for (channels, bits, samples) in [(0, 10, 4), (1, 0, 4), (1, 17, 4), (1, 10, 0)] {
        assert!(refused(channels, bits, samples).contains("needs"));
    }
}

#[test]
fn constructor_unrepresentable_shader_geometry_is_refused() {
    for (channels, bits, samples) in [
        (usize::MAX, 16, 1),
        (usize::MAX, 1, usize::MAX),
        (1, 10, usize::MAX),
        (1, 10, (u32::MAX as usize).saturating_add(1)),
    ] {
        assert!(!refused(channels, bits, samples).is_empty());
    }
}

#[test]
fn constructor_channel_count_matches_cpu_packet_contract() {
    physical_limits();
    let channels = u16::MAX as usize + 1;
    assert!(StreamEncoder::new(channels, 1, channels).is_err());
    assert!(refused(channels, 1, 1).contains("channel"));
    let channels = u16::MAX as usize;
    let mut gpu = GpuStreamEncoder::new(channels, 1, 1).unwrap();
    let mut cpu = StreamEncoder::new(channels, 1, channels).unwrap();
    let mut decoder = StreamDecoder::new(channels, 1, channels).unwrap();
    equal_packet(&mut gpu, &mut cpu, &mut decoder, &vec![-32768; channels], 1);
}

#[test]
fn device_storage_overflow_is_refused_before_buffer_creation() {
    let limits = physical_limits();
    let ceiling = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size)
        .min(u64::from(u32::MAX)) as usize;
    let oversized_output_rows = ceiling / 12 + 1;
    let oversized_table_channels = ceiling / (3 * 4 * (1 << 16)) + 1;
    for (channels, bits, samples) in [
        (1, 10, oversized_output_rows),
        (oversized_table_channels, 16, 1),
    ] {
        let message = refused(channels, bits, samples);
        assert!(message.contains("limit"), "{message}");
        assert!(
            message.contains("buffer") || message.contains("table"),
            "{message}"
        );
    }
}

#[test]
fn adc_widths_and_workgroup_boundaries_keep_streaming_byte_parity() {
    physical_limits();
    for (bits, values) in [
        (1, &[-32768, 32767][..]),
        (10, &[-32768, -1, 0, 32767][..]),
        (16, &[-32768, -1, 0, 1, 32767][..]),
    ] {
        for channels in [1, 63, 64, 65] {
            let mut gpu = GpuStreamEncoder::new(channels, bits, 4).unwrap();
            let mut cpu = StreamEncoder::new(channels, bits, channels).unwrap();
            let mut decoder = StreamDecoder::new(channels, bits, channels).unwrap();
            for offset in [0, 1, 2] {
                let input: Vec<i16> = (0..4 * channels)
                    .map(|index| values[(index + offset) % values.len()])
                    .collect();
                equal_packet(&mut gpu, &mut cpu, &mut decoder, &input, 4);
            }
        }
    }
}

#[test]
fn invalid_shapes_and_capacity_preserve_healthy_retry() {
    physical_limits();
    let mut gpu = GpuStreamEncoder::new(3, 10, 4).unwrap();
    let mut cpu = StreamEncoder::new(3, 10, 3).unwrap();
    let mut decoder = StreamDecoder::new(3, 10, 3).unwrap();
    let good = [0; 6];
    equal_packet(&mut gpu, &mut cpu, &mut decoder, &good, 2);
    for (input, rows, message) in [
        (vec![0; 5], 2, "window size"),
        (vec![], usize::MAX, "window size"),
        (vec![0; 15], 5, "max_samples"),
    ] {
        assert!(gpu.encode(&input, rows).unwrap_err().contains(message));
        equal_packet(&mut gpu, &mut cpu, &mut decoder, &good, 2);
    }
    equal_packet(&mut gpu, &mut cpu, &mut decoder, &[], 0);
    equal_packet(&mut gpu, &mut cpu, &mut decoder, &good, 2);
}

#[test]
fn late_channel_adc_conflicts_preserve_every_channel_and_pending_table() {
    physical_limits();
    let channels = 65;
    let mut gpu = GpuStreamEncoder::new(channels, 10, 2).unwrap();
    let mut cpu = StreamEncoder::new(channels, 10, channels).unwrap();
    let mut decoder = StreamDecoder::new(channels, 10, channels).unwrap();
    let good = vec![0; 2 * channels];
    // 0 and 1 are distinct int16 values at the same 10-bit ADC level.
    let mut conflict = good.clone();
    conflict[2 * channels - 1] = 1;
    assert!(gpu
        .encode(&conflict, 2)
        .unwrap_err()
        .contains("contradicts"));
    equal_packet(&mut gpu, &mut cpu, &mut decoder, &good, 2);
    // The same late-channel conflict now contradicts an established level table.
    assert!(gpu
        .encode(&conflict, 2)
        .unwrap_err()
        .contains("contradicts"));
    equal_packet(&mut gpu, &mut cpu, &mut decoder, &good, 2);
}

#[test]
fn level_count_halving_keeps_cpu_parity_across_windows() {
    physical_limits();
    let rows = COUNT_LIMIT as usize + 3;
    let mut gpu = GpuStreamEncoder::new(1, 1, rows).unwrap();
    let mut cpu = StreamEncoder::new(1, 1, 1).unwrap();
    let mut decoder = StreamDecoder::new(1, 1, 1).unwrap();
    equal_packet(&mut gpu, &mut cpu, &mut decoder, &vec![-32768; rows], rows);
    equal_packet(&mut gpu, &mut cpu, &mut decoder, &[32767, -32768], 2);
}

#[test]
fn missing_adc_code_exclusion_and_return_keep_cpu_parity() {
    physical_limits();
    let mut gpu = GpuStreamEncoder::new(1, 2, 64).unwrap();
    let mut cpu = StreamEncoder::new(1, 2, 1).unwrap();
    let mut decoder = StreamDecoder::new(1, 2, 1).unwrap();
    // Level 1 is absent while its neighbours 0 and 2 become common.
    let common: Vec<i16> = [-32768, 10922].into_iter().cycle().take(64).collect();
    equal_packet(&mut gpu, &mut cpu, &mut decoder, &common, 64);
    // Repeated observations of level 1 exercise its escape and subsequent return.
    equal_packet(&mut gpu, &mut cpu, &mut decoder, &[-10923; 8], 8);
    equal_packet(&mut gpu, &mut cpu, &mut decoder, &common, 64);
}
