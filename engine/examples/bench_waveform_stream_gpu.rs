// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — GPU real-time benchmark of the streaming lossless waveform coder

//! GPU encoder benchmark with CPU parity and decode checks.
//!
//! Usage: `cargo run --release --features gpu --example bench_waveform_stream_gpu -- <wav-list>
//! <rate-hz> <seconds> <channels,...> <window-samples,...>`. Channels replay the recordings as
//! in `bench_waveform_stream`. Every GPU packet is compared byte for byte with the CPU encoder
//! (one group per channel) and decoded by the CPU decoder. Prints ratio, GPU encode latency
//! per window (median, p99, max) and the real-time factor.

use sc_neurocore_engine::codecs::waveform_stream::{StreamDecoder, StreamEncoder};
use sc_neurocore_engine::codecs::waveform_stream_gpu::GpuStreamEncoder;
use std::time::Instant;

fn read_wav(path: &str) -> Vec<i16> {
    let bytes = std::fs::read(path).expect("readable WAV file");
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        if &bytes[pos..pos + 4] == b"data" {
            return bytes[pos + 8..pos + 8 + size]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| i16::from_le_bytes(*pair))
                .collect();
        }
        pos += 8 + size + (size & 1);
    }
    panic!("{path}: no data chunk");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let files: Vec<String> = std::fs::read_to_string(&args[1])
        .expect("file list")
        .lines()
        .map(String::from)
        .collect();
    let rate: f64 = args[2].parse().expect("rate");
    let seconds: f64 = args[3].parse().expect("seconds");
    let channel_counts: Vec<usize> = args[4]
        .split(',')
        .map(|v| v.parse().expect("channels"))
        .collect();
    let windows: Vec<usize> = args[5]
        .split(',')
        .map(|v| v.parse().expect("window"))
        .collect();
    let recordings: Vec<Vec<i16>> = files.iter().map(|f| read_wav(f)).collect();
    let length = recordings
        .iter()
        .map(Vec::len)
        .min()
        .expect("at least one recording");
    let total = ((rate * seconds) as usize).min(length);
    for &channels in &channel_counts {
        let mut stream = vec![0_i16; total * channels];
        for c in 0..channels {
            let source = &recordings[c % recordings.len()];
            let shift = (c / recordings.len()) * length / 7;
            for t in 0..total {
                stream[t * channels + c] = source[(t + shift) % length];
            }
        }
        for &window in &windows {
            let mut gpu = GpuStreamEncoder::new(channels, 10, window).expect("GPU encoder");
            let mut cpu = StreamEncoder::new(channels, 10, channels).expect("CPU encoder");
            let mut decoder = StreamDecoder::new(channels, 10, channels).expect("decoder");
            let (mut times, mut bytes) = (Vec::new(), 0_usize);
            let count = total / window;
            for k in 0..count {
                let slice = &stream[k * window * channels..(k + 1) * window * channels];
                let start = Instant::now();
                let packet = gpu.encode(slice, window).expect("GPU encode");
                times.push(start.elapsed().as_secs_f64());
                assert_eq!(
                    packet,
                    cpu.encode(slice, window).expect("CPU encode"),
                    "GPU packet differs from CPU"
                );
                assert_eq!(
                    decoder.decode(&packet, window).expect("decode"),
                    slice,
                    "lossless round trip"
                );
                bytes += packet.len();
            }
            let encode_total: f64 = times.iter().sum();
            times.sort_by(f64::total_cmp);
            println!(
                "{} channels {:6} window {:5} ({:7.3} ms): ratio {:.4}  GPU encode ms p50 {:8.3} p99 {:8.3} max {:8.3}  real-time factor {:7.2}",
                gpu.adapter_name,
                channels,
                window,
                window as f64 / rate * 1e3,
                (count * window * channels * 2) as f64 / bytes as f64,
                times[times.len() / 2] * 1e3,
                times[((times.len() - 1) as f64 * 0.99) as usize] * 1e3,
                times.last().copied().unwrap_or(0.0) * 1e3,
                (count * window) as f64 / rate / encode_total,
            );
        }
    }
}
