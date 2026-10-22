// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Real-time scaling benchmark of the streaming lossless waveform coder

//! Real-time scaling of the streaming lossless coder, pure Rust (no Python on the path).
//!
//! Usage: `cargo run --release --example bench_waveform_stream -- <wav-list> <rate-hz> <seconds>
//! <channels,...> <window-samples,...> <groups>`.
//!
//! The WAV files (mono int16, e.g. the Neuralink compression-challenge recordings) are read
//! once. Channel `c` replays recording `c mod files`, rotated by `(c / files) * len / 7` samples,
//! so channels beyond the number of recordings are time-shifted copies, not identical ones. Each
//! window is encoded and decoded in turn; the decoder output is compared with the input.
//! Printed per configuration: compression ratio (raw 16-bit bytes / packet bytes), encode and
//! decode latency per window (median, p99, max), and the real-time factor (stream time / encode
//! time).

use sc_neurocore_engine::codecs::waveform_stream::{StreamDecoder, StreamEncoder};
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

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
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
    let groups_arg = &args[6];
    let recordings: Vec<Vec<i16>> = files.iter().map(|f| read_wav(f)).collect();
    let length = recordings
        .iter()
        .map(Vec::len)
        .min()
        .expect("at least one recording");
    let total_samples = ((rate * seconds) as usize).min(length);
    println!(
        "threads {}, recordings {}, samples per channel {}",
        rayon::current_num_threads(),
        recordings.len(),
        total_samples
    );
    for &channels in &channel_counts {
        let mut stream = vec![0_i16; total_samples * channels];
        for c in 0..channels {
            let source = &recordings[c % recordings.len()];
            let shift = (c / recordings.len()) * length / 7;
            for t in 0..total_samples {
                stream[t * channels + c] = source[(t + shift) % length];
            }
        }
        for &window in &windows {
            let groups = if groups_arg == "threads" {
                rayon::current_num_threads().min(channels)
            } else {
                groups_arg.parse().expect("groups")
            };
            let mut encoder = StreamEncoder::new(channels, 10, groups).expect("encoder");
            let mut decoder = StreamDecoder::new(channels, 10, groups).expect("decoder");
            let (mut encode_times, mut decode_times, mut bytes) = (Vec::new(), Vec::new(), 0_usize);
            let count = total_samples / window;
            for k in 0..count {
                let slice = &stream[k * window * channels..(k + 1) * window * channels];
                let start = Instant::now();
                let packet = encoder.encode(slice, window).expect("encode");
                encode_times.push(start.elapsed().as_secs_f64());
                let start = Instant::now();
                let decoded = decoder.decode(&packet, window).expect("decode");
                decode_times.push(start.elapsed().as_secs_f64());
                assert_eq!(decoded, slice, "lossless round trip");
                bytes += packet.len();
            }
            let duration = (count * window) as f64 / rate;
            let encode_total: f64 = encode_times.iter().sum();
            encode_times.sort_by(f64::total_cmp);
            decode_times.sort_by(f64::total_cmp);
            println!(
                "channels {:6} window {:5} ({:7.3} ms) groups {:3}: ratio {:.4}  encode ms p50 {:8.3} p99 {:8.3} max {:8.3}  decode ms p50 {:8.3} p99 {:8.3}  real-time factor {:7.2}",
                channels,
                window,
                window as f64 / rate * 1e3,
                groups,
                (count * window * channels * 2) as f64 / bytes as f64,
                percentile(&encode_times, 0.5) * 1e3,
                percentile(&encode_times, 0.99) * 1e3,
                encode_times.last().copied().unwrap_or(0.0) * 1e3,
                percentile(&decode_times, 0.5) * 1e3,
                percentile(&decode_times, 0.99) * 1e3,
                duration / encode_total,
            );
        }
    }
}
