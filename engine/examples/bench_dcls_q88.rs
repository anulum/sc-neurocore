// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - DCLS Q8.8 Rust benchmark artefact writer

mod benchmark_context;

use benchmark_context::{load_average, measurement_context_json, rust_version};
use sc_neurocore_engine::scpn::dcls::dcls_max_forward_q88;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const TAPS: usize = 16;
const SAMPLES: usize = 4096;
const ITERATIONS: usize = 100;
const REPEATS: usize = 7;

fn deterministic_weights() -> Vec<i16> {
    (0..TAPS)
        .map(|tap| (((tap * 73 + 19) % 513) as i32 - 256) as i16)
        .collect()
}

fn spike_window(index: usize) -> [u8; TAPS] {
    let mut spikes = [0_u8; TAPS];
    for tap in 0..TAPS {
        let value = (index + tap * 3) % 7;
        spikes[tap] = if value == 0 || value == 1 || value == 4 {
            1
        } else {
            0
        };
    }
    spikes
}

fn time_dcls(weights: &[i16]) -> (f64, i64, usize, usize) {
    let centre_q88 = ((TAPS / 2) << 8) as i16;
    let sigma_q88 = (TAPS << 8) as i16;
    let start = Instant::now();
    let mut checksum = 0_i64;
    let mut overflow_count = 0_usize;
    let mut active_tap_total = 0_usize;
    for _ in 0..ITERATIONS {
        for sample in 0..SAMPLES {
            let spikes = spike_window(sample);
            let result = dcls_max_forward_q88(
                black_box(&spikes),
                black_box(weights),
                black_box(centre_q88),
                black_box(sigma_q88),
            )
            .expect("deterministic DCLS benchmark dimensions must be valid");
            checksum ^= i64::from(result.accumulator_q16_16);
            overflow_count += usize::from(result.overflow);
            active_tap_total += result.active_tap_count;
        }
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;
    (
        elapsed_ns / (ITERATIONS * SAMPLES) as f64,
        checksum,
        overflow_count,
        active_tap_total,
    )
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn values_json(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() {
    let load_average_before = load_average();
    let weights = deterministic_weights();
    let mut ns_per_sample = Vec::with_capacity(REPEATS);
    let mut checksum = 0_i64;
    let mut overflow_count = 0_usize;
    let mut active_tap_total = 0_usize;

    for _ in 0..REPEATS {
        let (ns, run_checksum, run_overflow_count, run_active_tap_total) = time_dcls(&weights);
        ns_per_sample.push(ns);
        checksum ^= run_checksum;
        overflow_count = run_overflow_count;
        active_tap_total = run_active_tap_total;
    }

    let mut sorted = ns_per_sample.clone();
    let median_ns_per_sample = median(&mut sorted);
    let timestamp_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after UNIX epoch")
        .as_secs();
    let total_samples = SAMPLES * ITERATIONS;

    let report = format!(
        concat!(
            "{{\n",
            "  \"benchmark\": \"dcls_q88_rtl_contract\",\n",
            "  \"language\": \"Rust\",\n",
            "  \"timestamp_unix\": {timestamp_unix},\n",
            "  \"command\": \"taskset -c 10-11 cargo run --manifest-path engine/Cargo.toml --release --example bench_dcls_q88\",\n",
            "  \"rustc\": \"{rust_version}\",\n",
            "  \"target_os\": \"{os}\",\n",
            "  \"target_arch\": \"{arch}\",\n",
            "  \"measurement_context\": {measurement_context},\n",
            "  \"taps\": {taps},\n",
            "  \"samples\": {samples},\n",
            "  \"iterations\": {iterations},\n",
            "  \"repeats\": {repeats},\n",
            "  \"total_samples\": {total_samples},\n",
            "  \"median_ns_per_sample\": {median_ns_per_sample:.3},\n",
            "  \"min_ns_per_sample\": {min_ns_per_sample:.3},\n",
            "  \"max_ns_per_sample\": {max_ns_per_sample:.3},\n",
            "  \"checksum\": {checksum},\n",
            "  \"overflow_count\": {overflow_count},\n",
            "  \"active_tap_total\": {active_tap_total},\n",
            "  \"results_ns_per_sample\": [{results}]\n",
            "}}\n"
        ),
        timestamp_unix = timestamp_unix,
        rust_version = rust_version(),
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        measurement_context = measurement_context_json(&load_average_before),
        taps = TAPS,
        samples = SAMPLES,
        iterations = ITERATIONS,
        repeats = REPEATS,
        total_samples = total_samples,
        median_ns_per_sample = median_ns_per_sample,
        min_ns_per_sample = sorted[0],
        max_ns_per_sample = sorted[sorted.len() - 1],
        checksum = checksum,
        overflow_count = overflow_count,
        active_tap_total = active_tap_total,
        results = values_json(&ns_per_sample),
    );

    let path = Path::new("benchmarks/results/local_rust_2026-06-04_dcls_q88.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("benchmark result directory must be writable");
    }
    fs::write(path, &report).expect("benchmark result artefact must be writable");
    print!("{report}");
}
