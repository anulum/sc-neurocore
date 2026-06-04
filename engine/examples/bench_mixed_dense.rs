// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Mixed dense Rust benchmark artefact writer

mod benchmark_context;

use benchmark_context::{load_average, measurement_context_json, rust_version};
use sc_neurocore_engine::ir::qformat::mixed_dense_q88_q1616;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const N_INPUTS: usize = 64;
const N_OUTPUTS: usize = 32;
const ITERATIONS: usize = 20_000;
const REPEATS: usize = 7;

fn deterministic_weights() -> Vec<i16> {
    (0..(N_INPUTS * N_OUTPUTS))
        .map(|i| (((i * 17 + 11) % 513) as i32 - 256) as i16)
        .collect()
}

fn deterministic_inputs() -> Vec<i32> {
    (0..N_INPUTS)
        .map(|i| (((i * 19 + 5) % 257) as i32 - 128) << 8)
        .collect()
}

fn run_once(weights: &[i16], inputs: &[i32]) -> (f64, i64, usize) {
    let start = Instant::now();
    let mut checksum = 0_i64;
    let mut overflow_count = 0_usize;
    for _ in 0..ITERATIONS {
        let result = mixed_dense_q88_q1616(
            black_box(weights),
            black_box(inputs),
            black_box(N_OUTPUTS),
            black_box(N_INPUTS),
        )
        .expect("deterministic benchmark dimensions must be valid");
        checksum ^= i64::from(result.outputs_q1616[0]);
        checksum ^= i64::from(result.outputs_q1616[N_OUTPUTS - 1]);
        overflow_count = result.overflow_count;
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;
    (elapsed_ns / ITERATIONS as f64, checksum, overflow_count)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn main() {
    let load_average_before = load_average();
    let weights = deterministic_weights();
    let inputs = deterministic_inputs();
    let mut ns_per_call = Vec::with_capacity(REPEATS);
    let mut checksum = 0_i64;
    let mut overflow_count = 0_usize;
    for _ in 0..REPEATS {
        let (ns, run_checksum, run_overflow_count) = run_once(&weights, &inputs);
        ns_per_call.push(ns);
        checksum ^= run_checksum;
        overflow_count = run_overflow_count;
    }
    let saturating_weights = vec![127_i16 << 8; N_INPUTS * N_OUTPUTS];
    let saturating_inputs = vec![32767_i32 << 16; N_INPUTS];
    let safe_envelope_report = mixed_dense_q88_q1616(&weights, &inputs, N_OUTPUTS, N_INPUTS)
        .expect("safe envelope dimensions must be valid")
        .precision_envelope_report();
    let saturating_probe =
        mixed_dense_q88_q1616(&saturating_weights, &saturating_inputs, N_OUTPUTS, N_INPUTS)
            .expect("saturating probe dimensions must be valid");
    let saturating_probe_overflow_count = saturating_probe.overflow_count;
    let saturating_probe_envelope_report = saturating_probe.precision_envelope_report();
    let mut sorted = ns_per_call.clone();
    let median_ns_per_call = median(&mut sorted);
    let min_ns_per_call = sorted[0];
    let max_ns_per_call = sorted[sorted.len() - 1];
    let timestamp_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after UNIX epoch")
        .as_secs();

    let results = ns_per_call
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(", ");
    let report = format!(
        concat!(
            "{{\n",
            "  \"benchmark\": \"mixed_dense_q88_q1616_64x32\",\n",
            "  \"benchmark_contract\": \"canonical_q88_weight_q1616_input\",\n",
            "  \"scale_per_tensor\": false,\n",
            "  \"language\": \"Rust\",\n",
            "  \"timestamp_unix\": {timestamp_unix_arg},\n",
            "  \"command\": \"taskset -c 10-11 cargo run --manifest-path engine/Cargo.toml --release --example bench_mixed_dense\",\n",
            "  \"rustc\": \"{rust_version_arg}\",\n",
            "  \"target_os\": \"{os}\",\n",
            "  \"target_arch\": \"{arch}\",\n",
            "  \"measurement_context\": {measurement_context},\n",
            "  \"n_inputs\": {n_inputs},\n",
            "  \"n_outputs\": {n_outputs},\n",
            "  \"iterations\": {iterations},\n",
            "  \"repeats\": {repeats},\n",
            "  \"median_ns_per_call\": {median_ns_per_call_arg:.3},\n",
            "  \"min_ns_per_call\": {min_ns_per_call_arg:.3},\n",
            "  \"max_ns_per_call\": {max_ns_per_call_arg:.3},\n",
            "  \"checksum\": {checksum_arg},\n",
            "  \"safe_overflow_count\": {overflow_count_arg},\n",
            "  \"safe_max_abs_bound_q1616\": {safe_max_abs_bound_q1616_arg},\n",
            "  \"safe_conservative_overflow_free\": {safe_conservative_overflow_free_arg},\n",
            "  \"safe_min_headroom_q1616\": {safe_min_headroom_q1616_arg},\n",
            "  \"saturating_probe_overflow_count\": {saturating_probe_overflow_count_arg},\n",
            "  \"saturating_probe_max_abs_bound_q1616\": {saturating_probe_max_abs_bound_q1616_arg},\n",
            "  \"saturating_probe_conservative_overflow_free\": {saturating_probe_conservative_overflow_free_arg},\n",
            "  \"results_ns_per_call\": [{results_arg}]\n",
            "}}\n"
        ),
        timestamp_unix_arg = timestamp_unix,
        rust_version_arg = rust_version(),
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        measurement_context = measurement_context_json(&load_average_before),
        n_inputs = N_INPUTS,
        n_outputs = N_OUTPUTS,
        iterations = ITERATIONS,
        repeats = REPEATS,
        median_ns_per_call_arg = median_ns_per_call,
        min_ns_per_call_arg = min_ns_per_call,
        max_ns_per_call_arg = max_ns_per_call,
        checksum_arg = checksum,
        overflow_count_arg = overflow_count,
        safe_max_abs_bound_q1616_arg = safe_envelope_report.max_abs_bound_q1616,
        safe_conservative_overflow_free_arg = safe_envelope_report.conservative_overflow_free,
        safe_min_headroom_q1616_arg = safe_envelope_report.min_headroom_q1616,
        saturating_probe_overflow_count_arg = saturating_probe_overflow_count,
        saturating_probe_max_abs_bound_q1616_arg = saturating_probe_envelope_report.max_abs_bound_q1616,
        saturating_probe_conservative_overflow_free_arg = saturating_probe_envelope_report.conservative_overflow_free,
        results_arg = results,
    );

    let path = Path::new("benchmarks/results/local_rust_2026-06-04_mixed_dense.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("benchmark result directory must be writable");
    }
    fs::write(path, &report).expect("benchmark result artefact must be writable");
    print!("{report}");
}
