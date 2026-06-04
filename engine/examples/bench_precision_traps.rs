// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Precision trap Rust benchmark artefact writer

mod benchmark_context;

use benchmark_context::{load_average, measurement_context_json, rust_version};
use sc_neurocore_engine::ir::qformat::{
    block_floating_dense_q16, mixed_dense_q88_q1616, BlockFloatingMode,
};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const N_INPUTS: usize = 64;
const N_OUTPUTS: usize = 32;
const ITERATIONS: usize = 20_000;
const REPEATS: usize = 7;

fn overflow_weights_mixed() -> Vec<i16> {
    vec![i16::MAX; N_INPUTS * N_OUTPUTS]
}

fn overflow_mantissas_bfp() -> Vec<i16> {
    vec![i16::MAX; N_INPUTS * N_OUTPUTS]
}

fn overflow_inputs() -> Vec<i32> {
    vec![i32::MAX; N_INPUTS]
}

fn time_mixed_traps(weights: &[i16], inputs: &[i32]) -> (f64, i64, usize) {
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
        let report = result.precision_trap_report();
        checksum ^= report.saturated_max_count as i64;
        checksum ^= report.saturated_min_count as i64;
        overflow_count = report.overflow_count;
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;
    (elapsed_ns / ITERATIONS as f64, checksum, overflow_count)
}

fn time_bfp_traps(
    mantissas: &[i16],
    exponents: &[u8],
    inputs: &[i32],
    mode: BlockFloatingMode,
) -> (f64, i64, usize) {
    let start = Instant::now();
    let mut checksum = 0_i64;
    let mut overflow_count = 0_usize;
    for _ in 0..ITERATIONS {
        let result = block_floating_dense_q16(
            black_box(mantissas),
            black_box(exponents),
            black_box(inputs),
            black_box(N_OUTPUTS),
            black_box(N_INPUTS),
            black_box(mode),
        )
        .expect("deterministic benchmark dimensions must be valid");
        let report = result.precision_trap_report();
        checksum ^= report.saturated_max_count as i64;
        checksum ^= report.saturated_min_count as i64;
        overflow_count = report.overflow_count;
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;
    (elapsed_ns / ITERATIONS as f64, checksum, overflow_count)
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
    let weights = overflow_weights_mixed();
    let inputs = overflow_inputs();
    let mode = BlockFloatingMode::bfp16_e3_x32();
    let mantissas = overflow_mantissas_bfp();
    let exponents = vec![
        mode.exponent_code_max();
        (N_INPUTS * N_OUTPUTS + mode.block_size - 1) / mode.block_size
    ];

    let mut mixed_ns = Vec::with_capacity(REPEATS);
    let mut bfp_ns = Vec::with_capacity(REPEATS);
    let mut mixed_checksum = 0_i64;
    let mut bfp_checksum = 0_i64;
    let mut mixed_overflow_count = 0_usize;
    let mut bfp_overflow_count = 0_usize;

    for _ in 0..REPEATS {
        let (ns, checksum, overflow_count) = time_mixed_traps(&weights, &inputs);
        mixed_ns.push(ns);
        mixed_checksum ^= checksum;
        mixed_overflow_count = overflow_count;
    }
    for _ in 0..REPEATS {
        let (ns, checksum, overflow_count) = time_bfp_traps(&mantissas, &exponents, &inputs, mode);
        bfp_ns.push(ns);
        bfp_checksum ^= checksum;
        bfp_overflow_count = overflow_count;
    }

    let mut mixed_sorted = mixed_ns.clone();
    let mut bfp_sorted = bfp_ns.clone();
    let mixed_median_ns_per_call = median(&mut mixed_sorted);
    let bfp_median_ns_per_call = median(&mut bfp_sorted);
    let timestamp_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after UNIX epoch")
        .as_secs();

    let report = format!(
        concat!(
            "{{\n",
            "  \"benchmark\": \"precision_trap_reports_64x32\",\n",
            "  \"language\": \"Rust\",\n",
            "  \"timestamp_unix\": {timestamp_unix},\n",
            "  \"command\": \"taskset -c 10-11 cargo run --manifest-path engine/Cargo.toml --release --example bench_precision_traps\",\n",
            "  \"rustc\": \"{rust_version}\",\n",
            "  \"target_os\": \"{os}\",\n",
            "  \"target_arch\": \"{arch}\",\n",
            "  \"measurement_context\": {measurement_context},\n",
            "  \"n_inputs\": {n_inputs},\n",
            "  \"n_outputs\": {n_outputs},\n",
            "  \"iterations\": {iterations},\n",
            "  \"repeats\": {repeats},\n",
            "  \"mixed_trap_median_ns_per_call\": {mixed_median_ns_per_call:.3},\n",
            "  \"mixed_trap_min_ns_per_call\": {mixed_min_ns_per_call:.3},\n",
            "  \"mixed_trap_max_ns_per_call\": {mixed_max_ns_per_call:.3},\n",
            "  \"mixed_overflow_count\": {mixed_overflow_count},\n",
            "  \"mixed_checksum\": {mixed_checksum},\n",
            "  \"mixed_results_ns_per_call\": [{mixed_results}],\n",
            "  \"bfp_trap_median_ns_per_call\": {bfp_median_ns_per_call:.3},\n",
            "  \"bfp_trap_min_ns_per_call\": {bfp_min_ns_per_call:.3},\n",
            "  \"bfp_trap_max_ns_per_call\": {bfp_max_ns_per_call:.3},\n",
            "  \"bfp_overflow_count\": {bfp_overflow_count},\n",
            "  \"bfp_checksum\": {bfp_checksum},\n",
            "  \"bfp_results_ns_per_call\": [{bfp_results}]\n",
            "}}\n"
        ),
        timestamp_unix = timestamp_unix,
        rust_version = rust_version(),
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        measurement_context = measurement_context_json(&load_average_before),
        n_inputs = N_INPUTS,
        n_outputs = N_OUTPUTS,
        iterations = ITERATIONS,
        repeats = REPEATS,
        mixed_median_ns_per_call = mixed_median_ns_per_call,
        mixed_min_ns_per_call = mixed_sorted[0],
        mixed_max_ns_per_call = mixed_sorted[mixed_sorted.len() - 1],
        mixed_overflow_count = mixed_overflow_count,
        mixed_checksum = mixed_checksum,
        mixed_results = values_json(&mixed_ns),
        bfp_median_ns_per_call = bfp_median_ns_per_call,
        bfp_min_ns_per_call = bfp_sorted[0],
        bfp_max_ns_per_call = bfp_sorted[bfp_sorted.len() - 1],
        bfp_overflow_count = bfp_overflow_count,
        bfp_checksum = bfp_checksum,
        bfp_results = values_json(&bfp_ns),
    );

    let path = Path::new("benchmarks/results/local_rust_2026-06-04_precision_traps.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("benchmark result directory must be writable");
    }
    fs::write(path, &report).expect("benchmark result artefact must be writable");
    print!("{report}");
}
