// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Precision envelope Rust benchmark artefact writer

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

fn deterministic_mixed_weights() -> Vec<i16> {
    (0..(N_INPUTS * N_OUTPUTS))
        .map(|i| (((i * 17 + 11) % 513) as i32 - 256) as i16)
        .collect()
}

fn deterministic_bfp_mantissas() -> Vec<i16> {
    (0..(N_INPUTS * N_OUTPUTS))
        .map(|i| (((i * 23 + 3) % 1025) as i32 - 512) as i16)
        .collect()
}

fn deterministic_inputs() -> Vec<i32> {
    (0..N_INPUTS)
        .map(|i| (((i * 19 + 5) % 257) as i32 - 128) << 6)
        .collect()
}

fn underflow_weights_mixed() -> Vec<i16> {
    let mut weights = vec![0_i16; N_INPUTS * N_OUTPUTS];
    for output_idx in 0..N_OUTPUTS {
        weights[output_idx * N_INPUTS] = 1_i16;
    }
    weights
}

fn underflow_mantissas_bfp() -> Vec<i16> {
    let mut mantissas = vec![0_i16; N_INPUTS * N_OUTPUTS];
    for output_idx in 0..N_OUTPUTS {
        mantissas[output_idx * N_INPUTS] = 1_i16;
    }
    mantissas
}

fn underflow_inputs() -> Vec<i32> {
    vec![1_i32; N_INPUTS]
}

fn time_mixed_envelopes(weights: &[i16], inputs: &[i32]) -> (f64, i64, bool, bool, i64) {
    let start = Instant::now();
    let mut checksum = 0_i64;
    let mut conservative_safe = false;
    let mut underflow_free = false;
    let mut max_abs_bound = 0_i64;
    for _ in 0..ITERATIONS {
        let result = mixed_dense_q88_q1616(
            black_box(weights),
            black_box(inputs),
            black_box(N_OUTPUTS),
            black_box(N_INPUTS),
        )
        .expect("deterministic benchmark dimensions must be valid");
        let report = result.precision_envelope_report();
        checksum ^= report.max_abs_bound_q1616;
        checksum ^= report.min_headroom_q1616;
        checksum ^= report.underflow_count as i64;
        conservative_safe = report.conservative_overflow_free;
        underflow_free = report.observed_underflow_free;
        max_abs_bound = report.max_abs_bound_q1616;
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;
    (
        elapsed_ns / ITERATIONS as f64,
        checksum,
        conservative_safe,
        underflow_free,
        max_abs_bound,
    )
}

fn time_bfp_envelopes(
    mantissas: &[i16],
    exponents: &[u8],
    inputs: &[i32],
    mode: BlockFloatingMode,
) -> (f64, i64, bool, bool, i64) {
    let start = Instant::now();
    let mut checksum = 0_i64;
    let mut conservative_safe = false;
    let mut underflow_free = false;
    let mut max_abs_bound = 0_i64;
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
        let report = result.precision_envelope_report();
        checksum ^= report.max_abs_bound_q1616;
        checksum ^= report.min_headroom_q1616;
        checksum ^= report.underflow_count as i64;
        conservative_safe = report.conservative_overflow_free;
        underflow_free = report.observed_underflow_free;
        max_abs_bound = report.max_abs_bound_q1616;
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;
    (
        elapsed_ns / ITERATIONS as f64,
        checksum,
        conservative_safe,
        underflow_free,
        max_abs_bound,
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
    let weights = deterministic_mixed_weights();
    let inputs = deterministic_inputs();
    let mode = BlockFloatingMode::bfp16_e3_x32();
    let mantissas = deterministic_bfp_mantissas();
    let exponents = vec![
        mode.exponent_bias() as u8;
        (N_INPUTS * N_OUTPUTS + mode.block_size - 1) / mode.block_size
    ];
    let underflow_weights = underflow_weights_mixed();
    let underflow_mantissas = underflow_mantissas_bfp();
    let underflow_exponents =
        vec![0_u8; (N_INPUTS * N_OUTPUTS + mode.block_size - 1) / mode.block_size];
    let underflow_inputs = underflow_inputs();

    let mut mixed_ns = Vec::with_capacity(REPEATS);
    let mut bfp_ns = Vec::with_capacity(REPEATS);
    let mut mixed_checksum = 0_i64;
    let mut bfp_checksum = 0_i64;
    let mut mixed_conservative_safe = false;
    let mut bfp_conservative_safe = false;
    let mut mixed_underflow_free = false;
    let mut bfp_underflow_free = false;
    let mut mixed_max_abs_bound = 0_i64;
    let mut bfp_max_abs_bound = 0_i64;

    for _ in 0..REPEATS {
        let (ns, checksum, conservative_safe, underflow_free, max_abs_bound) =
            time_mixed_envelopes(&weights, &inputs);
        mixed_ns.push(ns);
        mixed_checksum ^= checksum;
        mixed_conservative_safe = conservative_safe;
        mixed_underflow_free = underflow_free;
        mixed_max_abs_bound = max_abs_bound;
    }
    for _ in 0..REPEATS {
        let (ns, checksum, conservative_safe, underflow_free, max_abs_bound) =
            time_bfp_envelopes(&mantissas, &exponents, &inputs, mode);
        bfp_ns.push(ns);
        bfp_checksum ^= checksum;
        bfp_conservative_safe = conservative_safe;
        bfp_underflow_free = underflow_free;
        bfp_max_abs_bound = max_abs_bound;
    }
    let mixed_underflow_count =
        mixed_dense_q88_q1616(&underflow_weights, &underflow_inputs, N_OUTPUTS, N_INPUTS)
            .expect("underflow probe dimensions must be valid")
            .precision_envelope_report()
            .underflow_count;
    let bfp_underflow_count = block_floating_dense_q16(
        &underflow_mantissas,
        &underflow_exponents,
        &underflow_inputs,
        N_OUTPUTS,
        N_INPUTS,
        mode,
    )
    .expect("underflow probe dimensions must be valid")
    .precision_envelope_report()
    .underflow_count;
    let mixed_safe_report = mixed_dense_q88_q1616(&weights, &inputs, N_OUTPUTS, N_INPUTS)
        .expect("safe envelope dimensions must be valid")
        .precision_envelope_report();
    let bfp_safe_report =
        block_floating_dense_q16(&mantissas, &exponents, &inputs, N_OUTPUTS, N_INPUTS, mode)
            .expect("safe envelope dimensions must be valid")
            .precision_envelope_report();

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
            "  \"benchmark\": \"precision_envelope_reports_64x32\",\n",
            "  \"language\": \"Rust\",\n",
            "  \"timestamp_unix\": {timestamp_unix},\n",
            "  \"command\": \"taskset -c 8-9 cargo run --manifest-path engine/Cargo.toml --release --example bench_precision_envelopes\",\n",
            "  \"rustc\": \"{rust_version}\",\n",
            "  \"target_os\": \"{os}\",\n",
            "  \"target_arch\": \"{arch}\",\n",
            "  \"measurement_context\": {measurement_context},\n",
            "  \"n_inputs\": {n_inputs},\n",
            "  \"n_outputs\": {n_outputs},\n",
            "  \"iterations\": {iterations},\n",
            "  \"repeats\": {repeats},\n",
            "  \"mixed_envelope_median_ns_per_call\": {mixed_median_ns_per_call:.3},\n",
            "  \"mixed_envelope_min_ns_per_call\": {mixed_min_ns_per_call:.3},\n",
            "  \"mixed_envelope_max_ns_per_call\": {mixed_max_ns_per_call:.3},\n",
            "  \"mixed_conservative_overflow_free\": {mixed_conservative_safe},\n",
            "  \"mixed_observed_underflow_free\": {mixed_underflow_free},\n",
            "  \"mixed_max_abs_bound_q1616\": {mixed_max_abs_bound},\n",
            "  \"mixed_required_total_bits_q1616\": {mixed_required_total_bits_q1616},\n",
            "  \"mixed_required_integer_bits_q1616\": {mixed_required_integer_bits_q1616},\n",
            "  \"mixed_width_headroom_bits_q1616\": {mixed_width_headroom_bits_q1616},\n",
            "  \"mixed_saturation_required\": {mixed_saturation_required},\n",
            "  \"mixed_static_overflow_proven_safe\": {mixed_static_overflow_proven_safe},\n",
            "  \"mixed_underflow_count\": {mixed_underflow_count},\n",
            "  \"mixed_checksum\": {mixed_checksum},\n",
            "  \"mixed_results_ns_per_call\": [{mixed_results}],\n",
            "  \"bfp_envelope_median_ns_per_call\": {bfp_median_ns_per_call:.3},\n",
            "  \"bfp_envelope_min_ns_per_call\": {bfp_min_ns_per_call:.3},\n",
            "  \"bfp_envelope_max_ns_per_call\": {bfp_max_ns_per_call:.3},\n",
            "  \"bfp_conservative_overflow_free\": {bfp_conservative_safe},\n",
            "  \"bfp_observed_underflow_free\": {bfp_underflow_free},\n",
            "  \"bfp_max_abs_bound_q1616\": {bfp_max_abs_bound},\n",
            "  \"bfp_required_total_bits_q1616\": {bfp_required_total_bits_q1616},\n",
            "  \"bfp_required_integer_bits_q1616\": {bfp_required_integer_bits_q1616},\n",
            "  \"bfp_width_headroom_bits_q1616\": {bfp_width_headroom_bits_q1616},\n",
            "  \"bfp_saturation_required\": {bfp_saturation_required},\n",
            "  \"bfp_static_overflow_proven_safe\": {bfp_static_overflow_proven_safe},\n",
            "  \"bfp_underflow_count\": {bfp_underflow_count},\n",
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
        mixed_conservative_safe = mixed_conservative_safe,
        mixed_underflow_free = mixed_underflow_free,
        mixed_max_abs_bound = mixed_max_abs_bound,
        mixed_required_total_bits_q1616 = mixed_safe_report.required_total_bits_q1616,
        mixed_required_integer_bits_q1616 = mixed_safe_report.required_integer_bits_q1616,
        mixed_width_headroom_bits_q1616 = mixed_safe_report.width_headroom_bits_q1616,
        mixed_saturation_required = mixed_safe_report.saturation_required,
        mixed_static_overflow_proven_safe = mixed_safe_report.static_overflow_proven_safe,
        mixed_underflow_count = mixed_underflow_count,
        mixed_checksum = mixed_checksum,
        mixed_results = values_json(&mixed_ns),
        bfp_median_ns_per_call = bfp_median_ns_per_call,
        bfp_min_ns_per_call = bfp_sorted[0],
        bfp_max_ns_per_call = bfp_sorted[bfp_sorted.len() - 1],
        bfp_conservative_safe = bfp_conservative_safe,
        bfp_underflow_free = bfp_underflow_free,
        bfp_max_abs_bound = bfp_max_abs_bound,
        bfp_required_total_bits_q1616 = bfp_safe_report.required_total_bits_q1616,
        bfp_required_integer_bits_q1616 = bfp_safe_report.required_integer_bits_q1616,
        bfp_width_headroom_bits_q1616 = bfp_safe_report.width_headroom_bits_q1616,
        bfp_saturation_required = bfp_safe_report.saturation_required,
        bfp_static_overflow_proven_safe = bfp_safe_report.static_overflow_proven_safe,
        bfp_underflow_count = bfp_underflow_count,
        bfp_checksum = bfp_checksum,
        bfp_results = values_json(&bfp_ns),
    );

    let path = Path::new("benchmarks/results/local_rust_2026-06-04_precision_envelopes.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("benchmark result directory must be writable");
    }
    fs::write(path, &report).expect("benchmark result artefact must be writable");
    print!("{report}");
}
