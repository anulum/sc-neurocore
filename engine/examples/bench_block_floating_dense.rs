// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Block-floating dense Rust benchmark artefact writer

mod benchmark_context;

use benchmark_context::{load_average, measurement_context_json, rust_version};
use sc_neurocore_engine::ir::qformat::{block_floating_dense_q16, BlockFloatingMode};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const N_INPUTS: usize = 64;
const N_OUTPUTS: usize = 32;
const ITERATIONS: usize = 20_000;
const REPEATS: usize = 7;

fn round_div_nearest_even(value: i32, divisor: i32) -> i16 {
    let sign = if value < 0 { -1 } else { 1 };
    let magnitude = value.abs();
    let quotient = magnitude / divisor;
    let remainder = magnitude % divisor;
    let rounded_magnitude = if remainder * 2 < divisor {
        quotient
    } else if remainder * 2 > divisor {
        quotient + 1
    } else if quotient % 2 == 0 {
        quotient
    } else {
        quotient + 1
    };
    (sign * rounded_magnitude) as i16
}

fn deterministic_mantissas() -> Vec<i16> {
    (0..(N_INPUTS * N_OUTPUTS))
        .map(|i| {
            let raw_weight_code = ((i * 23 + 3) % 1025) as i32 - 512;
            round_div_nearest_even(raw_weight_code, 64)
        })
        .collect()
}

fn deterministic_inputs() -> Vec<i32> {
    (0..N_INPUTS)
        .map(|i| (((i * 19 + 5) % 257) as i32 - 128) << 8)
        .collect()
}

fn run_once(
    mode: BlockFloatingMode,
    mantissas: &[i16],
    exponents: &[u8],
    inputs: &[i32],
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

fn exponent_edge_sweep_json() -> String {
    let mode = BlockFloatingMode::new(16, 3, 2).expect("BFP16E3X2 is valid");
    let mantissas = [
        1_i16,
        -2_i16,
        i16::MAX,
        -i16::MAX,
        -3_i16,
        4_i16,
        -i16::MAX,
        i16::MAX,
    ];
    let exponents = [
        0_u8,
        mode.exponent_code_max(),
        0_u8,
        mode.exponent_code_max(),
    ];
    let inputs = [32768_i32, -16384_i32, 1_i32, -1_i32];
    let safe = block_floating_dense_q16(&mantissas, &exponents, &inputs, 2, 4, mode)
        .expect("seeded exponent-edge dimensions are valid");
    let safe_envelope = safe.precision_envelope_report();

    let saturating_mantissas = [i16::MAX, i16::MAX];
    let saturating_exponents = [mode.exponent_code_max()];
    let saturating_inputs = [32767_i32 << 16, 32767_i32 << 16];
    let saturating = block_floating_dense_q16(
        &saturating_mantissas,
        &saturating_exponents,
        &saturating_inputs,
        1,
        2,
        mode,
    )
    .expect("max-exponent trap dimensions are valid");
    let saturating_envelope = saturating.precision_envelope_report();

    format!(
        concat!(
            "{{",
            "\"format\":\"BFP16E3X2\",",
            "\"safe_exponent_codes\":[0,{max_code},0,{max_code}],",
            "\"safe_output_codes_q1616\":[{safe_output_0},{safe_output_1}],",
            "\"safe_overflow_count\":{safe_overflow_count},",
            "\"safe_underflow_count\":{safe_underflow_count},",
            "\"safe_max_abs_bound_q1616\":{safe_max_abs_bound_q1616},",
            "\"safe_min_headroom_q1616\":{safe_min_headroom_q1616},",
            "\"safe_conservative_overflow_free\":{safe_conservative_overflow_free},",
            "\"max_exponent_saturating_codes_q1616\":[{saturating_code}],",
            "\"max_exponent_saturating_exponent_codes\":[{max_code}],",
            "\"max_exponent_saturating_overflow_count\":{saturating_overflow_count},",
            "\"max_exponent_saturating_underflow_count\":{saturating_underflow_count},",
            "\"max_exponent_saturating_conservative_overflow_free\":{saturating_conservative_overflow_free},",
            "\"max_exponent_saturating_max_abs_bound_q1616\":{saturating_max_abs_bound_q1616}",
            "}}"
        ),
        max_code = mode.exponent_code_max(),
        safe_output_0 = safe.outputs_q1616[0],
        safe_output_1 = safe.outputs_q1616[1],
        safe_overflow_count = safe.overflow_count,
        safe_underflow_count = safe.underflow_count,
        safe_max_abs_bound_q1616 = safe_envelope.max_abs_bound_q1616,
        safe_min_headroom_q1616 = safe_envelope.min_headroom_q1616,
        safe_conservative_overflow_free = safe_envelope.conservative_overflow_free,
        saturating_code = saturating.outputs_q1616[0],
        saturating_overflow_count = saturating.overflow_count,
        saturating_underflow_count = saturating.underflow_count,
        saturating_conservative_overflow_free = saturating_envelope.conservative_overflow_free,
        saturating_max_abs_bound_q1616 = saturating_envelope.max_abs_bound_q1616,
    )
}

fn main() {
    let load_average_before = load_average();
    let mode = BlockFloatingMode::bfp16_e3_x32();
    let mantissas = deterministic_mantissas();
    let exponents = vec![0_u8; (N_INPUTS * N_OUTPUTS + mode.block_size - 1) / mode.block_size];
    let inputs = deterministic_inputs();
    let mut ns_per_call = Vec::with_capacity(REPEATS);
    let mut checksum = 0_i64;
    let mut overflow_count = 0_usize;
    for _ in 0..REPEATS {
        let (ns, run_checksum, run_overflow_count) =
            run_once(mode, &mantissas, &exponents, &inputs);
        ns_per_call.push(ns);
        checksum ^= run_checksum;
        overflow_count = run_overflow_count;
    }
    let saturating_mantissas = vec![16_384_i16; N_INPUTS * N_OUTPUTS];
    let saturating_exponents =
        vec![2_u8; (N_INPUTS * N_OUTPUTS + mode.block_size - 1) / mode.block_size];
    let saturating_inputs = vec![32767_i32 << 16; N_INPUTS];
    let safe_envelope_report =
        block_floating_dense_q16(&mantissas, &exponents, &inputs, N_OUTPUTS, N_INPUTS, mode)
            .expect("safe envelope dimensions must be valid")
            .precision_envelope_report();
    let mantissa_checksum = mantissas.iter().map(|&value| i64::from(value)).sum::<i64>();
    let exponent_checksum = exponents.iter().map(|&value| i64::from(value)).sum::<i64>();
    let exponent_code_min = exponents.iter().copied().min().unwrap_or(0);
    let exponent_code_max = exponents.iter().copied().max().unwrap_or(0);
    let parameter_count = N_INPUTS * N_OUTPUTS;
    let block_exponent_count = mode
        .block_exponent_count(parameter_count)
        .expect("benchmark parameter count must have a valid BFP layout");
    let saturating_probe = block_floating_dense_q16(
        &saturating_mantissas,
        &saturating_exponents,
        &saturating_inputs,
        N_OUTPUTS,
        N_INPUTS,
        mode,
    )
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
            "  \"benchmark\": \"block_floating_dense_q16_64x32\",\n",
            "  \"language\": \"Rust\",\n",
            "  \"timestamp_unix\": {timestamp_unix},\n",
            "  \"command\": \"taskset -c 8-9 cargo run --manifest-path engine/Cargo.toml --release --example bench_block_floating_dense\",\n",
            "  \"rustc\": \"{rust_version}\",\n",
            "  \"target_os\": \"{os}\",\n",
            "  \"target_arch\": \"{arch}\",\n",
            "  \"measurement_context\": {measurement_context},\n",
            "  \"n_inputs\": {n_inputs},\n",
            "  \"n_outputs\": {n_outputs},\n",
            "  \"mantissa_bits\": {mantissa_bits},\n",
            "  \"exponent_bits\": {exponent_bits},\n",
            "  \"block_size\": {block_size},\n",
            "  \"parameter_count\": {parameter_count},\n",
            "  \"block_exponent_count\": {block_exponent_count},\n",
            "  \"iterations\": {iterations},\n",
            "  \"repeats\": {repeats},\n",
            "  \"median_ns_per_call\": {median_ns_per_call:.3},\n",
            "  \"min_ns_per_call\": {min_ns_per_call:.3},\n",
            "  \"max_ns_per_call\": {max_ns_per_call:.3},\n",
            "  \"checksum\": {checksum},\n",
            "  \"safe_overflow_count\": {overflow_count},\n",
            "  \"safe_max_abs_bound_q1616\": {safe_max_abs_bound_q1616},\n",
            "  \"safe_conservative_overflow_free\": {safe_conservative_overflow_free},\n",
            "  \"safe_min_headroom_q1616\": {safe_min_headroom_q1616},\n",
            "  \"mantissa_checksum\": {mantissa_checksum},\n",
            "  \"exponent_checksum\": {exponent_checksum},\n",
            "  \"exponent_code_min\": {exponent_code_min},\n",
            "  \"exponent_code_max\": {exponent_code_max},\n",
            "  \"saturating_probe_overflow_count\": {saturating_probe_overflow_count},\n",
            "  \"saturating_probe_max_abs_bound_q1616\": {saturating_probe_max_abs_bound_q1616},\n",
            "  \"saturating_probe_conservative_overflow_free\": {saturating_probe_conservative_overflow_free},\n",
            "  \"exponent_edge_sweep\": {exponent_edge_sweep},\n",
            "  \"results_ns_per_call\": [{results}]\n",
            "}}\n"
        ),
        timestamp_unix = timestamp_unix,
        rust_version = rust_version(),
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        measurement_context = measurement_context_json(&load_average_before),
        n_inputs = N_INPUTS,
        n_outputs = N_OUTPUTS,
        mantissa_bits = mode.mantissa_bits,
        exponent_bits = mode.exponent_bits,
        block_size = mode.block_size,
        parameter_count = parameter_count,
        block_exponent_count = block_exponent_count,
        iterations = ITERATIONS,
        repeats = REPEATS,
        median_ns_per_call = median_ns_per_call,
        min_ns_per_call = min_ns_per_call,
        max_ns_per_call = max_ns_per_call,
        checksum = checksum,
        overflow_count = overflow_count,
        safe_max_abs_bound_q1616 = safe_envelope_report.max_abs_bound_q1616,
        safe_conservative_overflow_free = safe_envelope_report.conservative_overflow_free,
        safe_min_headroom_q1616 = safe_envelope_report.min_headroom_q1616,
        mantissa_checksum = mantissa_checksum,
        exponent_checksum = exponent_checksum,
        exponent_code_min = exponent_code_min,
        exponent_code_max = exponent_code_max,
        saturating_probe_overflow_count = saturating_probe_overflow_count,
        saturating_probe_max_abs_bound_q1616 = saturating_probe_envelope_report.max_abs_bound_q1616,
        saturating_probe_conservative_overflow_free = saturating_probe_envelope_report.conservative_overflow_free,
        exponent_edge_sweep = exponent_edge_sweep_json(),
        results = results,
    );

    let path = Path::new("benchmarks/results/local_rust_2026-06-04_block_floating_dense.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("benchmark result directory must be writable");
    }
    fs::write(path, &report).expect("benchmark result artefact must be writable");
    print!("{report}");
}
