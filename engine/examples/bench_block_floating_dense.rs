// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Block-floating dense Rust benchmark artefact writer

use sc_neurocore_engine::ir::qformat::{block_floating_dense_q16, BlockFloatingMode};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const N_INPUTS: usize = 64;
const N_OUTPUTS: usize = 32;
const ITERATIONS: usize = 20_000;
const REPEATS: usize = 7;

fn deterministic_mantissas() -> Vec<i16> {
    (0..(N_INPUTS * N_OUTPUTS))
        .map(|i| (((i * 23 + 3) % 1025) as i32 - 512) as i16)
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

fn main() {
    let mode = BlockFloatingMode::bfp16_e3_x32();
    let mantissas = deterministic_mantissas();
    let exponents = vec![
        mode.exponent_bias() as u8;
        (N_INPUTS * N_OUTPUTS + mode.block_size - 1) / mode.block_size
    ];
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
    let saturating_mantissas = vec![i16::MAX; N_INPUTS * N_OUTPUTS];
    let saturating_inputs = vec![32767_i32 << 16; N_INPUTS];
    let saturating_probe_overflow_count = block_floating_dense_q16(
        &saturating_mantissas,
        &exponents,
        &saturating_inputs,
        N_OUTPUTS,
        N_INPUTS,
        mode,
    )
    .expect("saturating probe dimensions must be valid")
    .overflow_count;
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
            "  \"command\": \"cargo run --manifest-path engine/Cargo.toml --release --example bench_block_floating_dense\",\n",
            "  \"target_os\": \"{os}\",\n",
            "  \"target_arch\": \"{arch}\",\n",
            "  \"n_inputs\": {n_inputs},\n",
            "  \"n_outputs\": {n_outputs},\n",
            "  \"mantissa_bits\": {mantissa_bits},\n",
            "  \"exponent_bits\": {exponent_bits},\n",
            "  \"block_size\": {block_size},\n",
            "  \"iterations\": {iterations},\n",
            "  \"repeats\": {repeats},\n",
            "  \"median_ns_per_call\": {median_ns_per_call:.3},\n",
            "  \"min_ns_per_call\": {min_ns_per_call:.3},\n",
            "  \"max_ns_per_call\": {max_ns_per_call:.3},\n",
            "  \"checksum\": {checksum},\n",
            "  \"safe_overflow_count\": {overflow_count},\n",
            "  \"saturating_probe_overflow_count\": {saturating_probe_overflow_count},\n",
            "  \"results_ns_per_call\": [{results}]\n",
            "}}\n"
        ),
        timestamp_unix = timestamp_unix,
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        n_inputs = N_INPUTS,
        n_outputs = N_OUTPUTS,
        mantissa_bits = mode.mantissa_bits,
        exponent_bits = mode.exponent_bits,
        block_size = mode.block_size,
        iterations = ITERATIONS,
        repeats = REPEATS,
        median_ns_per_call = median_ns_per_call,
        min_ns_per_call = min_ns_per_call,
        max_ns_per_call = max_ns_per_call,
        checksum = checksum,
        overflow_count = overflow_count,
        saturating_probe_overflow_count = saturating_probe_overflow_count,
        results = results,
    );

    let path = Path::new("benchmarks/results/local_rust_2026-06-04_block_floating_dense.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("benchmark result directory must be writable");
    }
    fs::write(path, &report).expect("benchmark result artefact must be writable");
    print!("{report}");
}
