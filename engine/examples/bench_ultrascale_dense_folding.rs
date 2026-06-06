// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 1996-2026 Miroslav Sotek. All rights reserved.
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - UltraScale+ dense folding Rust benchmark artefact writer

mod benchmark_context;

use benchmark_context::{load_average, measurement_context_json, rust_version};
use sc_neurocore_engine::ir::sv_target::{SkuKind, SvTarget};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const ITERATIONS: usize = 20_000;
const REPEATS: usize = 7;

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
    let target = SvTarget::zynq_ultrascale_plus(SkuKind::Zu3eg, 250);
    let plan = target
        .dense_fold_plan(64, 32)
        .expect("UltraScale+ target must produce a dense fold plan");
    let mut ns_per_plan = Vec::with_capacity(REPEATS);
    let mut checksum = 0_u32;
    for _ in 0..REPEATS {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let run_plan = black_box(&target)
                .dense_fold_plan(black_box(64), black_box(32))
                .expect("fold plan benchmark target must be valid");
            checksum ^= run_plan.dsp_per_cycle ^ run_plan.compute_cycles;
        }
        ns_per_plan.push(start.elapsed().as_nanos() as f64 / ITERATIONS as f64);
    }
    let mut sorted = ns_per_plan.clone();
    let median_ns_per_plan = median(&mut sorted);
    let timestamp_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after UNIX epoch")
        .as_secs();
    let report = format!(
        concat!(
            "{{\n",
            "  \"benchmark\": \"ultrascale_plus_dense_folding_contract\",\n",
            "  \"language\": \"Rust\",\n",
            "  \"timestamp_unix\": {timestamp_unix},\n",
            "  \"command\": \"taskset -c 10-11 cargo run --manifest-path engine/Cargo.toml --release --example bench_ultrascale_dense_folding\",\n",
            "  \"rustc\": \"{rust_version}\",\n",
            "  \"target_os\": \"{os}\",\n",
            "  \"target_arch\": \"{arch}\",\n",
            "  \"measurement_context\": {measurement_context},\n",
            "  \"iterations\": {iterations},\n",
            "  \"repeats\": {repeats},\n",
            "  \"median_ns_per_plan\": {median_ns_per_plan:.3},\n",
            "  \"min_ns_per_plan\": {min_ns_per_plan:.3},\n",
            "  \"max_ns_per_plan\": {max_ns_per_plan:.3},\n",
            "  \"checksum\": {checksum},\n",
            "  \"n_inputs\": {n_inputs},\n",
            "  \"n_outputs\": {n_outputs},\n",
            "  \"mac_count\": {mac_count},\n",
            "  \"dsp_budget\": {dsp_budget},\n",
            "  \"dsp_per_cycle\": {dsp_per_cycle},\n",
            "  \"output_parallelism\": {output_parallelism},\n",
            "  \"input_parallelism\": {input_parallelism},\n",
            "  \"input_fold_factor\": {input_fold_factor},\n",
            "  \"output_fold_factor\": {output_fold_factor},\n",
            "  \"compute_cycles\": {compute_cycles},\n",
            "  \"fold_required\": {fold_required},\n",
            "  \"fits_dsp_budget\": {fits_dsp_budget},\n",
            "  \"results_ns_per_plan\": [{results}]\n",
            "}}\n"
        ),
        timestamp_unix = timestamp_unix,
        rust_version = rust_version(),
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        measurement_context = measurement_context_json(&load_average_before),
        iterations = ITERATIONS,
        repeats = REPEATS,
        median_ns_per_plan = median_ns_per_plan,
        min_ns_per_plan = sorted[0],
        max_ns_per_plan = sorted[sorted.len() - 1],
        checksum = checksum,
        n_inputs = plan.n_inputs,
        n_outputs = plan.n_outputs,
        mac_count = plan.mac_count,
        dsp_budget = plan.dsp_budget,
        dsp_per_cycle = plan.dsp_per_cycle,
        output_parallelism = plan.output_parallelism,
        input_parallelism = plan.input_parallelism,
        input_fold_factor = plan.input_fold_factor,
        output_fold_factor = plan.output_fold_factor,
        compute_cycles = plan.compute_cycles,
        fold_required = plan.fold_required,
        fits_dsp_budget = plan.fits_dsp_budget,
        results = values_json(&ns_per_plan),
    );
    let path = Path::new("benchmarks/results/local_rust_2026-06-04_ultrascale_dense_folding.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("benchmark result directory must be writable");
    }
    fs::write(path, &report).expect("benchmark result artefact must be writable");
    print!("{report}");
}
