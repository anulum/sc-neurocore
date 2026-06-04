// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (C) 1996-2026 Miroslav Sotek. All rights reserved.
// Copyright (C) 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - UltraScale+ Rust benchmark artefact writer

mod benchmark_context;

use benchmark_context::{load_average, measurement_context_json, rust_version};
use sc_neurocore_engine::ir::builder::ScGraphBuilder;
use sc_neurocore_engine::ir::emit_sv::emit_systemverilog_with_target;
use sc_neurocore_engine::ir::graph::{DenseParams, ScConst, ScType};
use sc_neurocore_engine::ir::sv_target::{SkuKind, SvTarget};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const N_INPUTS: usize = 64;
const N_OUTPUTS: usize = 32;
const ITERATIONS: usize = 2_000;
const REPEATS: usize = 7;

fn build_graph() -> sc_neurocore_engine::ir::graph::ScGraph {
    let mut builder = ScGraphBuilder::new("ultrascale_bench_dense");
    let inputs = builder.input(
        "inputs",
        ScType::Vec {
            element: Box::new(ScType::FixedPoint { width: 16, frac: 8 }),
            count: N_INPUTS,
        },
    );
    let weights = builder.constant(
        ScConst::I64Vec(vec![128; N_INPUTS * N_OUTPUTS]),
        ScType::Vec {
            element: Box::new(ScType::FixedPoint { width: 16, frac: 8 }),
            count: N_INPUTS * N_OUTPUTS,
        },
    );
    let leak = builder.constant(ScConst::I64(16), ScType::FixedPoint { width: 16, frac: 8 });
    let gain = builder.constant(ScConst::I64(1), ScType::FixedPoint { width: 16, frac: 8 });
    let result = builder.dense_forward(
        inputs,
        weights,
        leak,
        gain,
        DenseParams {
            n_inputs: N_INPUTS,
            n_neurons: N_OUTPUTS,
            ..DenseParams::default()
        },
    );
    builder.output("spikes", result);
    builder.build()
}

fn time_emit(graph: &sc_neurocore_engine::ir::graph::ScGraph) -> (f64, usize, u32, u32, u32) {
    let target = SvTarget::zynq_ultrascale_plus(SkuKind::Zu3eg, 250);
    let start = Instant::now();
    let mut checksum = 0_usize;
    let mut dsp_estimated = 0_u32;
    let mut bram_36k_estimated = 0_u32;
    let mut lut_estimated = 0_u32;
    for _ in 0..ITERATIONS {
        let (sv, report) = emit_systemverilog_with_target(black_box(graph), target.clone())
            .expect("UltraScale+ emission benchmark graph must be valid");
        checksum ^= sv.len();
        dsp_estimated = report.dsp_estimated;
        bram_36k_estimated = report.bram_36k_estimated;
        lut_estimated = report.lut_estimated;
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;
    (
        elapsed_ns / ITERATIONS as f64,
        checksum,
        dsp_estimated,
        bram_36k_estimated,
        lut_estimated,
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
    let graph = build_graph();
    let resource_report = SvTarget::zynq_ultrascale_plus(SkuKind::Zu3eg, 250).estimate_graph(&graph);
    let mut ns_per_emit = Vec::with_capacity(REPEATS);
    let mut checksum = 0_usize;
    let mut dsp_estimated = 0_u32;
    let mut bram_36k_estimated = 0_u32;
    let mut lut_estimated = 0_u32;

    for _ in 0..REPEATS {
        let (ns, run_checksum, run_dsp, run_bram, run_lut) = time_emit(&graph);
        ns_per_emit.push(ns);
        checksum ^= run_checksum;
        dsp_estimated = run_dsp;
        bram_36k_estimated = run_bram;
        lut_estimated = run_lut;
    }

    let mut sorted = ns_per_emit.clone();
    let median_ns_per_emit = median(&mut sorted);
    let timestamp_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after UNIX epoch")
        .as_secs();

    let report = format!(
        concat!(
            "{{\n",
            "  \"benchmark\": \"ultrascale_plus_target_contract\",\n",
            "  \"language\": \"Rust\",\n",
            "  \"timestamp_unix\": {timestamp_unix},\n",
            "  \"command\": \"taskset -c 10-11 cargo run --manifest-path engine/Cargo.toml --release --example bench_ultrascale_plus_target\",\n",
            "  \"rustc\": \"{rust_version}\",\n",
            "  \"target_os\": \"{os}\",\n",
            "  \"target_arch\": \"{arch}\",\n",
            "  \"measurement_context\": {measurement_context},\n",
            "  \"sku\": \"ZU3EG\",\n",
            "  \"device_part\": \"xczu3eg-sbva484-1-e\",\n",
            "  \"dsp_primitive\": \"DSP48E2\",\n",
            "  \"n_inputs\": {n_inputs},\n",
            "  \"n_outputs\": {n_outputs},\n",
            "  \"iterations\": {iterations},\n",
            "  \"repeats\": {repeats},\n",
            "  \"median_ns_per_emit\": {median_ns_per_emit:.3},\n",
            "  \"min_ns_per_emit\": {min_ns_per_emit:.3},\n",
            "  \"max_ns_per_emit\": {max_ns_per_emit:.3},\n",
            "  \"checksum\": {checksum},\n",
            "  \"dsp_estimated\": {dsp_estimated},\n",
            "  \"fits_dsp_budget\": {fits_dsp_budget},\n",
            "  \"bram_36k_estimated\": {bram_36k_estimated},\n",
            "  \"fits_bram_budget\": {fits_bram_budget},\n",
            "  \"lut_estimated\": {lut_estimated},\n",
            "  \"dsp_budget\": 360,\n",
            "  \"bram_36k_budget\": 216,\n",
            "  \"uram_budget\": 0,\n",
            "  \"fits_uram_budget\": {fits_uram_budget},\n",
            "  \"results_ns_per_emit\": [{results}]\n",
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
        median_ns_per_emit = median_ns_per_emit,
        min_ns_per_emit = sorted[0],
        max_ns_per_emit = sorted[sorted.len() - 1],
        checksum = checksum,
        dsp_estimated = dsp_estimated,
        fits_dsp_budget = resource_report.fits_dsp_budget,
        bram_36k_estimated = bram_36k_estimated,
        fits_bram_budget = resource_report.fits_bram_budget,
        lut_estimated = lut_estimated,
        fits_uram_budget = resource_report.fits_uram_budget,
        results = values_json(&ns_per_emit),
    );

    let path = Path::new("benchmarks/results/local_rust_2026-06-04_ultrascale_plus_target.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("benchmark result directory must be writable");
    }
    fs::write(path, &report).expect("benchmark result artefact must be writable");
    print!("{report}");
}
