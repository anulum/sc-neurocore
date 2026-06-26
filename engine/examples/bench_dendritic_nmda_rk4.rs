// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dendritic NMDA RK4 Rust benchmark artefact writer

#[allow(dead_code)]
mod benchmark_context;

use benchmark_context::{load_average, measurement_context_json, rust_version};
use sc_neurocore_engine::neurons::DendriticNMDANeuron;
use std::hint::black_box;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const STEPS: usize = 20_000;
const REPEATS: usize = 5;
const I_SOMA: f64 = 50.0;
const GLUTAMATE: f64 = 0.5;

fn run_once() -> (f64, i32, f64, f64) {
    let mut neuron = DendriticNMDANeuron::default();
    let mut spikes = 0_i32;
    let start = Instant::now();
    for _ in 0..STEPS {
        spikes += neuron.step(black_box(I_SOMA), black_box(GLUTAMATE));
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;
    (
        elapsed_ns / STEPS as f64,
        spikes,
        neuron.v_soma,
        neuron.v_dend,
    )
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn values_json(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() {
    let load_average_before = load_average();
    let timestamp_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after UNIX epoch")
        .as_secs();
    let mut ns_per_step = Vec::with_capacity(REPEATS);
    let mut spike_counts = Vec::with_capacity(REPEATS);
    let mut final_v_soma = Vec::with_capacity(REPEATS);
    let mut final_v_dend = Vec::with_capacity(REPEATS);

    for _ in 0..REPEATS {
        let (ns, spikes, v_soma, v_dend) = run_once();
        ns_per_step.push(ns);
        spike_counts.push(spikes);
        final_v_soma.push(v_soma);
        final_v_dend.push(v_dend);
    }

    let mut sorted = ns_per_step.clone();
    let median_ns_per_step = median(&mut sorted);
    println!(
        concat!(
            "{{\n",
            "  \"backend\": \"rust\",\n",
            "  \"timestamp_unix\": {timestamp_unix},\n",
            "  \"command\": \"cargo run --release --manifest-path engine/Cargo.toml --example bench_dendritic_nmda_rk4\",\n",
            "  \"rustc\": \"{rust_version}\",\n",
            "  \"target_os\": \"{os}\",\n",
            "  \"target_arch\": \"{arch}\",\n",
            "  \"measurement_context\": {measurement_context},\n",
            "  \"steps\": {steps},\n",
            "  \"repeats\": {repeats},\n",
            "  \"i_soma\": {i_soma},\n",
            "  \"glutamate\": {glutamate},\n",
            "  \"median_ns_per_step\": {median_ns_per_step:.6},\n",
            "  \"min_ns_per_step\": {min_ns_per_step:.6},\n",
            "  \"max_ns_per_step\": {max_ns_per_step:.6},\n",
            "  \"results_ns_per_step\": [{results}],\n",
            "  \"spike_counts\": {spike_counts:?},\n",
            "  \"final_v_soma\": {final_v_soma:?},\n",
            "  \"final_v_dend\": {final_v_dend:?}\n",
            "}}\n"
        ),
        timestamp_unix = timestamp_unix,
        rust_version = rust_version(),
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        measurement_context = measurement_context_json(&load_average_before),
        steps = STEPS,
        repeats = REPEATS,
        i_soma = I_SOMA,
        glutamate = GLUTAMATE,
        median_ns_per_step = median_ns_per_step,
        min_ns_per_step = sorted[0],
        max_ns_per_step = sorted[sorted.len() - 1],
        results = values_json(&ns_per_step),
        spike_counts = spike_counts,
        final_v_soma = final_v_soma,
        final_v_dend = final_v_dend,
    );
}
