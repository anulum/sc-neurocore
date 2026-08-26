// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained unit-capacitance respiratory Rust benchmark writer

#[allow(dead_code)]
mod benchmark_context;

use benchmark_context::{load_average, measurement_context_json, rust_version};
use sc_neurocore_engine::neurons::SCUnitCapacitanceRespiratoryNeuron;
use std::hint::black_box;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const STEPS: usize = 20_000;
const REPEATS: usize = 5;
const CURRENT: f64 = 20.0;

fn run_once() -> (f64, i32, [f64; 3]) {
    let mut neuron = SCUnitCapacitanceRespiratoryNeuron::default();
    let mut spikes = 0_i32;
    let started = Instant::now();
    for _ in 0..STEPS {
        spikes += neuron.step(black_box(CURRENT));
    }
    let elapsed_ns = started.elapsed().as_nanos() as f64;
    (
        elapsed_ns / STEPS as f64,
        spikes,
        [neuron.inner.v, neuron.inner.n, neuron.inner.h_nap],
    )
}

fn main() {
    let load_average_before = load_average();
    let timestamp_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after UNIX epoch")
        .as_secs();
    let mut timings = Vec::with_capacity(REPEATS);
    let mut spike_counts = Vec::with_capacity(REPEATS);
    let mut final_states = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let (timing, spikes, state) = run_once();
        timings.push(timing);
        spike_counts.push(spikes);
        final_states.push(state);
    }
    let mut sorted = timings.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    println!(
        concat!(
            "{{\n",
            "  \"backend\": \"rust\",\n",
            "  \"timestamp_unix\": {timestamp_unix},\n",
            "  \"rustc\": \"{rustc}\",\n",
            "  \"measurement_context\": {measurement_context},\n",
            "  \"steps\": {steps},\n",
            "  \"repeats\": {repeats},\n",
            "  \"current\": {current},\n",
            "  \"median_ns_per_step\": {median:.6},\n",
            "  \"min_ns_per_step\": {minimum:.6},\n",
            "  \"max_ns_per_step\": {maximum:.6},\n",
            "  \"results_ns_per_step\": {timings:?},\n",
            "  \"spike_counts\": {spike_counts:?},\n",
            "  \"final_states\": {final_states:?}\n",
            "}}\n"
        ),
        timestamp_unix = timestamp_unix,
        rustc = rust_version(),
        measurement_context = measurement_context_json(&load_average_before),
        steps = STEPS,
        repeats = REPEATS,
        current = CURRENT,
        median = sorted[REPEATS / 2],
        minimum = sorted[0],
        maximum = sorted[REPEATS - 1],
        timings = timings,
        spike_counts = spike_counts,
        final_states = final_states,
    );
}
