// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — Rust benchmark runner for quantum cognition kernels
// Compile: rustc --edition 2021 -O bench_runner.rs -o bench_runner
// Run: ./bench_runner

#![allow(non_snake_case, non_camel_case_types)]

use std::time::Instant;

// ─── Inline the kernel (single-file benchmark) ───

#[derive(Debug, Clone)]
struct QuantumSpinChainMPS {
    sites: usize,
    correlation_length: f64,
    update_rate: f64,
    entanglement_map: Vec<f64>,
    measurement_count: u64,
}

impl QuantumSpinChainMPS {
    fn new(sites: usize) -> Self {
        let uniform = 1.0 / sites as f64;
        Self {
            sites,
            correlation_length: 2.0,
            update_rate: 0.1,
            entanglement_map: vec![uniform; sites],
            measurement_count: 0,
        }
    }

    fn apply_measurement(&mut self, site_idx: usize, intensity: f64) {
        let alpha = self.update_rate;
        let one_minus_alpha = 1.0 - alpha;
        let mut total: f64 = 0.0;
        for i in 0..self.sites {
            let distance = (i as f64 - site_idx as f64).abs();
            let influence = (-distance / self.correlation_length).exp() * intensity;
            self.entanglement_map[i] =
                one_minus_alpha * self.entanglement_map[i] + alpha * influence;
            total += self.entanglement_map[i];
        }
        if total > 0.0 {
            let inv_total = 1.0 / total;
            for val in self.entanglement_map.iter_mut() {
                *val *= inv_total;
            }
        }
        self.measurement_count += 1;
    }

    fn get_local_atp_telemetry(&self, site_idx: usize) -> f64 {
        self.entanglement_map[site_idx].clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone)]
struct Neuron {
    Vm: f64,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau_m: f64,
    atp_level: f64,
    atp_consumption: f64,
    total_spikes: u64,
    metabolic_failures: u64,
}

impl Neuron {
    fn new() -> Self {
        Self {
            Vm: -70.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 20.0,
            atp_level: 1.0,
            atp_consumption: 0.05,
            total_spikes: 0,
            metabolic_failures: 0,
        }
    }
}

fn batch_step(neurons: &mut [Neuron], pool: &mut QuantumSpinChainMPS, currents: &[f64]) -> usize {
    let mut spike_count = 0;
    for (i, n) in neurons.iter_mut().enumerate() {
        let eff = pool.get_local_atp_telemetry(i);
        n.atp_level = (n.atp_level + eff * 0.01).min(1.0);
        let i_pump = (eff - 0.5) * 2.0 * n.atp_level;
        let dv = (-(n.Vm - n.v_rest) + currents[i] + i_pump) / n.tau_m;
        n.Vm += dv;
        if n.Vm >= n.v_threshold {
            if n.atp_level >= n.atp_consumption {
                n.Vm = n.v_reset;
                n.atp_level -= n.atp_consumption;
                n.total_spikes += 1;
                spike_count += 1;
                pool.apply_measurement(i, 1.0);
            } else {
                n.Vm = n.v_threshold - 1.0;
                n.metabolic_failures += 1;
            }
        }
    }
    spike_count
}

fn main() {
    println!("SC-NeuroCore Quantum Cognition — Rust Benchmark Suite");
    println!("=====================================================");
    println!();

    // ── Benchmark 1: SpinPool only ──
    println!("--- Benchmark 1: apply_measurement ---");
    for &sites in &[32, 128, 256] {
        let n_steps = 10_000;
        let mut pool = QuantumSpinChainMPS::new(sites);
        let t0 = Instant::now();
        for step in 0..n_steps {
            pool.apply_measurement(step % sites, 1.0);
        }
        let elapsed = t0.elapsed();
        let us_per_call = elapsed.as_secs_f64() / n_steps as f64 * 1e6;
        println!(
            "  sites={:3}  steps={}  time={:.1}ms  per_call={:.3}µs",
            sites,
            n_steps,
            elapsed.as_secs_f64() * 1000.0,
            us_per_call
        );
    }

    // ── Benchmark 2: Population step ──
    println!("\n--- Benchmark 2: batch_step_population ---");
    for &(n_neurons, n_steps) in &[(32, 1000), (128, 1000), (256, 500), (512, 200), (1024, 100)] {
        let mut pool = QuantumSpinChainMPS::new(n_neurons);
        let mut neurons: Vec<Neuron> = (0..n_neurons).map(|_| Neuron::new()).collect();
        let mut currents = vec![25.0_f64; n_neurons];
        let mut total_spikes: u64 = 0;

        let t0 = Instant::now();
        for step in 0..n_steps {
            for i in 0..n_neurons {
                currents[i] = 20.0 + 10.0 * ((step * 7 + i * 3) as f64 * 0.01).sin();
            }
            total_spikes += batch_step(&mut neurons, &mut pool, &currents) as u64;
        }
        let elapsed = t0.elapsed();
        let total_neuron_steps = n_neurons * n_steps;
        let us_per = elapsed.as_secs_f64() / total_neuron_steps as f64 * 1e6;
        let throughput = total_neuron_steps as f64 / elapsed.as_secs_f64();
        println!("  neurons={:4}  steps={:4}  time={:>8.1}ms  per_neuron_step={:.3}µs  throughput={:.0}/s  spikes={}",
                 n_neurons, n_steps, elapsed.as_secs_f64() * 1000.0, us_per, throughput, total_spikes);
    }

    println!("\nRust kernel: ALL BENCHMARKS COMPLETE");
}
