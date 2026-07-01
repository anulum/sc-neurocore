// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Integration tests for GPU compute backend.
//!
//! These tests require:
//! - `cargo test --features gpu`
//! - A Vulkan/Metal/DX12-capable GPU

#![cfg(feature = "gpu")]

use sc_neurocore_engine::gpu::{
    is_available, GpuDenseLayer, GpuIzhikevichBatch, GpuKuramoto, GpuLifBatch,
};
use sc_neurocore_engine::layer::DenseLayer;
use sc_neurocore_engine::neuron::{FixedPointLif, Izhikevich};
use sc_neurocore_engine::scpn::kuramoto::KuramotoSolver;

/// Skip-guard: all tests in this file require a real GPU.
fn require_gpu() -> bool {
    if !is_available() {
        eprintln!("No GPU available — skipping GPU tests");
        return false;
    }
    true
}

#[test]
fn gpu_available_check() {
    // This test always passes — it just reports availability.
    let avail = is_available();
    eprintln!("GPU available: {avail}");
}

#[test]
fn gpu_dense_layer_creation() {
    if !require_gpu() {
        return;
    }
    let layer = GpuDenseLayer::try_new(64, 32, 1024, 42, 1);
    assert!(layer.is_some(), "GpuDenseLayer::try_new should succeed");
    let layer = layer.unwrap();
    assert_eq!(layer.cpu.n_inputs, 64);
    assert_eq!(layer.cpu.n_neurons, 32);
    assert!(!layer.gpu_name().is_empty());
    eprintln!("GPU adapter: {}", layer.gpu_name());
}

#[test]
fn gpu_forward_single_output_shape() {
    if !require_gpu() {
        return;
    }
    let n_inputs = 64;
    let n_neurons = 32;
    let layer = GpuDenseLayer::try_new(n_inputs, n_neurons, 1024, 42, 1).unwrap();
    let inputs: Vec<f64> = vec![0.5; n_inputs];
    let output = layer.forward_gpu(&inputs, 42);
    assert_eq!(
        output.len(),
        n_neurons,
        "Output length must equal n_neurons"
    );
}

#[test]
fn gpu_forward_single_values_non_negative() {
    if !require_gpu() {
        return;
    }
    let n_inputs = 128;
    let n_neurons = 64;
    let layer = GpuDenseLayer::try_new(n_inputs, n_neurons, 1024, 42, 1).unwrap();
    let inputs: Vec<f64> = (0..n_inputs)
        .map(|i| (i as f64) / (n_inputs as f64))
        .collect();
    let output = layer.forward_gpu(&inputs, 42);
    // Output range is [0, n_inputs] (not [0,1]) because it is the
    // length-normalised dot product of bitstreams across all inputs.
    let upper_bound = n_inputs as f64;
    for (i, &v) in output.iter().enumerate() {
        assert!(v >= 0.0, "Output[{i}] = {v} is negative");
        assert!(
            v <= upper_bound,
            "Output[{i}] = {v} exceeds upper bound {upper_bound}"
        );
    }
}

#[test]
fn gpu_forward_deterministic() {
    if !require_gpu() {
        return;
    }
    let n_inputs = 64;
    let n_neurons = 32;
    let layer = GpuDenseLayer::try_new(n_inputs, n_neurons, 1024, 42, 4).unwrap();
    let inputs: Vec<f64> = vec![0.5; n_inputs];
    let out1 = layer.forward_gpu(&inputs, 42);
    let out2 = layer.forward_gpu(&inputs, 42);
    assert_eq!(out1, out2, "Same seed must produce identical output");
}

#[test]
fn gpu_vs_cpu_approximate_agreement() {
    if !require_gpu() {
        return;
    }
    let n_inputs = 64;
    let n_neurons = 32;
    let length = 4096; // Longer bitstream for tighter convergence.
    let seed = 42u64;

    let cpu_layer = DenseLayer::new(n_inputs, n_neurons, length, seed);
    let gpu_layer = GpuDenseLayer::try_new(n_inputs, n_neurons, length, seed, 1).unwrap();

    let inputs: Vec<f64> = vec![0.5; n_inputs];
    let cpu_out = cpu_layer.forward_fast(&inputs, seed).unwrap();
    let gpu_out = gpu_layer.forward_gpu(&inputs, seed);

    assert_eq!(cpu_out.len(), gpu_out.len());

    // GPU uses f32 internally + different PRNG (Philox vs ChaCha), so outputs
    // are statistically similar but not bitwise identical. Output range is
    // [0, n_inputs], so we use relative tolerance.
    let max_diff: f64 = cpu_out
        .iter()
        .zip(gpu_out.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f64, f64::max);

    let mean_output: f64 = cpu_out.iter().sum::<f64>() / cpu_out.len() as f64;

    eprintln!("GPU vs CPU max |diff| = {max_diff:.6}, mean = {mean_output:.4}");
    // Tolerance: 5% of n_inputs (stochastic variance + different PRNGs).
    let tolerance = n_inputs as f64 * 0.05;
    assert!(
        max_diff < tolerance,
        "GPU and CPU outputs differ by {max_diff:.6} (tolerance {tolerance:.3})"
    );
}

#[test]
fn gpu_batch_forward_shape() {
    if !require_gpu() {
        return;
    }
    let n_inputs = 64;
    let n_neurons = 32;
    let n_samples = 8;
    let layer = GpuDenseLayer::try_new(n_inputs, n_neurons, 1024, 42, n_samples).unwrap();
    let inputs: Vec<f64> = vec![0.5; n_samples * n_inputs];
    let output = layer.forward_batch_gpu(&inputs, n_samples, 42);
    assert_eq!(
        output.len(),
        n_samples * n_neurons,
        "Batch output length must be n_samples × n_neurons"
    );
}

#[test]
fn gpu_batch_values_non_negative() {
    if !require_gpu() {
        return;
    }
    let n_inputs = 128;
    let n_neurons = 64;
    let n_samples = 16;
    let layer = GpuDenseLayer::try_new(n_inputs, n_neurons, 1024, 42, n_samples).unwrap();
    let inputs: Vec<f64> = (0..n_samples * n_inputs)
        .map(|i| (i as f64) / (n_samples * n_inputs) as f64)
        .collect();
    let output = layer.forward_batch_gpu(&inputs, n_samples, 42);
    let upper_bound = n_inputs as f64;
    for (i, &v) in output.iter().enumerate() {
        assert!(v >= 0.0, "Batch output[{i}] = {v} is negative");
        assert!(
            v <= upper_bound,
            "Batch output[{i}] = {v} exceeds upper bound {upper_bound}"
        );
    }
}

#[test]
fn gpu_zero_inputs_produce_near_zero_output() {
    if !require_gpu() {
        return;
    }
    let n_inputs = 64;
    let n_neurons = 32;
    let layer = GpuDenseLayer::try_new(n_inputs, n_neurons, 1024, 42, 1).unwrap();
    let inputs: Vec<f64> = vec![0.0; n_inputs];
    let output = layer.forward_gpu(&inputs, 42);
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v < 0.01,
            "With all-zero inputs, output[{i}] = {v} should be near zero"
        );
    }
}

#[test]
fn gpu_one_inputs_match_cpu() {
    if !require_gpu() {
        return;
    }
    let n_inputs = 32;
    let n_neurons = 16;
    let length = 4096;
    let seed = 99u64;

    let cpu_layer = DenseLayer::new(n_inputs, n_neurons, length, seed);
    let gpu_layer = GpuDenseLayer::try_new(n_inputs, n_neurons, length, seed, 1).unwrap();

    let inputs: Vec<f64> = vec![1.0; n_inputs];
    let cpu_out = cpu_layer.forward_fast(&inputs, seed).unwrap();
    let gpu_out = gpu_layer.forward_gpu(&inputs, seed);

    let max_diff: f64 = cpu_out
        .iter()
        .zip(gpu_out.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f64, f64::max);

    eprintln!("All-ones GPU vs CPU max |diff| = {max_diff:.6}");
    // All-ones inputs are deterministic (no PRNG), diff should be near zero.
    let tolerance = n_inputs as f64 * 0.01;
    assert!(
        max_diff < tolerance,
        "All-ones: GPU and CPU differ by {max_diff:.6} (tolerance {tolerance:.3})"
    );
}

// ---- Fixed-point LIF batch: GPU vs CPU bit-exact parity ----

/// CPU reference: run `n_neurons` LIF neurons (constant current, zero noise) for
/// `n_steps`, mirroring `batch_lif_run_multi`. Returns row-major spikes/voltages.
#[allow(clippy::too_many_arguments)]
fn cpu_lif_batch(
    n_neurons: usize,
    n_steps: usize,
    leak_k: i16,
    gain_k: i16,
    currents: &[i32],
    data_width: u32,
    fraction: u32,
    v_rest: i16,
    v_reset: i16,
    v_threshold: i16,
    refractory_period: i32,
) -> (Vec<i32>, Vec<i32>) {
    let mut spikes = vec![0i32; n_neurons * n_steps];
    let mut voltages = vec![0i32; n_neurons * n_steps];
    for neuron in 0..n_neurons {
        let mut lif = FixedPointLif::new(
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        );
        for step in 0..n_steps {
            let (s, v) = lif.step(leak_k, gain_k, currents[neuron] as i16, 0);
            spikes[neuron * n_steps + step] = s;
            voltages[neuron * n_steps + step] = v as i32;
        }
    }
    (spikes, voltages)
}

#[test]
fn gpu_lif_creation() {
    if !require_gpu() {
        return;
    }
    let batch = GpuLifBatch::try_new();
    assert!(batch.is_some(), "GpuLifBatch::try_new should succeed");
    assert!(!batch.unwrap().gpu_name().is_empty());
}

#[test]
fn gpu_lif_bit_exact_with_cpu() {
    if !require_gpu() {
        return;
    }
    let n_neurons = 64;
    let n_steps = 50;
    let leak_k: i16 = 16;
    let gain_k: i16 = 256; // Q8.8 unit gain.
    let (data_width, fraction) = (16u32, 8u32);
    let (v_rest, v_reset, v_threshold, refractory) = (0i16, 0i16, 256i16, 2i32);
    // Current sweep: some neurons stay sub-threshold, some spike and enter refractory.
    let currents: Vec<i32> = (0..n_neurons as i32).map(|n| n * 10).collect();

    let gpu = GpuLifBatch::try_new().unwrap();
    let result = gpu.run(
        n_neurons,
        n_steps,
        leak_k,
        gain_k,
        &currents,
        data_width,
        fraction,
        v_rest,
        v_reset,
        v_threshold,
        refractory,
        0,
    );
    let (cpu_spikes, cpu_volts) = cpu_lif_batch(
        n_neurons,
        n_steps,
        leak_k,
        gain_k,
        &currents,
        data_width,
        fraction,
        v_rest,
        v_reset,
        v_threshold,
        refractory,
    );

    assert_eq!(result.spikes, cpu_spikes, "spikes must be bit-exact");
    assert_eq!(result.voltages, cpu_volts, "voltages must be bit-exact");
    // Sanity: the workload actually spikes (otherwise parity is vacuous).
    assert!(
        result.spikes.contains(&1),
        "test workload should produce spikes"
    );
}

#[test]
fn gpu_lif_negative_rest_bit_exact() {
    if !require_gpu() {
        return;
    }
    // Negative resting potential exercises the signed leak/diff path.
    let n_neurons = 32;
    let n_steps = 40;
    let currents: Vec<i32> = (0..n_neurons as i32).map(|n| 200 + n * 20).collect();
    let gpu = GpuLifBatch::try_new().unwrap();
    let result = gpu.run(
        n_neurons, n_steps, 32, 256, &currents, 16, 8, -64, -64, 300, 3, 0,
    );
    let (cpu_spikes, cpu_volts) = cpu_lif_batch(
        n_neurons, n_steps, 32, 256, &currents, 16, 8, -64, -64, 300, 3,
    );
    assert_eq!(result.spikes, cpu_spikes);
    assert_eq!(result.voltages, cpu_volts);
}

#[test]
fn gpu_lif_shape_and_empty() {
    if !require_gpu() {
        return;
    }
    let gpu = GpuLifBatch::try_new().unwrap();
    let result = gpu.run(8, 5, 16, 256, &[100; 8], 16, 8, 0, 0, 256, 2, 0);
    assert_eq!(result.spikes.len(), 40);
    assert_eq!(result.voltages.len(), 40);
    // Zero-sized batch returns empty without dispatching.
    let empty = gpu.run(0, 5, 16, 256, &[], 16, 8, 0, 0, 256, 2, 0);
    assert!(empty.spikes.is_empty());
}

#[test]
fn gpu_different_seeds_produce_different_output() {
    if !require_gpu() {
        return;
    }
    let n_inputs = 64;
    let n_neurons = 32;
    let layer = GpuDenseLayer::try_new(n_inputs, n_neurons, 1024, 42, 1).unwrap();
    let inputs: Vec<f64> = vec![0.5; n_inputs];
    let out1 = layer.forward_gpu(&inputs, 1);
    let out2 = layer.forward_gpu(&inputs, 2);
    // With different seeds, at least some outputs should differ.
    let any_diff = out1
        .iter()
        .zip(out2.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(any_diff, "Different seeds should produce different outputs");
}

// ── GPU Kuramoto oscillator kernel ──────────────────────────────────────

/// Deterministic all-to-all Kuramoto system used for GPU↔CPU parity checks.
fn kuramoto_system(n: usize, k: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let omega: Vec<f64> = (0..n).map(|i| 0.2 * ((i as f64) * 0.3).sin()).collect();
    // Uniform positive coupling K on every edge — drives synchronisation.
    let coupling: Vec<f64> = vec![k; n * n];
    let phases: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5).collect();
    (omega, coupling, phases)
}

/// Kuramoto order parameter R = |mean(exp(i·θ))|.
fn order_parameter(phases: &[f64]) -> f64 {
    let n_inv = 1.0 / phases.len() as f64;
    let c: f64 = phases.iter().map(|t| t.cos()).sum::<f64>() * n_inv;
    let s: f64 = phases.iter().map(|t| t.sin()).sum::<f64>() * n_inv;
    (c * c + s * s).sqrt()
}

/// Smallest circular distance between two angles, in [0, π].
fn circular_distance(a: f64, b: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let d = (a - b).rem_euclid(two_pi);
    d.min(two_pi - d)
}

#[test]
fn gpu_kuramoto_creation() {
    if !require_gpu() {
        return;
    }
    let solver = GpuKuramoto::try_new();
    assert!(
        solver.is_some(),
        "GpuKuramoto::try_new should succeed on a GPU"
    );
    assert!(!solver.unwrap().gpu_name().is_empty());
}

#[test]
fn gpu_kuramoto_empty_and_zero_steps() {
    if !require_gpu() {
        return;
    }
    let gpu = GpuKuramoto::try_new().unwrap();
    assert!(gpu.run(0, &[], &[], &[], 10, 0.01).is_empty());
    // Zero steps returns the initial phases unchanged.
    let phases = vec![0.1_f32, 0.2, 0.3];
    let coupling = vec![0.0_f32; 9];
    let out = gpu.run(3, &[0.0; 3], &coupling, &phases, 0, 0.01);
    assert_eq!(out, phases);
}

#[test]
fn gpu_kuramoto_matches_cpu_within_tolerance() {
    if !require_gpu() {
        return;
    }
    let n = 128;
    let dt = 0.01;
    let steps = 200;
    let (omega, coupling, phases) = kuramoto_system(n, 1.5);

    // CPU oracle: noise-free baseline (seed=0 disables the PRNG term).
    let mut solver = KuramotoSolver::new(omega.clone(), coupling.clone(), phases.clone(), 0.0);
    solver.run(steps, dt, 0);
    let cpu_phases = solver.get_phases().to_vec();

    // GPU: same system in f32.
    let gpu = GpuKuramoto::try_new().unwrap();
    let omega_f: Vec<f32> = omega.iter().map(|&x| x as f32).collect();
    let coupling_f: Vec<f32> = coupling.iter().map(|&x| x as f32).collect();
    let phases_f: Vec<f32> = phases.iter().map(|&x| x as f32).collect();
    let gpu_phases: Vec<f64> = gpu
        .run(n, &omega_f, &coupling_f, &phases_f, steps, dt as f32)
        .iter()
        .map(|&x| x as f64)
        .collect();

    assert_eq!(gpu_phases.len(), n);
    // Every GPU phase must be finite and wrapped into [0, 2π).
    for &p in &gpu_phases {
        assert!(
            p.is_finite() && (0.0..std::f64::consts::TAU).contains(&p),
            "phase out of range: {p}"
        );
    }
    // The order parameter (a smooth aggregate) must agree closely.
    let r_cpu = order_parameter(&cpu_phases);
    let r_gpu = order_parameter(&gpu_phases);
    assert!(
        (r_cpu - r_gpu).abs() < 1e-3,
        "order parameter diverged: CPU {r_cpu} vs GPU {r_gpu}"
    );
    // Per-oscillator circular agreement (f32 vs f64 + libm sin over 200 steps).
    let max_circ = gpu_phases
        .iter()
        .zip(cpu_phases.iter())
        .map(|(&g, &c)| circular_distance(g, c))
        .fold(0.0_f64, f64::max);
    assert!(
        max_circ < 1e-2,
        "max circular phase divergence {max_circ} too large"
    );
}

#[test]
fn gpu_kuramoto_synchronises_under_strong_coupling() {
    if !require_gpu() {
        return;
    }
    // Strong uniform coupling with identical natural frequencies must pull the
    // order parameter towards 1 (phase locking) — a behavioural sanity check that
    // the coupling term is wired, not just that two numbers match.
    let n = 64;
    let coupling = vec![4.0_f32; n * n];
    let omega = vec![0.0_f32; n];
    let phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.4).collect();
    let r_start = {
        let p: Vec<f64> = phases.iter().map(|&x| x as f64).collect();
        order_parameter(&p)
    };
    let gpu = GpuKuramoto::try_new().unwrap();
    let out: Vec<f64> = gpu
        .run(n, &omega, &coupling, &phases, 500, 0.02)
        .iter()
        .map(|&x| x as f64)
        .collect();
    let r_end = order_parameter(&out);
    assert!(
        r_end > r_start,
        "coupling should raise R: {r_start} -> {r_end}"
    );
    assert!(
        r_end > 0.9,
        "strong coupling should nearly synchronise: R = {r_end}"
    );
}

// ── GPU Izhikevich neuron kernel ────────────────────────────────────────

/// CPU oracle: run `n_neurons` independent `Izhikevich` neurons for `n_steps`
/// with per-neuron constant currents, returning row-major `[n_neurons × n_steps]`
/// spikes (i32) and voltages (f32, cast from the f64 model). Mirrors the GPU
/// kernel's per-neuron contract and shared `(a, b, c, d, dt)` parameters.
#[allow(clippy::too_many_arguments)]
fn cpu_izhikevich_batch(
    n_neurons: usize,
    n_steps: usize,
    currents: &[f32],
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    dt: f32,
) -> (Vec<i32>, Vec<f32>) {
    let mut spikes = Vec::with_capacity(n_neurons * n_steps);
    let mut voltages = Vec::with_capacity(n_neurons * n_steps);
    for &current in currents.iter().take(n_neurons) {
        let mut neuron = Izhikevich::new(a as f64, b as f64, c as f64, d as f64, dt as f64);
        for _ in 0..n_steps {
            let spike = neuron.step(current as f64);
            spikes.push(spike);
            voltages.push(neuron.v as f32);
        }
    }
    (spikes, voltages)
}

#[test]
fn gpu_izhikevich_creation() {
    if !require_gpu() {
        return;
    }
    let batch = GpuIzhikevichBatch::try_new();
    assert!(
        batch.is_some(),
        "GpuIzhikevichBatch::try_new should succeed"
    );
    assert!(!batch.unwrap().gpu_name().is_empty());
}

#[test]
fn gpu_izhikevich_shape_and_empty() {
    if !require_gpu() {
        return;
    }
    let gpu = GpuIzhikevichBatch::try_new().unwrap();
    // Regular-spiking parameters; shape must be exactly n_neurons × n_steps.
    let result = gpu.run(8, 5, &[5.0; 8], 0.02, 0.2, -65.0, 8.0, 1.0, 30.0);
    assert_eq!(result.spikes.len(), 40);
    assert_eq!(result.voltages.len(), 40);
    // Zero-sized batch returns empty without dispatching.
    let empty = gpu.run(0, 5, &[], 0.02, 0.2, -65.0, 8.0, 1.0, 30.0);
    assert!(empty.spikes.is_empty());
    assert!(empty.voltages.is_empty());
    // Zero steps also returns empty rows.
    let no_steps = gpu.run(4, 0, &[5.0; 4], 0.02, 0.2, -65.0, 8.0, 1.0, 30.0);
    assert!(no_steps.spikes.is_empty());
}

#[test]
fn gpu_izhikevich_subthreshold_trace_matches_cpu() {
    if !require_gpu() {
        return;
    }
    // A weak constant current keeps the regular-spiking neuron below threshold, so
    // there is no spike/reset discontinuity — the full voltage trace is smooth and
    // the f32 GPU integration must track the f64 CPU model tightly.
    let n_neurons = 16;
    let n_steps = 100;
    let (a, b, c, d, dt) = (0.02, 0.2, -65.0, 8.0, 1.0);
    // 2 pA sits comfortably below the regular-spiking rheobase (~4 pA), clear of
    // the saddle-node ghost where coarse Euler would overshoot into a spike.
    let currents = vec![2.0_f32; n_neurons];

    let gpu = GpuIzhikevichBatch::try_new().unwrap();
    let result = gpu.run(n_neurons, n_steps, &currents, a, b, c, d, dt, 30.0);
    let (cpu_spikes, cpu_volts) =
        cpu_izhikevich_batch(n_neurons, n_steps, &currents, a, b, c, d, dt);

    // No spikes expected — parity of the raw trace is meaningful.
    assert!(
        !cpu_spikes.contains(&1),
        "chosen current should stay sub-threshold on the CPU oracle"
    );
    assert_eq!(result.spikes, cpu_spikes, "sub-threshold spikes must match");
    let max_dv = result
        .voltages
        .iter()
        .zip(cpu_volts.iter())
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_dv < 0.5,
        "sub-threshold voltage trace diverged: max |Δv| = {max_dv} mV"
    );
}

#[test]
fn gpu_izhikevich_spike_count_matches_cpu() {
    if !require_gpu() {
        return;
    }
    // A supra-threshold current drives regular spiking. f32 vs f64 can shift the
    // exact reset step by a tick, so the spike count agrees within a small margin
    // rather than bit-exactly (unlike the fixed-point LIF kernel).
    let n_neurons = 32;
    let n_steps = 300;
    let (a, b, c, d, dt) = (0.02, 0.2, -65.0, 8.0, 1.0);
    let currents = vec![10.0_f32; n_neurons];

    let gpu = GpuIzhikevichBatch::try_new().unwrap();
    let result = gpu.run(n_neurons, n_steps, &currents, a, b, c, d, dt, 30.0);
    let (cpu_spikes, _) = cpu_izhikevich_batch(n_neurons, n_steps, &currents, a, b, c, d, dt);

    let gpu_count: i32 = result.spikes.iter().sum();
    let cpu_count: i32 = cpu_spikes.iter().sum();
    assert!(cpu_count > 0, "workload should spike on the CPU oracle");
    assert!(
        (gpu_count - cpu_count).abs() <= 2,
        "spike count diverged: GPU {gpu_count} vs CPU {cpu_count}"
    );
    // Every voltage after a reset must sit at or below the peak threshold.
    assert!(
        result.voltages.iter().all(|&v| v <= 30.0),
        "no voltage should exceed the peak threshold after reset"
    );
}

#[test]
fn gpu_izhikevich_firing_rate_monotonic_in_current() {
    if !require_gpu() {
        return;
    }
    // Behavioural check: a stronger drive must produce at least as many spikes —
    // confirms the current term is wired, not just that two numbers coincide.
    let (a, b, c, d, dt) = (0.02, 0.2, -65.0, 8.0, 1.0);
    let n_steps = 300;
    let gpu = GpuIzhikevichBatch::try_new().unwrap();
    let count_for = |i: f32| -> i32 {
        gpu.run(1, n_steps, &[i], a, b, c, d, dt, 30.0)
            .spikes
            .iter()
            .sum()
    };
    let low = count_for(5.0);
    let mid = count_for(10.0);
    let high = count_for(20.0);
    assert!(
        low <= mid && mid <= high,
        "firing rate should be monotonic in current: {low} <= {mid} <= {high}"
    );
    assert!(high > 0, "strong drive should spike");
}
