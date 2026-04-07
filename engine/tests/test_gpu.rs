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

use sc_neurocore_engine::gpu::{is_available, GpuDenseLayer};
use sc_neurocore_engine::layer::DenseLayer;

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
