// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! GPU vs CPU benchmark comparison for DenseLayer forward pass and LIF batches.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;
use sc_neurocore_engine::gpu::{GpuDenseLayer, GpuLifBatch};
use sc_neurocore_engine::layer::DenseLayer;
use sc_neurocore_engine::neuron::FixedPointLif;

fn bench_gpu_vs_cpu(c: &mut Criterion) {
    let configs: &[(usize, usize)] = &[(64, 32), (128, 64), (256, 128), (512, 256), (1000, 500)];

    let mut group = c.benchmark_group("gpu_vs_cpu_forward");

    for &(n_inputs, n_neurons) in configs {
        let length = 1024;
        let seed = 42u64;
        let inputs: Vec<f64> = (0..n_inputs)
            .map(|i| (i as f64) / (n_inputs as f64))
            .collect();

        // CPU benchmark.
        let cpu_layer = DenseLayer::new(n_inputs, n_neurons, length, seed);
        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{n_inputs}x{n_neurons}")),
            &inputs,
            |b, inp| {
                b.iter(|| cpu_layer.forward_fast(inp, seed));
            },
        );

        // GPU benchmark.
        if let Some(gpu_layer) = GpuDenseLayer::try_new(n_inputs, n_neurons, length, seed, 1) {
            group.bench_with_input(
                BenchmarkId::new("gpu", format!("{n_inputs}x{n_neurons}")),
                &inputs,
                |b, inp| {
                    b.iter(|| gpu_layer.forward_gpu(inp, seed));
                },
            );
        }
    }

    group.finish();
}

fn bench_gpu_batch(c: &mut Criterion) {
    let n_inputs = 256;
    let n_neurons = 128;
    let length = 1024;
    let seed = 42u64;
    let batch_sizes: &[usize] = &[1, 16, 64, 256];

    let mut group = c.benchmark_group("gpu_batch_scaling");

    for &batch in batch_sizes {
        let inputs: Vec<f64> = (0..batch * n_inputs)
            .map(|i| (i as f64) / (batch * n_inputs) as f64)
            .collect();

        // CPU batch.
        let cpu_layer = DenseLayer::new(n_inputs, n_neurons, length, seed);
        group.bench_with_input(BenchmarkId::new("cpu", batch), &inputs, |b, inp| {
            let mut out = vec![0.0f64; batch * n_neurons];
            b.iter(|| cpu_layer.forward_batch_into(inp, batch, seed, &mut out));
        });

        // GPU batch.
        if let Some(gpu_layer) = GpuDenseLayer::try_new(n_inputs, n_neurons, length, seed, batch) {
            group.bench_with_input(BenchmarkId::new("gpu", batch), &inputs, |b, inp| {
                b.iter(|| gpu_layer.forward_batch_gpu(inp, batch, seed));
            });
        }
    }

    group.finish();
}

/// CPU reference for the LIF batch: rayon-parallel over neurons, mirroring the
/// `batch_lif_run_multi` hot path. Returns total spikes to defeat dead-code elision.
fn cpu_lif_batch(n_neurons: usize, n_steps: usize, currents: &[i32]) -> usize {
    (0..n_neurons)
        .into_par_iter()
        .map(|n| {
            let mut lif = FixedPointLif::new(16, 8, 0, 0, 256, 2);
            let mut spikes = 0usize;
            for _ in 0..n_steps {
                let (s, _) = lif.step(16, 256, currents[n] as i16, 0);
                spikes += s as usize;
            }
            spikes
        })
        .sum()
}

/// LIF batch: GPU (one thread per neuron, time loop in-kernel) vs rayon CPU, across
/// neuron counts — used to locate the crossover where the GPU overtakes the CPU.
fn bench_gpu_lif(c: &mut Criterion) {
    let sizes: &[usize] = &[256, 1024, 4096, 16_384, 65_536];
    let n_steps = 100;
    let mut group = c.benchmark_group("gpu_lif_batch");

    for &n in sizes {
        let currents: Vec<i32> = (0..n as i32).map(|i| 100 + (i % 500)).collect();

        group.bench_with_input(BenchmarkId::new("cpu", n), &currents, |b, curr| {
            b.iter(|| cpu_lif_batch(n, n_steps, curr));
        });

        if let Some(gpu) = GpuLifBatch::try_new() {
            group.bench_with_input(BenchmarkId::new("gpu", n), &currents, |b, curr| {
                b.iter(|| gpu.run(n, n_steps, 16, 256, curr, 16, 8, 0, 0, 256, 2, 0));
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_gpu_vs_cpu, bench_gpu_batch, bench_gpu_lif);
criterion_main!(benches);
