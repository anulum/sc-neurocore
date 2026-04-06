// SPDX-License-Identifier: AGPL-3.0-or-later
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! GPU vs CPU benchmark comparison for DenseLayer forward pass.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sc_neurocore_engine::gpu::GpuDenseLayer;
use sc_neurocore_engine::layer::DenseLayer;

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

criterion_group!(benches, bench_gpu_vs_cpu, bench_gpu_batch);
criterion_main!(benches);
