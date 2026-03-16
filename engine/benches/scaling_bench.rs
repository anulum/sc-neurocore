// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scaling Bench

//
// Scaling benchmarks: measures wall-clock vs neuron/node count for
// Kuramoto, GNN forward, and Dense SC layer at multiple sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sc_neurocore_engine::graph::StochasticGraphLayer;
use sc_neurocore_engine::layer::DenseLayer;
use sc_neurocore_engine::scpn::KuramotoSolver;
use std::hint::black_box;

fn bench_kuramoto_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("kuramoto_scaling");
    group.sample_size(10);

    for &n in &[50, 100, 200, 500, 1000] {
        let omega = vec![1.0; n];
        let coupling = vec![0.3; n * n];
        let phases: Vec<f64> = (0..n)
            .map(|i| 2.0 * std::f64::consts::PI * (i as f64) / (n as f64))
            .collect();

        group.bench_with_input(BenchmarkId::new("kuramoto_1000steps", n), &n, |b, _| {
            b.iter(|| {
                let mut solver =
                    KuramotoSolver::new(omega.clone(), coupling.clone(), phases.clone(), 0.0);
                black_box(solver.run(1000, 0.01, 42));
            })
        });
    }
    group.finish();
}

fn bench_gnn_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gnn_scaling");
    group.sample_size(10);

    for &n in &[10, 20, 50, 100, 200] {
        let n_features = 8;
        // Band-diagonal adjacency (5-neighbor)
        let adj: Vec<f64> = (0..n * n)
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;
                if (i as i64 - j as i64).unsigned_abs() <= 2 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let gnn = StochasticGraphLayer::new(adj, n, n_features, 42);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let features: Vec<f64> = (0..n * n_features).map(|_| rng.random::<f64>()).collect();

        group.bench_with_input(
            BenchmarkId::new("gnn_forward", format!("{n}x{n_features}")),
            &n,
            |b, _| b.iter(|| black_box(gnn.forward(black_box(&features)).unwrap())),
        );
    }
    group.finish();
}

fn bench_dense_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_scaling");
    group.sample_size(10);

    for &(n_in, n_out) in &[(16, 8), (32, 16), (64, 32), (128, 64), (256, 128)] {
        let bitstream_length = 1024;
        let layer = DenseLayer::new(n_in, n_out, bitstream_length, 42);

        let inputs: Vec<f64> = (0..n_in)
            .map(|i| (i as f64 + 1.0) / (n_in as f64 + 1.0))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("dense_fused", format!("{n_in}x{n_out}")),
            &n_in,
            |b, _| b.iter(|| black_box(layer.forward_fused(black_box(&inputs), 42).unwrap())),
        );
    }
    group.finish();
}

fn bench_popcount_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("popcount_scaling");

    for &n_words in &[64, 256, 1024, 4096, 16384, 65536] {
        let data: Vec<u64> = (0..n_words)
            .map(|i| 0xAAAA_BBBB_CCCC_DDDDu64.wrapping_mul(i as u64))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("popcount_simd", n_words),
            &n_words,
            |b, _| {
                b.iter(|| {
                    black_box(sc_neurocore_engine::simd::popcount_dispatch(black_box(
                        &data,
                    )))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("popcount_portable", n_words),
            &n_words,
            |b, _| {
                b.iter(|| {
                    black_box(sc_neurocore_engine::bitstream::popcount_words_portable(
                        black_box(&data),
                    ))
                })
            },
        );
    }
    group.finish();
}

/// LIF network benchmark: encode → dense SC layer → decode loop.
///
/// Simulates a stochastic-computing spiking network for 100 timesteps
/// at each scale, measuring per-step throughput.
fn bench_lif_network_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lif_network_scaling");
    group.sample_size(10);

    for &n in &[100, 500, 1000, 2000, 5000] {
        let bitstream_length = 1024;
        let layer = DenseLayer::new(n, n, bitstream_length, 42);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let inputs: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();

        group.bench_with_input(BenchmarkId::new("lif_sc_100steps", n), &n, |b, _| {
            b.iter(|| {
                let mut v = vec![0.0f64; n];
                let mut total_spikes = 0u64;
                for step in 0..100u64 {
                    let out = layer.forward_fused(black_box(&inputs), 42 + step).unwrap();
                    for (vi, oi) in v.iter_mut().zip(out.iter()) {
                        *vi = 0.9 * *vi + *oi;
                        if *vi >= 0.5 {
                            total_spikes += 1;
                            *vi = 0.0;
                        }
                    }
                }
                black_box(total_spikes)
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_kuramoto_scaling,
    bench_gnn_scaling,
    bench_dense_scaling,
    bench_popcount_scaling,
    bench_lif_network_scaling,
);
criterion_main!(benches);
