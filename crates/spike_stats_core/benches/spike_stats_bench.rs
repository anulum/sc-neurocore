// SPDX-License-Identifier: AGPL-3.0-or-later
// SC-NeuroCore — Spike Stats Criterion Benchmarks

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use spike_stats_core::*;

fn bench_victor_purpura(c: &mut Criterion) {
    let mut group = c.benchmark_group("victor_purpura");
    for &n in &[10, 50, 100, 200] {
        let a: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
        let b: Vec<f64> = (0..n).map(|i| i as f64 * 0.01 + 0.002).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| victor_purpura_distance(black_box(&a), black_box(&b), 1000.0));
        });
    }
    group.finish();
}

fn bench_spike_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("spike_distance");
    for &n in &[10, 50, 100, 200] {
        let a: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
        let b: Vec<f64> = (0..n).map(|i| i as f64 * 0.01 + 0.003).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| spike_distance(black_box(&a), black_box(&b), 0.0, n as f64 * 0.01));
        });
    }
    group.finish();
}

fn bench_spike_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("spike_sync");
    for &n in &[10, 50, 100, 200] {
        let a: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
        let b: Vec<f64> = (0..n).map(|i| i as f64 * 0.01 + 0.001).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| spike_sync(black_box(&a), black_box(&b), 0.0, n as f64 * 0.01));
        });
    }
    group.finish();
}

fn bench_cross_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_correlation");
    for &n in &[50, 200, 500] {
        let a: Vec<f64> = (0..n).map(|i| i as f64 * 0.002).collect();
        let b: Vec<f64> = (0..n).map(|i| i as f64 * 0.002 + 0.001).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| cross_correlation(black_box(&a), black_box(&b), 0.001, 0.05));
        });
    }
    group.finish();
}

fn bench_multi_vp(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_neuron_vp");
    for &n_trains in &[4, 8, 16] {
        let trains: Vec<Vec<f64>> = (0..n_trains)
            .map(|k| {
                (0..20)
                    .map(|i| i as f64 * 0.05 + k as f64 * 0.002)
                    .collect()
            })
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(n_trains),
            &n_trains,
            |bench, _| {
                bench.iter(|| multi_neuron_victor_purpura(black_box(&trains), 1000.0));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_victor_purpura,
    bench_spike_distance,
    bench_spike_sync,
    bench_cross_correlation,
    bench_multi_vp,
    bench_approximate_entropy,
    bench_sample_entropy,
    bench_lempel_ziv,
    bench_permutation_entropy,
    bench_kl_mi,
    bench_spike_train_entropy,
);
criterion_main!(benches);

fn bench_approximate_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("approximate_entropy");
    for &n in &[100, 500, 1000] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| approximate_entropy(black_box(&data), 2, 0.2));
        });
    }
    group.finish();
}

fn bench_sample_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_entropy");
    for &n in &[100, 500, 1000] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| sample_entropy(black_box(&data), 2, 0.2));
        });
    }
    group.finish();
}

fn bench_lempel_ziv(c: &mut Criterion) {
    let mut group = c.benchmark_group("lempel_ziv");
    for &n in &[100, 1000, 10000] {
        let data: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| lempel_ziv_complexity(black_box(&data)));
        });
    }
    group.finish();
}

fn bench_permutation_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("permutation_entropy");
    for &n in &[100, 500, 1000] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| permutation_entropy(black_box(&data), 3, 1));
        });
    }
    group.finish();
}

fn bench_kl_mi(c: &mut Criterion) {
    let mut group = c.benchmark_group("kozachenko_leonenko_mi");
    for &n in &[100, 500, 1000] {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).cos()).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| kozachenko_leonenko_mi(black_box(&x), black_box(&y), 3));
        });
    }
    group.finish();
}

fn bench_spike_train_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("spike_train_entropy");
    for &n in &[100, 1000, 5000] {
        let data: Vec<u8> = (0..n).map(|i| (i % 3 == 0) as u8).collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| spike_train_entropy(black_box(&data), 4));
        });
    }
    group.finish();
}
