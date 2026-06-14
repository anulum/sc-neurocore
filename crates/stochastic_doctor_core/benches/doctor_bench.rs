// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stochastic Doctor Core Criterion Benchmarks

//! Benchmarks for SCC, batch SCC, precision, histogram, and drift detector.
//!
//! Run with: `cargo bench -p stochastic_doctor_core`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use stochastic_doctor_core::*;

fn bench_scc_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("scc_bytes");
    for &size in &[100, 1_000, 10_000, 100_000] {
        let a: Vec<u8> = (0..size).map(|i| ((i * 7 + 3) % 2) as u8).collect();
        let b: Vec<u8> = (0..size).map(|i| ((i * 13 + 5) % 2) as u8).collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| scc_bytes(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_scc_packed(c: &mut Criterion) {
    let mut group = c.benchmark_group("scc_packed");
    for &bit_length in &[256, 1024, 8192, 65536] {
        let word_count = (bit_length + 63) / 64;
        let a: Vec<u64> = (0..word_count)
            .map(|i| 0xAAAA_AAAA_AAAA_AAAAu64.wrapping_add(i as u64))
            .collect();
        let b: Vec<u64> = (0..word_count)
            .map(|i| 0x5555_5555_5555_5555u64.wrapping_add(i as u64))
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(bit_length),
            &bit_length,
            |bench, _| {
                bench.iter(|| scc_packed(black_box(&a), black_box(&b), black_box(bit_length)));
            },
        );
    }
    group.finish();
}

fn bench_scc_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("scc_batch");
    let stream_len = 2048;
    for &n in &[4, 8, 16, 32] {
        let streams: Vec<u8> = (0..n * stream_len)
            .map(|i| ((i * 7 + 3) % 2) as u8)
            .collect();
        let mut out = vec![0.0f64; n * n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                scc_batch_impl(
                    black_box(&streams),
                    black_box(n),
                    black_box(stream_len),
                    black_box(&mut out),
                )
            });
        });
    }
    group.finish();
}

fn bench_precision_packed(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_packed");
    for &bit_length in &[256, 1024, 8192, 65536] {
        let word_count = (bit_length + 63) / 64;
        let data: Vec<u64> = (0..word_count)
            .map(|i| 0xAAAA_AAAA_AAAA_AAAAu64.wrapping_add(i as u64))
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(bit_length),
            &bit_length,
            |bench, _| {
                bench.iter(|| precision_packed(black_box(&data), black_box(bit_length)));
            },
        );
    }
    group.finish();
}

fn bench_histogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram_u64");
    for &word_count in &[64, 256, 1024, 4096] {
        let data: Vec<u64> = (0..word_count)
            .map(|i| (i as u64).wrapping_mul(0x517cc1b727220a95))
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(word_count),
            &word_count,
            |bench, _| {
                bench.iter(|| histogram_u64_vec(black_box(&data)));
            },
        );
    }
    group.finish();
}

fn bench_drift_detector(c: &mut Criterion) {
    c.bench_function("drift_detector_1000_observations", |bench| {
        bench.iter(|| {
            let mut dd = DriftDetector::new(0.1, 0.3);
            for i in 0..1000 {
                dd.observe(black_box((i as f64 / 1000.0).sin()));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_scc_bytes,
    bench_scc_packed,
    bench_scc_batch,
    bench_precision_packed,
    bench_histogram,
    bench_drift_detector,
);
criterion_main!(benches);
