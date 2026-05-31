// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — HDC/VSA Benchmarks
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use neuro_symbolic::{AssociativeMemory, Hypervector, SymbolEncoder, HYPERVECTOR_DIM};

fn bench_bind(c: &mut Criterion) {
    let a = Hypervector::random(1);
    let b = Hypervector::random(2);
    c.bench_function("bind", |bench| {
        bench.iter(|| black_box(a.bind(&b)));
    });
}

fn bench_permute(c: &mut Criterion) {
    let mut group = c.benchmark_group("permute");
    for shift in [1, 7, 64, 1000] {
        let hv = Hypervector::random(42);
        group.bench_with_input(BenchmarkId::from_parameter(shift), &shift, |bench, &s| {
            bench.iter(|| {
                let mut v = hv.clone();
                v.permute(s);
                black_box(v)
            });
        });
    }
    group.finish();
}

fn bench_threshold_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_bundle");
    for n in [3, 5, 9, 21] {
        let vecs: Vec<Hypervector> = (0..n).map(|i| Hypervector::random(i as u64)).collect();
        let refs: Vec<&Hypervector> = vecs.iter().collect();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| black_box(Hypervector::threshold_bundle(&refs)));
        });
    }
    group.finish();
}

fn bench_hamming_distance(c: &mut Criterion) {
    let a = Hypervector::random(10);
    let b = Hypervector::random(20);
    c.bench_function("hamming_distance", |bench| {
        bench.iter(|| black_box(a.hamming_distance(&b)));
    });
}

fn bench_associative_memory_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_query");
    for size in [10, 100, 1000] {
        let mut mem = AssociativeMemory::new();
        for i in 0..size {
            mem.store(format!("item_{i}"), Hypervector::random(i as u64));
        }
        let probe = Hypervector::random(0);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| black_box(mem.query(&probe)));
        });
    }
    group.finish();
}

fn bench_symbol_encode(c: &mut Criterion) {
    c.bench_function("symbol_encode", |bench| {
        bench.iter(|| {
            let mut enc = SymbolEncoder::new(42);
            for i in 0..100 {
                enc.encode(black_box(&format!("sym_{i}")));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_bind,
    bench_permute,
    bench_threshold_bundle,
    bench_hamming_distance,
    bench_associative_memory_query,
    bench_symbol_encode,
);
criterion_main!(benches);
