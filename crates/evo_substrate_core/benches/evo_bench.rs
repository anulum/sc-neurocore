// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — evo_substrate_core criterion benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use evo_substrate_core::{crossover_uniform, genomic_distance, point_mutation};

fn bench_distance(c: &mut Criterion) {
    let a: Vec<f64> = (0..19).map(|i| i as f64 * 0.1 + 0.05).collect();
    let b: Vec<f64> = (0..19).map(|i| i as f64 * 0.2 - 0.1).collect();
    c.bench_function("genomic_distance_19d", |bn| {
        bn.iter(|| {
            black_box(genomic_distance(black_box(&a), black_box(&b)));
        });
    });
}

fn bench_crossover(c: &mut Criterion) {
    let a: Vec<f64> = (0..19).map(|i| i as f64).collect();
    let b: Vec<f64> = (0..19).map(|i| -(i as f64)).collect();
    let mask: Vec<u8> = (0..19).map(|i| (i % 2) as u8).collect();
    let mut out = vec![0.0; 19];
    c.bench_function("crossover_uniform_19d", |bn| {
        bn.iter(|| {
            crossover_uniform(black_box(&a), black_box(&b), black_box(&mask), &mut out);
        });
    });
}

fn bench_mutation(c: &mut Criterion) {
    let mask: Vec<u8> = (0..19).map(|i| (i % 3 == 0) as u8).collect();
    let noise: Vec<f64> = (0..19).map(|i| 0.01 * i as f64).collect();
    c.bench_function("point_mutation_19d", |bn| {
        bn.iter_batched(
            || (0..19).map(|i| i as f64 * 0.1).collect::<Vec<_>>(),
            |mut gene| {
                point_mutation(black_box(&mut gene), black_box(&mask), black_box(&noise));
                black_box(gene);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_distance, bench_crossover, bench_mutation);
criterion_main!(benches);
