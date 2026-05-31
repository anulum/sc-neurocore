// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Core Engine Benchmarks
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use core_engine::bitstream::{pack, scc, cordiv, Bitstream, Lfsr16};

fn bench_sc_and(c: &mut Criterion) {
    let mut group = c.benchmark_group("sc_and");
    for length in [256, 1024, 4096] {
        let a = Bitstream::ones(length);
        let b = Bitstream::ones(length);
        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |bench, _| {
            bench.iter(|| black_box(a.sc_and(&b)));
        });
    }
    group.finish();
}

fn bench_sc_mux(c: &mut Criterion) {
    let mut group = c.benchmark_group("sc_mux");
    for length in [256, 1024, 4096] {
        let a = Bitstream::ones(length);
        let b = Bitstream::zeros(length);
        let s = Bitstream::ones(length);
        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |bench, _| {
            bench.iter(|| black_box(a.sc_mux(&b, &s)));
        });
    }
    group.finish();
}

fn bench_popcount(c: &mut Criterion) {
    let mut group = c.benchmark_group("popcount");
    for length in [256, 1024, 4096, 16384] {
        let bs = Bitstream::ones(length);
        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |bench, _| {
            bench.iter(|| black_box(bs.popcount()));
        });
    }
    group.finish();
}

fn bench_scc(c: &mut Criterion) {
    let mut group = c.benchmark_group("scc");
    for length in [256, 1024, 4096] {
        let mut lfsr_a = Lfsr16::new(0xACE1);
        let mut lfsr_b = Lfsr16::new(0xBEEF);
        let a = lfsr_a.encode(32768, length);
        let b = lfsr_b.encode(32768, length);
        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |bench, _| {
            bench.iter(|| black_box(scc(&a, &b)));
        });
    }
    group.finish();
}

fn bench_cordiv(c: &mut Criterion) {
    let mut group = c.benchmark_group("cordiv");
    for length in [256, 1024, 4096] {
        let mut lfsr_x = Lfsr16::new(0xACE1);
        let mut lfsr_y = Lfsr16::new(0xBEEF);
        let x = lfsr_x.encode((0.3 * 65535.0) as u16, length);
        let y = lfsr_y.encode((0.6 * 65535.0) as u16, length);
        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |bench, _| {
            bench.iter(|| black_box(cordiv(&x, &y)));
        });
    }
    group.finish();
}

fn bench_lfsr_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("lfsr_encode");
    for length in [256, 1024, 4096] {
        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |bench, &len| {
            bench.iter(|| {
                let mut lfsr = Lfsr16::new(0xACE1);
                black_box(lfsr.encode(32768, len))
            });
        });
    }
    group.finish();
}

fn bench_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack");
    for length in [256, 1024, 4096] {
        let bits: Vec<u8> = (0..length).map(|i| (i % 2) as u8).collect();
        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |bench, _| {
            bench.iter(|| black_box(pack(&bits)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sc_and,
    bench_sc_mux,
    bench_popcount,
    bench_scc,
    bench_cordiv,
    bench_lfsr_encode,
    bench_pack,
);
criterion_main!(benches);
