// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bitstream Bench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::time::Duration;

use sc_neurocore_engine::bitstream::{pack, popcount_words_portable};
use sc_neurocore_engine::simd::popcount_dispatch;

const MEBIBIT: usize = 1024 * 1024;
const POPCOUNT_BIT_SIZES: [usize; 3] = [MEBIBIT, 64 * MEBIBIT, 1024 * MEBIBIT];

fn count_ones_baseline(data: &[u64]) -> u64 {
    data.iter().map(|word| u64::from(word.count_ones())).sum()
}

fn patterned_words(bit_count: usize) -> Vec<u64> {
    assert_eq!(bit_count % u64::BITS as usize, 0);
    (0..bit_count / u64::BITS as usize)
        .map(|index| 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(index as u64 + 1))
        .collect()
}

fn bench_pack(c: &mut Criterion) {
    let bits: Vec<u8> = (0..(1024 * 1024))
        .map(|idx| if idx % 3 == 0 { 1 } else { 0 })
        .collect();

    c.bench_function("pack_u8_bits_to_u64_1m", |b| {
        b.iter(|| {
            let packed = pack(black_box(&bits));
            black_box(packed);
        })
    });
}

fn bench_popcount(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_popcount");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    for bit_count in POPCOUNT_BIT_SIZES {
        let words = patterned_words(bit_count);
        let expected = count_ones_baseline(&words);
        assert_eq!(popcount_words_portable(&words), expected);
        assert_eq!(popcount_dispatch(&words), expected);

        group.throughput(Throughput::Bits(bit_count as u64));
        group.bench_with_input(
            BenchmarkId::new("u64_count_ones", bit_count),
            &words,
            |b, data| {
                b.iter(|| black_box(count_ones_baseline(black_box(data))));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("simd_dispatch", bit_count),
            &words,
            |b, data| {
                b.iter(|| black_box(popcount_dispatch(black_box(data))));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_pack, bench_popcount);
criterion_main!(benches);
