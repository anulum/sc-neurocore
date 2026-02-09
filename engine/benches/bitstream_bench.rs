use criterion::{black_box, criterion_group, criterion_main, Criterion};

use sc_neurocore_engine::bitstream::{pack, popcount_words_portable};
use sc_neurocore_engine::simd::popcount_dispatch;

fn bench_pack_and_popcount(c: &mut Criterion) {
    let bits: Vec<u8> = (0..(1024 * 1024))
        .map(|idx| if idx % 3 == 0 { 1 } else { 0 })
        .collect();

    c.bench_function("pack_bitstream_1m", |b| {
        b.iter(|| {
            let packed = pack(black_box(&bits));
            black_box(packed);
        })
    });

    let packed = pack(&bits);
    c.bench_function("popcount_portable_1m", |b| {
        b.iter(|| {
            let count = popcount_words_portable(black_box(&packed.data));
            black_box(count);
        })
    });

    c.bench_function("popcount_simd_dispatch_1m", |b| {
        b.iter(|| {
            let count = popcount_dispatch(black_box(&packed.data));
            black_box(count);
        })
    });
}

criterion_group!(benches, bench_pack_and_popcount);
criterion_main!(benches);
