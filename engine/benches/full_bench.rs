use criterion::{black_box, criterion_group, criterion_main, Criterion};

use sc_neurocore_engine::bitstream::{pack, popcount_words_portable};
use sc_neurocore_engine::encoder::BitstreamEncoder;
use sc_neurocore_engine::layer::DenseLayer;
use sc_neurocore_engine::neuron::FixedPointLif;
use sc_neurocore_engine::scpn::KuramotoSolver;
use sc_neurocore_engine::simd::popcount_dispatch;

fn bench_all(c: &mut Criterion) {
    // -- Bitstream --
    let bits_1m: Vec<u8> = (0..(1024 * 1024))
        .map(|i| if i % 3 == 0 { 1 } else { 0 })
        .collect();

    c.bench_function("pack_1m", |b| {
        b.iter(|| black_box(pack(black_box(&bits_1m))))
    });

    let packed = pack(&bits_1m);

    c.bench_function("popcount_portable_1m", |b| {
        b.iter(|| black_box(popcount_words_portable(black_box(&packed.data))))
    });

    c.bench_function("popcount_simd_1m", |b| {
        b.iter(|| black_box(popcount_dispatch(black_box(&packed.data))))
    });

    // -- Encoder --
    c.bench_function("encoder_64k_steps", |b| {
        b.iter(|| {
            let mut enc = BitstreamEncoder::new(16, 0xACE1);
            for _ in 0..65535 {
                black_box(enc.step(32768));
            }
        })
    });

    // -- LIF Neuron --
    c.bench_function("lif_10k_steps", |b| {
        b.iter(|| {
            let mut lif = FixedPointLif::new(16, 8, 0, 0, 256, 2);
            for _ in 0..10_000 {
                black_box(lif.step(20, 256, 128, 0));
            }
        })
    });

    // -- Dense Layer --
    let layer = DenseLayer::new(64, 32, 1024, 42);
    let inputs = vec![0.5_f64; 64];
    c.bench_function("dense_64x32_l1024", |b| {
        b.iter(|| black_box(layer.forward(black_box(&inputs), 42).unwrap()))
    });

    // -- Kuramoto --
    let n = 100;
    let omega = vec![1.0; n];
    let coupling = vec![0.3; n * n];
    let phases: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * (i as f64) / (n as f64))
        .collect();

    c.bench_function("kuramoto_100_osc_1000_steps", |b| {
        b.iter(|| {
            let mut solver =
                KuramotoSolver::new(omega.clone(), coupling.clone(), phases.clone(), 0.0);
            black_box(solver.run(1000, 0.01, 42));
        })
    });
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
