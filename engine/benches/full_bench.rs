// CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
// Contact us: www.anulum.li  protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AFFERO GENERAL PUBLIC LICENSE v3
// Commercial Licensing: Available

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use sc_neurocore_engine::attention::StochasticAttention;
use sc_neurocore_engine::bitstream::{
    bernoulli_packed, bernoulli_packed_fast, bernoulli_packed_simd, bernoulli_stream,
    encode_and_popcount, pack, pack_fast, popcount_words_portable,
};
use sc_neurocore_engine::encoder::BitstreamEncoder;
use sc_neurocore_engine::graph::StochasticGraphLayer;
use sc_neurocore_engine::layer::DenseLayer;
use sc_neurocore_engine::neuron::FixedPointLif;
use sc_neurocore_engine::scpn::KuramotoSolver;
use sc_neurocore_engine::simd::{fused_and_popcount_dispatch, pack_dispatch, popcount_dispatch};

fn bench_all(c: &mut Criterion) {
    // -- Bitstream --
    let bits_1m: Vec<u8> = (0..(1024 * 1024))
        .map(|i| if i % 3 == 0 { 1 } else { 0 })
        .collect();

    c.bench_function("pack_1m", |b| {
        b.iter(|| black_box(pack(black_box(&bits_1m))))
    });

    c.bench_function("pack_fast_1m", |b| {
        b.iter(|| black_box(pack_fast(black_box(&bits_1m))))
    });

    c.bench_function("pack_dispatch_1m", |b| {
        b.iter(|| black_box(pack_dispatch(black_box(&bits_1m))))
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

    c.bench_function("lif_100k_steps", |b| {
        b.iter(|| {
            let mut lif = FixedPointLif::new(16, 8, 0, 0, 256, 2);
            for _ in 0..100_000 {
                black_box(lif.step(20, 256, 128, 0));
            }
        })
    });

    // -- Bernoulli encoding comparison --
    c.bench_function("bernoulli_stream_1024", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            black_box(bernoulli_stream(0.5, 1024, &mut rng))
        })
    });

    c.bench_function("bernoulli_stream_pack_1024", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let bits = bernoulli_stream(0.5, 1024, &mut rng);
            black_box(pack(&bits).data)
        })
    });

    c.bench_function("bernoulli_packed_1024", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            black_box(bernoulli_packed(0.5, 1024, &mut rng))
        })
    });

    c.bench_function("bernoulli_packed_fast_1024", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            black_box(bernoulli_packed_fast(0.5, 1024, &mut rng))
        })
    });

    c.bench_function("bernoulli_packed_simd_1024", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            black_box(bernoulli_packed_simd(0.5, 1024, &mut rng))
        })
    });

    c.bench_function("bernoulli_packed_simd_xoshiro_1024", |b| {
        b.iter(|| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            black_box(bernoulli_packed_simd(0.5, 1024, &mut rng))
        })
    });

    let a_words: Vec<u64> = (0..16)
        .map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_A5A5_5A5A_5A5A)
        .collect();
    let b_words: Vec<u64> = (0..16)
        .map(|i| (i as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ 0x0F0F_F0F0_33CC_CC33)
        .collect();

    c.bench_function("fused_and_popcount_scalar_16w", |b| {
        b.iter(|| {
            black_box(
                a_words
                    .iter()
                    .zip(b_words.iter())
                    .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
                    .sum::<u64>(),
            )
        })
    });

    c.bench_function("fused_and_popcount_dispatch_16w", |b| {
        b.iter(|| {
            black_box(fused_and_popcount_dispatch(
                black_box(&a_words),
                black_box(&b_words),
            ))
        })
    });

    // -- Dense forward variants --
    let layer = DenseLayer::new(64, 32, 1024, 42);
    let inputs = vec![0.5_f64; 64];

    c.bench_function("dense_forward_64x32", |b| {
        b.iter(|| black_box(layer.forward(black_box(&inputs), 42).unwrap()))
    });

    c.bench_function("dense_forward_fast_64x32", |b| {
        b.iter(|| black_box(layer.forward_fast(black_box(&inputs), 42).unwrap()))
    });

    c.bench_function("dense_forward_fast_flat_64x32", |b| {
        b.iter(|| black_box(layer.forward_fast(black_box(&inputs), 42).unwrap()))
    });

    c.bench_function("dense_forward_fused_64x32", |b| {
        b.iter(|| black_box(layer.forward_fused(black_box(&inputs), 42).unwrap()))
    });

    let weights_16w: Vec<u64> = (0..16)
        .map(|i| (i as u64).wrapping_mul(0xD6E8_FD9D_5A2B_1C47) ^ 0x1357_9BDF_2468_ACE0)
        .collect();
    c.bench_function("bernoulli_encode_and_popcount_1024", |b| {
        b.iter(|| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            black_box(encode_and_popcount(
                black_box(&weights_16w),
                0.5,
                1024,
                &mut rng,
            ))
        })
    });

    let n_samples = 100_usize;
    let batch_inputs: Vec<f64> = (0..(n_samples * 64))
        .map(|i| ((i * 13 + 7) % 100) as f64 / 100.0)
        .collect();
    c.bench_function("dense_forward_batch_64x32_x100", |b| {
        b.iter(|| {
            black_box(
                layer
                    .forward_batch(black_box(&batch_inputs), n_samples, 42)
                    .unwrap(),
            )
        })
    });

    c.bench_function("prng_chacha_fill_1024", |b| {
        b.iter(|| {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let mut buf = [0_u8; 1024];
            rand::RngCore::fill_bytes(&mut rng, &mut buf);
            black_box(buf)
        })
    });

    c.bench_function("prng_xoshiro_fill_1024", |b| {
        b.iter(|| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            let mut buf = [0_u8; 1024];
            rand::RngCore::fill_bytes(&mut rng, &mut buf);
            black_box(buf)
        })
    });

    let packed_inputs: Vec<Vec<u64>> = inputs
        .iter()
        .enumerate()
        .map(|(idx, &p)| {
            let mut rng = ChaCha8Rng::seed_from_u64(42_u64.wrapping_add(idx as u64));
            sc_neurocore_engine::bitstream::bernoulli_packed(p, 1024, &mut rng)
        })
        .collect();

    c.bench_function("dense_forward_prepacked_64x32", |b| {
        b.iter(|| black_box(layer.forward_prepacked(black_box(&packed_inputs)).unwrap()))
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

    // -- Attention (rate-mode) --
    {
        let attn = StochasticAttention::new(16);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let q: Vec<f64> = (0..10 * 16).map(|_| rng.gen()).collect();
        let k: Vec<f64> = (0..20 * 16).map(|_| rng.gen()).collect();
        let v: Vec<f64> = (0..20 * 32).map(|_| rng.gen()).collect();

        c.bench_function("attention_10x16_20x32", |b| {
            b.iter(|| {
                black_box(
                    attn.forward(
                        black_box(&q),
                        10,
                        16,
                        black_box(&k),
                        20,
                        16,
                        black_box(&v),
                        20,
                        32,
                    )
                    .unwrap(),
                )
            })
        });
    }

    // -- Graph Layer --
    {
        let adj: Vec<f64> = {
            let mut a = vec![0.0; 20 * 20];
            for i in 0..20 {
                for j in 0..20 {
                    if (i as i32 - j as i32).abs() <= 2 {
                        a[i * 20 + j] = 1.0;
                    }
                }
            }
            a
        };
        let gnn = StochasticGraphLayer::new(adj, 20, 8, 42);
        let features: Vec<f64> = (0..20 * 8).map(|i| (i as f64) * 0.01).collect();

        c.bench_function("gnn_20x8_forward", |b| {
            b.iter(|| black_box(gnn.forward(black_box(&features)).unwrap()))
        });
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
