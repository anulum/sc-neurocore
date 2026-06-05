// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Full Bench

use criterion::{criterion_group, criterion_main, Criterion};
use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use sc_neurocore_engine::attention::StochasticAttention;
use sc_neurocore_engine::bitstream::{
    bernoulli_packed, bernoulli_packed_fast, bernoulli_packed_simd, bernoulli_stream,
    encode_and_popcount, pack, pack_fast, popcount_words_portable,
};
use sc_neurocore_engine::encoder::BitstreamEncoder;
use sc_neurocore_engine::graph::StochasticGraphLayer;
use sc_neurocore_engine::ir::qformat::{
    block_floating_dense_q16, mixed_dense_q88_q1616, BlockFloatingMode,
};
use sc_neurocore_engine::layer::DenseLayer;
use sc_neurocore_engine::neuron::{AdExNeuron, ExpIfNeuron, FixedPointLif, LapicqueNeuron};
use sc_neurocore_engine::neurons::{
    ATypeKNeuron,
    AdaptiveThresholdIFNeuron,
    AiharaMapNeuron,
    AkidaNeuron,
    AlphaMotorNeuron,
    AlphaNeuron,
    AmariNeuralField,
    ArcaneNeuron,
    AstrocyteModel,
    AttentionGatedNeuron,
    AvRonCardiacNeuron,
    BKNeuron,
    BalancedResonateAndFireNeuron,
    BendaHerzNeuron,
    BertramPhantomBurster,
    BoothRinzelNeuron,
    BrainScaleSAdExNeuron,
    BrunelNetwork,
    BrunelWangNeuron,
    ButeraRespiratoryNeuron,
    COBALIFNeuron,
    CardiacPurkinjeFibre,
    CazellesMapNeuron,
    CerebellarBasketNeuron,
    ChandelierNeuron,
    ChayKeizerNeuron,
    ChayNeuron,
    // maps (remaining 6)
    ChialvoMapNeuron,
    ClosedFormContinuousNeuron,
    ComplementaryLIFNeuron,
    CompositionalBindingNeuron,
    CompteWMNeuron,
    ConePhotoreceptor,
    ConnorStevensNeuron,
    ContinuousAttractorNeuron,
    CourageNekorkinMapNeuron,
    DCNNeuron,
    DPINeuron,
    DeSchutterPurkinjeNeuron,
    DendrifyNeuron,
    DestexheThalamicNeuron,
    DifferentiableSurrogateNeuron,
    DurstewitzDopamineNeuron,
    EPropALIFNeuron,
    ElBoustaniNetwork,
    EndocrineBetaCell,
    EnergyLIFNeuron,
    ErmentroutKopellMapNeuron,
    ErmentroutKopellPopulation,
    EscapeRateNeuron,
    FitzHughNagumoNeuron,
    FitzHughRinzelNeuron,
    FractionalLIFNeuron,
    FrankenhaeUserHuxleyAxon,
    GIFPopulationNeuron,
    GLIFNeuron,
    GLMNeuron,
    GalvesLocherbachNeuron,
    GammaRenewalNeuron,
    GapJunctionNeuron,
    GatedLIFNeuron,
    GolgiCell,
    GolombFSNeuron,
    GradedSynapseNeuron,
    GranuleCell,
    GutkinErmentroutNeuron,
    HayL5PyramidalNeuron,
    HillTononiNeuron,
    HindmarshRoseNeuron,
    HodgkinHuxleyNeuron,
    HuberBraunNeuron,
    IbarzTanakaMapNeuron,
    IhNeuron,
    InhibitoryLIFNeuron,
    InhomogeneousPoissonNeuron,
    InnerHairCell,
    IntegerQIFNeuron,
    JansenRitUnit,
    KLIFNeuron,
    KilincBhattMapNeuron,
    LarterBreakspearNeuron,
    LeakyCompeteFireNeuron,
    LearnableNeuronModel,
    LiquidTimeConstantNeuron,
    Loihi2Neuron,
    // hardware
    LoihiCUBANeuron,
    LugaroCell,
    MATNeuron,
    MainenSejnowskiNeuron,
    MarderSTGNeuron,
    MartinottiNeuron,
    McKeanNeuron,
    MedvedevMapNeuron,
    MerkelCell,
    MetaPlasticNeuron,
    MihalasNieburNeuron,
    MontbrioMeanField,
    MorrisLecarNeuron,
    // ai_optimized
    MultiTimescaleNeuron,
    MyelinatedAxon,
    NMDANeuron,
    NeuroGridNeuron,
    Nociceptor,
    NodeOfRanvier,
    NonResettingLIFNeuron,
    NonlinearLIFNeuron,
    OlfactoryReceptorNeuron,
    // sensory (remaining 3)
    OuterHairCell,
    PVFastSpikingNeuron,
    PacinianCorpuscle,
    ParallelSpikingNeuron,
    ParametricLIFNeuron,
    PerfectIntegratorNeuron,
    PernarowskiNeuron,
    PersistentNaNeuron,
    // multi_compartment
    PinskyRinzelNeuron,
    PlantR15Neuron,
    PoissonNeuron,
    PospischilNeuron,
    PredictiveCodingNeuron,
    PrescottNeuron,
    QuadraticIFNeuron,
    RallCableNeuron,
    ResonateAndFireNeuron,
    RetinalGanglionCell,
    RodPhotoreceptor,
    RulkovMapNeuron,
    SFANeuron,
    SKNeuron,
    SSTNeuron,
    SelfReferentialNeuron,
    ShermanRinzelKeizerNeuron,
    SiegertTransferFunction,
    SigmaDeltaNeuron,
    // rate
    SigmoidRateNeuron,
    SmoothMuscleCell,
    SpiNNaker2Neuron,
    SpiNNakerLIFNeuron,
    SpikeResponseNeuron,
    StellateCell,
    StochasticIFNeuron,
    StochasticLIFNeuron,
    SuperSpikeNeuron,
    TTypeCaNeuron,
    TUMNetwork,
    TasteReceptorCell,
    TermanWangOscillator,
    ThetaNeuron,
    ThresholdLinearRateNeuron,
    TraubMilesNeuron,
    TrueNorthNeuron,
    TsodyksMarkramNeuron,
    TwoCompartmentLIFNeuron,
    UnipolarBrushCell,
    VIPNeuron,
    WangBuzsakiNeuron,
    WendlingNeuron,
    WilsonCowanUnit,
    WilsonHRNeuron,
    WongWangUnit,
    YamadaNeuron,
};
use sc_neurocore_engine::scpn::KuramotoSolver;
use sc_neurocore_engine::simd::{fused_and_popcount_dispatch, pack_dispatch, popcount_dispatch};
use std::hint::black_box;

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

    c.bench_function("dense_forward_fast_flat_64x32_b", |b| {
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
            rng.fill_bytes(&mut buf);
            black_box(buf)
        })
    });

    c.bench_function("prng_xoshiro_fill_1024", |b| {
        b.iter(|| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            let mut buf = [0_u8; 1024];
            rng.fill_bytes(&mut buf);
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

    let mixed_weights_q88: Vec<i16> = (0..(64 * 32))
        .map(|i| (((i * 17 + 11) % 513) as i32 - 256) as i16)
        .collect();
    let mixed_inputs_q1616: Vec<i32> = (0..64)
        .map(|i| (((i * 19 + 5) % 257) as i32 - 128) << 8)
        .collect();
    c.bench_function("mixed_dense_q88_q1616_64x32", |b| {
        b.iter(|| {
            black_box(
                mixed_dense_q88_q1616(
                    black_box(&mixed_weights_q88),
                    black_box(&mixed_inputs_q1616),
                    32,
                    64,
                )
                .unwrap(),
            )
        })
    });
    let bfp_mode = BlockFloatingMode::bfp16_e3_x32();
    let bfp_mantissas: Vec<i16> = (0..(64 * 32))
        .map(|i| (((i * 23 + 3) % 1025) as i32 - 512) as i16)
        .collect();
    let bfp_exponents: Vec<u8> = vec![
        bfp_mode.exponent_bias() as u8;
        (64 * 32 + bfp_mode.block_size - 1) / bfp_mode.block_size
    ];
    c.bench_function("block_floating_dense_q16_64x32", |b| {
        b.iter(|| {
            black_box(
                block_floating_dense_q16(
                    black_box(&bfp_mantissas),
                    black_box(&bfp_exponents),
                    black_box(&mixed_inputs_q1616),
                    32,
                    64,
                    bfp_mode,
                )
                .unwrap(),
            )
        })
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
        let q: Vec<f64> = (0..10 * 16).map(|_| rng.random()).collect();
        let k: Vec<f64> = (0..20 * 16).map(|_| rng.random()).collect();
        let v: Vec<f64> = (0..20 * 32).map(|_| rng.random()).collect();

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

    // -- AdEx, ExpIF, Lapicque neurons --
    c.bench_function("adex_1k_steps", |b| {
        b.iter(|| {
            let mut n = AdExNeuron::new();
            for _ in 0..1000 {
                black_box(n.step(500.0));
            }
        })
    });

    c.bench_function("adex_10k_steps", |b| {
        b.iter(|| {
            let mut n = AdExNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(500.0));
            }
        })
    });

    c.bench_function("expif_1k_steps", |b| {
        b.iter(|| {
            let mut n = ExpIfNeuron::new();
            for _ in 0..1000 {
                black_box(n.step(500.0));
            }
        })
    });

    c.bench_function("expif_10k_steps", |b| {
        b.iter(|| {
            let mut n = ExpIfNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(500.0));
            }
        })
    });

    c.bench_function("lapicque_1k_steps", |b| {
        b.iter(|| {
            let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
            for _ in 0..1000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("lapicque_10k_steps", |b| {
        b.iter(|| {
            let mut n = LapicqueNeuron::new(20.0, 1.0, 1.0, 1.0);
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    // -- Interneurons --
    c.bench_function("pv_fs_1k_steps", |b| {
        b.iter(|| {
            let mut n = PVFastSpikingNeuron::new();
            for _ in 0..1000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("sst_1k_steps", |b| {
        b.iter(|| {
            let mut n = SSTNeuron::new();
            for _ in 0..1000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("vip_1k_steps", |b| {
        b.iter(|| {
            let mut n = VIPNeuron::new();
            for _ in 0..1000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("chandelier_1k_steps", |b| {
        b.iter(|| {
            let mut n = ChandelierNeuron::new();
            for _ in 0..1000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("basket_cerebellar_1k_steps", |b| {
        b.iter(|| {
            let mut n = CerebellarBasketNeuron::new();
            for _ in 0..1000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("martinotti_1k_steps", |b| {
        b.iter(|| {
            let mut n = MartinottiNeuron::new();
            for _ in 0..1000 {
                black_box(n.step(4.0));
            }
        })
    });

    // -- Motor Neurons --
    c.bench_function("alpha_motor_1k_steps", |b| {
        b.iter(|| {
            let mut n = AlphaMotorNeuron::new();
            for _ in 0..1000 {
                black_box(n.step(4.0));
            }
        })
    });

    c.bench_function("upper_motor_1k_steps", |b| {
        b.iter(|| {
            let mut n = sc_neurocore_engine::neurons::UpperMotorNeuron::new();
            for _ in 0..1000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("motor_unit_10k_steps", |b| {
        b.iter(|| {
            let mut n = sc_neurocore_engine::neurons::MotorUnit::new();
            for _ in 0..10_000 {
                black_box(n.step(20.0));
            }
        })
    });

    c.bench_function("renshaw_1k_steps", |b| {
        b.iter(|| {
            let mut n = sc_neurocore_engine::neurons::RenshawCell::new();
            for _ in 0..1000 {
                black_box(n.step(4.0));
            }
        })
    });

    c.bench_function("gamma_motor_10k_steps", |b| {
        b.iter(|| {
            let mut n = sc_neurocore_engine::neurons::GammaMotorNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(20.0));
            }
        })
    });

    // -- Sensory Neurons --
    c.bench_function("ihc_10k_steps", |b| {
        b.iter(|| {
            let mut n = InnerHairCell::new();
            for _ in 0..10_000 {
                black_box(n.step(50.0));
            }
        })
    });

    c.bench_function("rod_10k_steps", |b| {
        b.iter(|| {
            let mut n = RodPhotoreceptor::new();
            for _ in 0..10_000 {
                black_box(n.step(100.0));
            }
        })
    });

    c.bench_function("rgc_10k_steps", |b| {
        b.iter(|| {
            let mut n = RetinalGanglionCell::new();
            for _ in 0..10_000 {
                black_box(n.step(20.0));
            }
        })
    });

    c.bench_function("merkel_10k_steps", |b| {
        b.iter(|| {
            let mut n = MerkelCell::new();
            for _ in 0..10_000 {
                black_box(n.step(20.0));
            }
        })
    });

    c.bench_function("pacinian_10k_steps", |b| {
        b.iter(|| {
            let mut n = PacinianCorpuscle::new();
            for i in 0..10_000 {
                black_box(n.step((i as f64 * 0.1).sin() * 50.0));
            }
        })
    });

    c.bench_function("nociceptor_10k_steps", |b| {
        b.iter(|| {
            let mut n = Nociceptor::new();
            for _ in 0..10_000 {
                black_box(n.step(50.0));
            }
        })
    });

    c.bench_function("olfactory_10k_steps", |b| {
        b.iter(|| {
            let mut n = OlfactoryReceptorNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    // -- Cerebellar Neurons --
    c.bench_function("granule_10k_steps", |b| {
        b.iter(|| {
            let mut n = GranuleCell::new();
            for _ in 0..10_000 {
                black_box(n.step(15.0));
            }
        })
    });

    c.bench_function("golgi_1k_steps", |b| {
        b.iter(|| {
            let mut n = GolgiCell::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("stellate_1k_steps", |b| {
        b.iter(|| {
            let mut n = StellateCell::new();
            for _ in 0..1_000 {
                black_box(n.step(10.0));
            }
        })
    });

    c.bench_function("lugaro_10k_steps", |b| {
        b.iter(|| {
            let mut n = LugaroCell::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("ubc_10k_steps", |b| {
        b.iter(|| {
            let mut n = UnipolarBrushCell::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("dcn_1k_steps", |b| {
        b.iter(|| {
            let mut n = DCNNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0));
            }
        })
    });

    // -- Ion Channel Variant Neurons --
    c.bench_function("persistent_na_1k_steps", |b| {
        b.iter(|| {
            let mut n = PersistentNaNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(2.0));
            }
        })
    });

    c.bench_function("ih_1k_steps", |b| {
        b.iter(|| {
            let mut n = IhNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(2.0));
            }
        })
    });

    c.bench_function("ttype_ca_1k_steps", |b| {
        b.iter(|| {
            let mut n = TTypeCaNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(2.0));
            }
        })
    });

    c.bench_function("atype_k_1k_steps", |b| {
        b.iter(|| {
            let mut n = ATypeKNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("bk_1k_steps", |b| {
        b.iter(|| {
            let mut n = BKNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("sk_1k_steps", |b| {
        b.iter(|| {
            let mut n = SKNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("nmda_1k_steps", |b| {
        b.iter(|| {
            let mut n = NMDANeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(3.0));
            }
        })
    });

    // -- Map Neurons --
    c.bench_function("aihara_100k_steps", |b| {
        b.iter(|| {
            let mut n = AiharaMapNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("kilinc_bhatt_100k_steps", |b| {
        b.iter(|| {
            let mut n = KilincBhattMapNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("ermentrout_kopell_100k_steps", |b| {
        b.iter(|| {
            let mut n = ErmentroutKopellMapNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(0.5));
            }
        })
    });

    // -- Population / Mean-Field --
    c.bench_function("montbrio_100k_steps", |b| {
        b.iter(|| {
            let mut n = MontbrioMeanField::new();
            for _ in 0..100_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("brunel_100k_steps", |b| {
        b.iter(|| {
            let mut n = BrunelNetwork::new();
            for _ in 0..100_000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("tum_100k_steps", |b| {
        b.iter(|| {
            let mut n = TUMNetwork::new();
            for _ in 0..100_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("elboustani_100k_steps", |b| {
        b.iter(|| {
            let mut n = ElBoustaniNetwork::new();
            for _ in 0..100_000 {
                black_box(n.step(3.0));
            }
        })
    });

    // -- Misc Models --
    c.bench_function("graded_synapse_100k_steps", |b| {
        b.iter(|| {
            let mut n = GradedSynapseNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(100.0));
            }
        })
    });

    c.bench_function("gap_junction_100k_steps", |b| {
        b.iter(|| {
            let mut n = GapJunctionNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(-20.0));
            }
        })
    });

    c.bench_function("fh_axon_1k_steps", |b| {
        b.iter(|| {
            let mut n = FrankenhaeUserHuxleyAxon::new();
            for _ in 0..1_000 {
                black_box(n.step(1500.0));
            }
        })
    });

    c.bench_function("node_of_ranvier_1k_steps", |b| {
        b.iter(|| {
            let mut n = NodeOfRanvier::new();
            for _ in 0..1_000 {
                black_box(n.step(500.0));
            }
        })
    });

    c.bench_function("myelinated_axon_1k_steps", |b| {
        b.iter(|| {
            let mut n = MyelinatedAxon::new();
            for _ in 0..1_000 {
                black_box(n.step(500.0));
            }
        })
    });

    c.bench_function("cardiac_purkinje_1k_steps", |b| {
        b.iter(|| {
            let mut n = CardiacPurkinjeFibre::new();
            for _ in 0..1_000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("smooth_muscle_1k_steps", |b| {
        b.iter(|| {
            let mut n = SmoothMuscleCell::new();
            for _ in 0..1_000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("beta_cell_1k_steps", |b| {
        b.iter(|| {
            let mut n = EndocrineBetaCell::new();
            for _ in 0..1_000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("brunel_wang_10k_steps", |b| {
        b.iter(|| {
            let mut n = BrunelWangNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(3.0));
            }
        })
    });

    // -- Biophysical models --

    c.bench_function("hh_1k_steps", |b| {
        b.iter(|| {
            let mut n = HodgkinHuxleyNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(10.0));
            }
        })
    });

    c.bench_function("traub_miles_1k_steps", |b| {
        b.iter(|| {
            let mut n = TraubMilesNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("wang_buzsaki_1k_steps", |b| {
        b.iter(|| {
            let mut n = WangBuzsakiNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(2.0));
            }
        })
    });

    c.bench_function("connor_stevens_1k_steps", |b| {
        b.iter(|| {
            let mut n = ConnorStevensNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(10.0));
            }
        })
    });

    c.bench_function("destexhe_1k_steps", |b| {
        b.iter(|| {
            let mut n = DestexheThalamicNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("huber_braun_1k_steps", |b| {
        b.iter(|| {
            let mut n = HuberBraunNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(10.0));
            }
        })
    });

    c.bench_function("golomb_fs_1k_steps", |b| {
        b.iter(|| {
            let mut n = GolombFSNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(200.0));
            }
        })
    });

    c.bench_function("pospischil_1k_steps", |b| {
        b.iter(|| {
            let mut n = PospischilNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("mainen_sejnowski_1k_steps", |b| {
        b.iter(|| {
            let mut n = MainenSejnowskiNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(500.0));
            }
        })
    });

    c.bench_function("de_schutter_purkinje_1k_steps", |b| {
        b.iter(|| {
            let mut n = DeSchutterPurkinjeNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(50.0));
            }
        })
    });

    c.bench_function("plant_r15_1k_steps", |b| {
        b.iter(|| {
            let mut n = PlantR15Neuron::new();
            for _ in 0..1_000 {
                black_box(n.step(2.0));
            }
        })
    });

    c.bench_function("prescott_10k_steps", |b| {
        b.iter(|| {
            let mut n = PrescottNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("mihalas_niebur_10k_steps", |b| {
        b.iter(|| {
            let mut n = MihalasNieburNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("glif_10k_steps", |b| {
        b.iter(|| {
            let mut n = GLIFNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(30.0));
            }
        })
    });

    c.bench_function("gif_pop_10k_steps", |b| {
        b.iter(|| {
            let mut n = GIFPopulationNeuron::new(42);
            for _ in 0..10_000 {
                black_box(n.step(30.0));
            }
        })
    });

    c.bench_function("avron_cardiac_1k_steps", |b| {
        b.iter(|| {
            let mut n = AvRonCardiacNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("durstewitz_1k_steps", |b| {
        b.iter(|| {
            let mut n = DurstewitzDopamineNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("hill_tononi_1k_steps", |b| {
        b.iter(|| {
            let mut n = HillTononiNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("bertram_phantom_1k_steps", |b| {
        b.iter(|| {
            let mut n = BertramPhantomBurster::new();
            for _ in 0..1_000 {
                black_box(n.step(200.0));
            }
        })
    });

    c.bench_function("yamada_1k_steps", |b| {
        b.iter(|| {
            let mut n = YamadaNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0));
            }
        })
    });

    // -- Simple spiking models --

    c.bench_function("fhn_10k_steps", |b| {
        b.iter(|| {
            let mut n = FitzHughNagumoNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(1.0));
            }
        })
    });

    c.bench_function("morris_lecar_10k_steps", |b| {
        b.iter(|| {
            let mut n = MorrisLecarNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(100.0));
            }
        })
    });

    c.bench_function("hindmarsh_rose_10k_steps", |b| {
        b.iter(|| {
            let mut n = HindmarshRoseNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("resonate_and_fire_10k_steps", |b| {
        b.iter(|| {
            let mut n = ResonateAndFireNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("balanced_resonate_and_fire_10k_steps", |b| {
        b.iter(|| {
            let mut n = BalancedResonateAndFireNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(3.0));
            }
        })
    });

    c.bench_function("fitzhugh_rinzel_10k_steps", |b| {
        b.iter(|| {
            let mut n = FitzHughRinzelNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(1.0));
            }
        })
    });

    c.bench_function("mckean_10k_steps", |b| {
        b.iter(|| {
            let mut n = McKeanNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("terman_wang_10k_steps", |b| {
        b.iter(|| {
            let mut n = TermanWangOscillator::new();
            for _ in 0..10_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("benda_herz_10k_steps", |b| {
        b.iter(|| {
            let mut n = BendaHerzNeuron::new(42);
            for _ in 0..10_000 {
                black_box(n.step(20.0));
            }
        })
    });

    c.bench_function("alpha_10k_steps", |b| {
        b.iter(|| {
            let mut n = AlphaNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(0.5, 0.0));
            }
        })
    });

    c.bench_function("coba_lif_10k_steps", |b| {
        b.iter(|| {
            let mut n = COBALIFNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(500.0, 0.0, 0.0));
            }
        })
    });

    c.bench_function("gutkin_ermentrout_10k_steps", |b| {
        b.iter(|| {
            let mut n = GutkinErmentroutNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(15.0));
            }
        })
    });

    c.bench_function("wilson_hr_10k_steps", |b| {
        b.iter(|| {
            let mut n = WilsonHRNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("chay_1k_steps", |b| {
        b.iter(|| {
            let mut n = ChayNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(20.0));
            }
        })
    });

    c.bench_function("chay_keizer_1k_steps", |b| {
        b.iter(|| {
            let mut n = ChayKeizerNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(10.0));
            }
        })
    });

    c.bench_function("sherman_rinzel_keizer_1k_steps", |b| {
        b.iter(|| {
            let mut n = ShermanRinzelKeizerNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("butera_respiratory_1k_steps", |b| {
        b.iter(|| {
            let mut n = ButeraRespiratoryNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(50.0));
            }
        })
    });

    c.bench_function("eprop_alif_10k_steps", |b| {
        b.iter(|| {
            let mut n = EPropALIFNeuron::default();
            for _ in 0..10_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("superspike_10k_steps", |b| {
        b.iter(|| {
            let mut n = SuperSpikeNeuron::default();
            for _ in 0..10_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("learnable_neuron_10k_steps", |b| {
        b.iter(|| {
            let mut n = LearnableNeuronModel::new();
            for _ in 0..10_000 {
                black_box(n.step(2.0));
            }
        })
    });

    c.bench_function("pernarowski_10k_steps", |b| {
        b.iter(|| {
            let mut n = PernarowskiNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(1.0));
            }
        })
    });

    // -- Trivial IF variants --

    c.bench_function("qif_100k_steps", |b| {
        b.iter(|| {
            let mut n = QuadraticIFNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("theta_100k_steps", |b| {
        b.iter(|| {
            let mut n = ThetaNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("perfect_integrator_100k_steps", |b| {
        b.iter(|| {
            let mut n = PerfectIntegratorNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("gated_lif_100k_steps", |b| {
        b.iter(|| {
            let mut n = GatedLIFNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("nlif_10k_steps", |b| {
        b.iter(|| {
            let mut n = NonlinearLIFNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(500.0));
            }
        })
    });

    c.bench_function("sfa_10k_steps", |b| {
        b.iter(|| {
            let mut n = SFANeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(30.0));
            }
        })
    });

    c.bench_function("mat_10k_steps", |b| {
        b.iter(|| {
            let mut n = MATNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(30.0));
            }
        })
    });

    c.bench_function("escape_rate_10k_steps", |b| {
        b.iter(|| {
            let mut n = EscapeRateNeuron::new(42);
            for _ in 0..10_000 {
                black_box(n.step(30.0));
            }
        })
    });

    c.bench_function("klif_100k_steps", |b| {
        b.iter(|| {
            let mut n = KLIFNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("ilif_100k_steps", |b| {
        b.iter(|| {
            let mut n = InhibitoryLIFNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(0.8));
            }
        })
    });

    c.bench_function("clif_100k_steps", |b| {
        b.iter(|| {
            let mut n = ComplementaryLIFNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("plif_100k_steps", |b| {
        b.iter(|| {
            let mut n = ParametricLIFNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(1.5));
            }
        })
    });

    c.bench_function("nrlif_10k_steps", |b| {
        b.iter(|| {
            let mut n = NonResettingLIFNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(30.0));
            }
        })
    });

    c.bench_function("atif_10k_steps", |b| {
        b.iter(|| {
            let mut n = AdaptiveThresholdIFNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(30.0));
            }
        })
    });

    c.bench_function("sigma_delta_100k_steps", |b| {
        b.iter(|| {
            let mut n = SigmaDeltaNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(0.3));
            }
        })
    });

    c.bench_function("energy_lif_10k_steps", |b| {
        b.iter(|| {
            let mut n = EnergyLIFNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(30.0));
            }
        })
    });

    c.bench_function("iqif_100k_steps", |b| {
        b.iter(|| {
            let mut n = IntegerQIFNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(50));
            }
        })
    });

    c.bench_function("cfc_100k_steps", |b| {
        b.iter(|| {
            let mut n = ClosedFormContinuousNeuron {
                v_threshold: 0.9,
                ..ClosedFormContinuousNeuron::new()
            };
            for _ in 0..100_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("stochastic_lif_10k_steps", |b| {
        b.iter(|| {
            let mut n = StochasticLIFNeuron::new(42);
            for _ in 0..10_000 {
                black_box(n.step(2.0));
            }
        })
    });

    // -- Special / stochastic / population models --

    c.bench_function("poisson_100k_steps", |b| {
        b.iter(|| {
            let mut n = PoissonNeuron::new(200.0, 1.0, 42);
            for _ in 0..100_000 {
                black_box(n.step(-1.0));
            }
        })
    });

    c.bench_function("inhom_poisson_100k_steps", |b| {
        b.iter(|| {
            let mut n = InhomogeneousPoissonNeuron::new(1.0, 42);
            for _ in 0..100_000 {
                black_box(n.step(200.0));
            }
        })
    });

    c.bench_function("gamma_renewal_100k_steps", |b| {
        b.iter(|| {
            let mut n = GammaRenewalNeuron::new(100.0, 3, 42);
            for _ in 0..100_000 {
                black_box(n.step(-1.0));
            }
        })
    });

    c.bench_function("stochastic_if_10k_steps", |b| {
        b.iter(|| {
            let mut n = StochasticIFNeuron::new(42);
            for _ in 0..10_000 {
                black_box(n.step(30.0));
            }
        })
    });

    c.bench_function("galves_locherbach_10k_steps", |b| {
        b.iter(|| {
            let mut n = GalvesLocherbachNeuron::new(42);
            for _ in 0..10_000 {
                black_box(n.step(2.0));
            }
        })
    });

    c.bench_function("spike_response_10k_steps", |b| {
        b.iter(|| {
            let mut n = SpikeResponseNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(10.0));
            }
        })
    });

    c.bench_function("glm_10k_steps", |b| {
        b.iter(|| {
            let mut n = GLMNeuron::new(5, 10, 42);
            for _ in 0..10_000 {
                black_box(n.step(20.0));
            }
        })
    });

    c.bench_function("wilson_cowan_100k_steps", |b| {
        b.iter(|| {
            let mut n = WilsonCowanUnit::new();
            for _ in 0..100_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("jansen_rit_100k_steps", |b| {
        b.iter(|| {
            let mut n = JansenRitUnit::new();
            for _ in 0..100_000 {
                black_box(n.step(220.0));
            }
        })
    });

    c.bench_function("wong_wang_100k_steps", |b| {
        b.iter(|| {
            let mut n = WongWangUnit::new(42);
            for _ in 0..100_000 {
                black_box(n.step(0.02, 0.0));
            }
        })
    });

    c.bench_function("ek_population_100k_steps", |b| {
        b.iter(|| {
            let mut n = ErmentroutKopellPopulation::new();
            for _ in 0..100_000 {
                black_box(n.step(0.0));
            }
        })
    });

    c.bench_function("wendling_100k_steps", |b| {
        b.iter(|| {
            let mut n = WendlingNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(220.0));
            }
        })
    });

    c.bench_function("larter_breakspear_100k_steps", |b| {
        b.iter(|| {
            let mut n = LarterBreakspearNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(0.0));
            }
        })
    });

    // -- Rate models --

    c.bench_function("sigmoid_rate_100k_steps", |b| {
        b.iter(|| {
            let mut n = SigmoidRateNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("threshold_linear_100k_steps", |b| {
        b.iter(|| {
            let mut n = ThresholdLinearRateNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("astrocyte_10k_steps", |b| {
        b.iter(|| {
            let mut n = AstrocyteModel::new();
            for _ in 0..10_000 {
                black_box(n.step(0.1));
            }
        })
    });

    c.bench_function("tsodyks_markram_10k_steps", |b| {
        b.iter(|| {
            let mut n = TsodyksMarkramNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(50.0, false));
            }
        })
    });

    c.bench_function("ltc_10k_steps", |b| {
        b.iter(|| {
            let mut n = LiquidTimeConstantNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("compte_wm_10k_steps", |b| {
        b.iter(|| {
            let mut n = CompteWMNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0, false));
            }
        })
    });

    c.bench_function("parallel_spiking_10k_steps", |b| {
        b.iter(|| {
            let mut n = ParallelSpikingNeuron::new(4, 0.5);
            for _ in 0..10_000 {
                black_box(n.step(1.0));
            }
        })
    });

    c.bench_function("fractional_lif_10k_steps", |b| {
        b.iter(|| {
            let mut n = FractionalLIFNeuron::new(0.8, 50);
            for _ in 0..10_000 {
                black_box(n.step(2.0));
            }
        })
    });

    c.bench_function("siegert_100k_steps", |b| {
        b.iter(|| {
            let n = SiegertTransferFunction::new();
            for i in 0..100_000 {
                black_box(n.step(i as f64 * 0.001));
            }
        })
    });

    c.bench_function("amari_field_10k_steps", |b| {
        b.iter(|| {
            let mut n = AmariNeuralField::new(32);
            let inp = vec![0.5; 32];
            for _ in 0..10_000 {
                black_box(n.step(&inp));
            }
        })
    });

    c.bench_function("leaky_compete_fire_10k_steps", |b| {
        b.iter(|| {
            let mut n = LeakyCompeteFireNeuron::new(4);
            let inp = vec![3.0; 4];
            for _ in 0..10_000 {
                black_box(n.step(&inp));
            }
        })
    });

    // -- Hardware neuromorphic --

    c.bench_function("loihi_cuba_100k_steps", |b| {
        b.iter(|| {
            let mut n = LoihiCUBANeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(100));
            }
        })
    });

    c.bench_function("loihi2_100k_steps", |b| {
        b.iter(|| {
            let mut n = Loihi2Neuron {
                tau3: 8,
                ..Loihi2Neuron::new()
            };
            for _ in 0..100_000 {
                black_box(n.step(200));
            }
        })
    });

    c.bench_function("truenorth_100k_steps", |b| {
        b.iter(|| {
            let mut n = TrueNorthNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(50));
            }
        })
    });

    c.bench_function("brainscales_1k_steps", |b| {
        b.iter(|| {
            let mut n = BrainScaleSAdExNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(500.0));
            }
        })
    });

    c.bench_function("spinnaker_lif_10k_steps", |b| {
        b.iter(|| {
            let mut n = SpiNNakerLIFNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(30.0));
            }
        })
    });

    c.bench_function("spinnaker2_100k_steps", |b| {
        b.iter(|| {
            let mut n = SpiNNaker2Neuron::new();
            for _ in 0..100_000 {
                black_box(n.step(100));
            }
        })
    });

    c.bench_function("dpi_100k_steps", |b| {
        b.iter(|| {
            let mut n = DPINeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(1.0));
            }
        })
    });

    c.bench_function("akida_100k_steps", |b| {
        b.iter(|| {
            let mut n = AkidaNeuron::default();
            for _ in 0..100_000 {
                black_box(n.step(50.0));
            }
        })
    });

    c.bench_function("neurogrid_1k_steps", |b| {
        b.iter(|| {
            let mut n = NeuroGridNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(500.0));
            }
        })
    });

    // -- AI-optimised --

    c.bench_function("multi_timescale_10k_steps", |b| {
        b.iter(|| {
            let mut n = MultiTimescaleNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("attention_gated_10k_steps", |b| {
        b.iter(|| {
            let mut n = AttentionGatedNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("predictive_coding_10k_steps", |b| {
        b.iter(|| {
            let mut n = PredictiveCodingNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("self_referential_10k_steps", |b| {
        b.iter(|| {
            let mut n = SelfReferentialNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("compositional_binding_10k_steps", |b| {
        b.iter(|| {
            let mut n = CompositionalBindingNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("differentiable_surrogate_10k_steps", |b| {
        b.iter(|| {
            let mut n = DifferentiableSurrogateNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("continuous_attractor_10k_steps", |b| {
        b.iter(|| {
            let mut n = ContinuousAttractorNeuron::new(16);
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("meta_plastic_10k_steps", |b| {
        b.iter(|| {
            let mut n = MetaPlasticNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("arcane_10k_steps", |b| {
        b.iter(|| {
            let mut n = ArcaneNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });

    // -- Multi-compartment --

    c.bench_function("pinsky_rinzel_1k_steps", |b| {
        b.iter(|| {
            let mut n = PinskyRinzelNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0, 0.0));
            }
        })
    });

    c.bench_function("hay_l5_1k_steps", |b| {
        b.iter(|| {
            let mut n = HayL5PyramidalNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(20.0, 0.0));
            }
        })
    });

    c.bench_function("marder_stg_1k_steps", |b| {
        b.iter(|| {
            let mut n = MarderSTGNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("rall_cable_1k_steps", |b| {
        b.iter(|| {
            let mut n = RallCableNeuron::new(5);
            for _ in 0..1_000 {
                black_box(n.step(50.0));
            }
        })
    });

    c.bench_function("booth_rinzel_1k_steps", |b| {
        b.iter(|| {
            let mut n = BoothRinzelNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(5.0));
            }
        })
    });

    c.bench_function("dendrify_1k_steps", |b| {
        b.iter(|| {
            let mut n = DendrifyNeuron::new();
            for _ in 0..1_000 {
                black_box(n.step(50.0));
            }
        })
    });

    c.bench_function("two_comp_lif_10k_steps", |b| {
        b.iter(|| {
            let mut n = TwoCompartmentLIFNeuron::new();
            for _ in 0..10_000 {
                black_box(n.step(0.5, 0.3));
            }
        })
    });

    // -- Maps (remaining 6) --

    c.bench_function("chialvo_100k_steps", |b| {
        b.iter(|| {
            let mut n = ChialvoMapNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(0.04));
            }
        })
    });

    c.bench_function("rulkov_100k_steps", |b| {
        b.iter(|| {
            let mut n = RulkovMapNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(0.001));
            }
        })
    });

    c.bench_function("ibarz_tanaka_100k_steps", |b| {
        b.iter(|| {
            let mut n = IbarzTanakaMapNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(0.0));
            }
        })
    });

    c.bench_function("medvedev_100k_steps", |b| {
        b.iter(|| {
            let mut n = MedvedevMapNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(0.5));
            }
        })
    });

    c.bench_function("cazelles_100k_steps", |b| {
        b.iter(|| {
            let mut n = CazellesMapNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(0.0));
            }
        })
    });

    c.bench_function("courage_nekorkin_100k_steps", |b| {
        b.iter(|| {
            let mut n = CourageNekorkinMapNeuron::new();
            for _ in 0..100_000 {
                black_box(n.step(0.0));
            }
        })
    });

    // -- Sensory (remaining 3) --

    c.bench_function("outer_hair_cell_10k_steps", |b| {
        b.iter(|| {
            let mut n = OuterHairCell::new();
            for _ in 0..10_000 {
                black_box(n.step(0.01));
            }
        })
    });

    c.bench_function("cone_photoreceptor_10k_steps", |b| {
        b.iter(|| {
            let mut n = ConePhotoreceptor::new();
            for _ in 0..10_000 {
                black_box(n.step(1000.0));
            }
        })
    });

    c.bench_function("taste_receptor_10k_steps", |b| {
        b.iter(|| {
            let mut n = TasteReceptorCell::new();
            for _ in 0..10_000 {
                black_box(n.step(5.0));
            }
        })
    });
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
