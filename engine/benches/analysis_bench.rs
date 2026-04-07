// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Analysis Module Benchmarks (all 22 P0-A modules)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sc_neurocore_engine::analysis;
use std::hint::black_box;

// ── Test data generators ────────────────────────────────────────────

fn make_binary_train(n: usize, rate: f64, seed: u64) -> Vec<i32> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 33) as f64 / (1u64 << 31) as f64 <= rate {
                1
            } else {
                0
            }
        })
        .collect()
}

fn make_f64_train(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f64 / (1u64 << 31) as f64
        })
        .collect()
}

fn make_waveform(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64 * 2.0 * std::f64::consts::PI;
            -t.sin() + 0.3 * (2.0 * t).sin()
        })
        .collect()
}

fn make_feature_matrix(n: usize, d: usize, offset: f64, seed: u64) -> Vec<f64> {
    let mut rng = seed;
    (0..n * d)
        .map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            offset + (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        })
        .collect()
}

// ── Benchmarks ──────────────────────────────────────────────────────

fn bench_basic(c: &mut Criterion) {
    let mut g = c.benchmark_group("basic");
    for &n in &[100, 10_000, 100_000] {
        let train = make_binary_train(n, 0.05, 42);
        g.bench_with_input(BenchmarkId::new("spike_times", n), &train, |b, t| {
            b.iter(|| black_box(analysis::basic::spike_times(t, 0.001)))
        });
        g.bench_with_input(BenchmarkId::new("isi", n), &train, |b, t| {
            b.iter(|| black_box(analysis::basic::isi(t, 0.001)))
        });
        g.bench_with_input(BenchmarkId::new("firing_rate", n), &train, |b, t| {
            b.iter(|| black_box(analysis::basic::firing_rate(t, 0.001)))
        });
        g.bench_with_input(BenchmarkId::new("bin_spike_train", n), &train, |b, t| {
            b.iter(|| black_box(analysis::basic::bin_spike_train(t, 10)))
        });
    }
    g.finish();
}

fn bench_rate(c: &mut Criterion) {
    let mut g = c.benchmark_group("rate");
    for &n in &[100, 10_000, 100_000] {
        let train: Vec<f64> = make_binary_train(n, 0.05, 42)
            .into_iter()
            .map(|v| v as f64)
            .collect();
        g.bench_with_input(BenchmarkId::new("instantaneous_rate", n), &train, |b, t| {
            b.iter(|| {
                black_box(analysis::rate::instantaneous_rate(
                    t, 0.001, "gaussian", 10.0,
                ))
            })
        });
    }
    g.finish();
}

fn bench_variability(c: &mut Criterion) {
    let mut g = c.benchmark_group("variability");
    for &n in &[100, 10_000, 100_000] {
        let train = make_binary_train(n, 0.05, 42);
        g.bench_with_input(BenchmarkId::new("cv_isi", n), &train, |b, t| {
            b.iter(|| black_box(analysis::variability::cv_isi(t, 0.001)))
        });
        g.bench_with_input(BenchmarkId::new("fano_factor", n), &train, |b, t| {
            b.iter(|| black_box(analysis::variability::fano_factor(t, 50.0, 0.001)))
        });
    }
    // sample_entropy is O(n^2) — only bench small inputs
    for &n in &[100, 1_000] {
        let train = make_binary_train(n, 0.05, 42);
        g.bench_with_input(BenchmarkId::new("sample_entropy", n), &train, |b, t| {
            b.iter(|| black_box(analysis::variability::sample_entropy(t, 2, 0.2)))
        });
    }
    g.finish();
}

fn bench_correlation(c: &mut Criterion) {
    let mut g = c.benchmark_group("correlation");
    for &n in &[100, 10_000] {
        let a = make_binary_train(n, 0.05, 42);
        let b = make_binary_train(n, 0.05, 99);
        g.bench_with_input(BenchmarkId::new("cross_correlation", n), &n, |bench, _| {
            bench.iter(|| {
                black_box(analysis::correlation::cross_correlation(
                    &a, &b, 50.0, 0.001,
                ))
            })
        });
        g.bench_with_input(
            BenchmarkId::new("event_synchronization", n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    black_box(analysis::correlation::event_synchronization(
                        &a, &b, 0.001, 5.0,
                    ))
                })
            },
        );
    }
    g.finish();
}

fn bench_distance(c: &mut Criterion) {
    let mut g = c.benchmark_group("distance");
    for &n in &[100, 5_000] {
        let a = make_binary_train(n, 0.05, 42);
        let b = make_binary_train(n, 0.05, 99);
        g.bench_with_input(BenchmarkId::new("van_rossum", n), &n, |bench, _| {
            bench.iter(|| black_box(analysis::distance::van_rossum_distance(&a, &b, 0.001, 10.0)))
        });
        let times_a = analysis::basic::spike_times(&a, 0.001);
        let times_b = analysis::basic::spike_times(&b, 0.001);
        g.bench_with_input(BenchmarkId::new("victor_purpura", n), &n, |bench, _| {
            bench.iter(|| {
                black_box(analysis::distance::victor_purpura_distance(
                    &times_a, &times_b, 100.0,
                ))
            })
        });
        g.bench_with_input(BenchmarkId::new("isi_distance", n), &n, |bench, _| {
            bench.iter(|| black_box(analysis::distance::isi_distance(&a, &b, 0.001)))
        });
    }
    g.finish();
}

fn bench_information(c: &mut Criterion) {
    let mut g = c.benchmark_group("information");
    for &n in &[100, 10_000] {
        let a = make_binary_train(n, 0.05, 42);
        let b = make_binary_train(n, 0.05, 99);
        g.bench_with_input(BenchmarkId::new("mutual_information", n), &n, |bench, _| {
            bench.iter(|| black_box(analysis::information::mutual_information(&a, &b, 10)))
        });
        g.bench_with_input(BenchmarkId::new("transfer_entropy", n), &n, |bench, _| {
            bench.iter(|| black_box(analysis::information::transfer_entropy(&a, &b, 10, 1)))
        });
    }
    g.finish();
}

fn bench_causality(c: &mut Criterion) {
    let mut g = c.benchmark_group("causality");
    for &n in &[100, 5_000] {
        let a = make_binary_train(n, 0.05, 42);
        let b = make_binary_train(n, 0.05, 99);
        g.bench_with_input(BenchmarkId::new("pairwise_granger", n), &n, |bench, _| {
            bench.iter(|| {
                black_box(analysis::causality::pairwise_granger_causality(
                    &a, &b, 10, 5,
                ))
            })
        });
    }
    g.finish();
}

fn bench_decoding(c: &mut Criterion) {
    let mut g = c.benchmark_group("decoding");
    let n_neurons = 20;
    let _n_trials = 50;
    let n_directions = 8;

    let mut trains = Vec::new();
    for i in 0..n_neurons {
        trains.push(make_binary_train(
            1000,
            0.03 + 0.005 * i as f64,
            42 + i as u64,
        ));
    }
    let directions: Vec<f64> = (0..n_neurons)
        .map(|i| i as f64 / n_neurons as f64 * 2.0 * std::f64::consts::PI)
        .collect();
    let window = 100;

    let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
    g.bench_function("population_vector_decode", |b| {
        b.iter(|| {
            black_box(analysis::decoding::population_vector_decode(
                &refs,
                &directions,
                window,
            ))
        })
    });

    // Bayesian decode
    let counts: Vec<f64> = trains
        .iter()
        .map(|t| t.iter().map(|&v| v as f64).sum())
        .collect();
    let tuning = make_feature_matrix(n_directions, n_neurons, 5.0, 42);
    let prior: Vec<f64> = vec![1.0 / n_directions as f64; n_directions];
    g.bench_function("bayesian_decode", |b| {
        b.iter(|| {
            black_box(analysis::decoding::bayesian_decode(
                &counts,
                &tuning,
                n_directions,
                n_neurons,
                &prior,
            ))
        })
    });
    g.finish();
}

fn bench_network(c: &mut Criterion) {
    let mut g = c.benchmark_group("network");
    let n_neurons = 10;
    let mut trains = Vec::new();
    for i in 0..n_neurons {
        trains.push(make_binary_train(2000, 0.04, 42 + i as u64));
    }
    let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
    g.bench_function("functional_connectivity_10n", |b| {
        b.iter(|| {
            black_box(analysis::network::functional_connectivity(
                &refs, 20.0, 0.001,
            ))
        })
    });
    g.finish();
}

fn bench_surrogates(c: &mut Criterion) {
    let mut g = c.benchmark_group("surrogates");
    for &n in &[1_000, 100_000] {
        let train = make_binary_train(n, 0.05, 42);
        g.bench_with_input(BenchmarkId::new("isi_shuffle", n), &train, |b, t| {
            b.iter(|| black_box(analysis::surrogates::surrogate_isi_shuffle(t, 42)))
        });
        g.bench_with_input(BenchmarkId::new("homogeneous_poisson", n), &n, |b, &sz| {
            b.iter(|| {
                black_box(analysis::surrogates::homogeneous_poisson(
                    50.0,
                    sz as f64 * 0.001,
                    0.001,
                    42,
                ))
            })
        });
    }
    g.finish();
}

fn bench_temporal(c: &mut Criterion) {
    let mut g = c.benchmark_group("temporal");
    for &n in &[1_000, 100_000] {
        let train = make_binary_train(n, 0.05, 42);
        g.bench_with_input(BenchmarkId::new("burst_detection", n), &train, |b, t| {
            b.iter(|| black_box(analysis::temporal::burst_detection(t, 0.001, 10.0, 3)))
        });
        g.bench_with_input(
            BenchmarkId::new("change_point_detection", n),
            &train,
            |b, t| b.iter(|| black_box(analysis::temporal::change_point_detection(t, 50, 3.0))),
        );
    }
    g.finish();
}

fn bench_patterns(c: &mut Criterion) {
    let mut g = c.benchmark_group("patterns");
    let n = 5_000;
    let a_train = make_binary_train(n, 0.05, 42);
    let b_train = make_binary_train(n, 0.05, 99);
    let times_a = analysis::basic::spike_times(&a_train, 0.001);
    let times_b = analysis::basic::spike_times(&b_train, 0.001);
    g.bench_function("spike_directionality_5k", |b| {
        b.iter(|| {
            black_box(analysis::patterns::spike_directionality(
                &times_a, &times_b, 0.0, 5.0,
            ))
        })
    });
    g.bench_function("cubic_higher_order_5k", |b| {
        b.iter(|| black_box(analysis::patterns::cubic_higher_order(&a_train, 0.001, 20)))
    });
    g.finish();
}

fn bench_spectral(c: &mut Criterion) {
    let mut g = c.benchmark_group("spectral");
    for &n in &[256, 10_000, 100_000] {
        let train = make_binary_train(n, 0.05, 42);
        g.bench_with_input(BenchmarkId::new("power_spectrum", n), &train, |b, t| {
            b.iter(|| black_box(analysis::spectral::power_spectrum(t, 0.001)))
        });
    }
    g.finish();
}

fn bench_waveform(c: &mut Criterion) {
    let mut g = c.benchmark_group("waveform");
    let wf = make_waveform(64);
    let dt = 1.0 / 30000.0;
    g.bench_function("waveform_width", |b| {
        b.iter(|| black_box(analysis::waveform::waveform_width(&wf, dt)))
    });
    g.bench_function("waveform_amplitude", |b| {
        b.iter(|| black_box(analysis::waveform::waveform_amplitude(&wf)))
    });
    g.bench_function("waveform_repolarization_slope", |b| {
        b.iter(|| black_box(analysis::waveform::waveform_repolarization_slope(&wf, dt)))
    });
    g.bench_function("waveform_halfwidth", |b| {
        b.iter(|| black_box(analysis::waveform::waveform_halfwidth(&wf, dt)))
    });
    g.bench_function("waveform_pt_ratio", |b| {
        b.iter(|| black_box(analysis::waveform::waveform_pt_ratio(&wf)))
    });
    g.finish();
}

fn bench_point_process(c: &mut Criterion) {
    let mut g = c.benchmark_group("point_process");
    for &n in &[1_000, 100_000] {
        let train = make_binary_train(n, 0.05, 42);
        g.bench_with_input(
            BenchmarkId::new("conditional_intensity", n),
            &train,
            |b, t| {
                b.iter(|| {
                    black_box(analysis::point_process::conditional_intensity(
                        t, 0.001, 50.0,
                    ))
                })
            },
        );
        g.bench_with_input(BenchmarkId::new("isi_hazard", n), &train, |b, t| {
            b.iter(|| black_box(analysis::point_process::isi_hazard_function(t, 0.001, 30)))
        });
    }
    g.finish();
}

fn bench_statistics(c: &mut Criterion) {
    let mut g = c.benchmark_group("statistics");
    let a: Vec<f64> = make_f64_train(500, 42);
    let b_data: Vec<f64> = make_f64_train(500, 99);
    g.bench_function("significance_bootstrap_200surr", |b| {
        b.iter(|| {
            black_box(analysis::statistics::significance_bootstrap(
                |x, y| {
                    let ma = x.iter().sum::<f64>() / x.len() as f64;
                    let mb = y.iter().sum::<f64>() / y.len() as f64;
                    ma - mb
                },
                &a,
                &b_data,
                200,
                42,
            ))
        })
    });
    g.finish();
}

fn bench_stimulus(c: &mut Criterion) {
    let mut g = c.benchmark_group("stimulus");
    for &n in &[1_000, 50_000] {
        let stim = make_f64_train(n, 42);
        let train = make_binary_train(n, 0.05, 99);
        g.bench_with_input(BenchmarkId::new("sta", n), &n, |b, _| {
            b.iter(|| {
                black_box(analysis::stimulus::spike_triggered_average(
                    &stim, &train, 50,
                ))
            })
        });
        let positions: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 100.0).collect();
        g.bench_with_input(BenchmarkId::new("spatial_information", n), &n, |b, _| {
            b.iter(|| {
                black_box(analysis::stimulus::spatial_information(
                    &train, &positions, 20, 0.001,
                ))
            })
        });
    }
    g.finish();
}

fn bench_lfp(c: &mut Criterion) {
    let mut g = c.benchmark_group("lfp");
    for &n in &[500, 10_000] {
        let train = make_binary_train(n, 0.05, 42);
        let lfp: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 * 0.001).sin())
            .collect();
        g.bench_with_input(BenchmarkId::new("phase_locking_value", n), &n, |b, _| {
            b.iter(|| black_box(analysis::lfp::phase_locking_value(&train, &lfp)))
        });
        g.bench_with_input(BenchmarkId::new("spike_field_coherence", n), &n, |b, _| {
            b.iter(|| black_box(analysis::lfp::spike_field_coherence(&train, &lfp, 0.001)))
        });
    }
    g.finish();
}

fn bench_sorting_quality(c: &mut Criterion) {
    let mut g = c.benchmark_group("sorting_quality");
    let d = 4;
    for &n in &[50, 200] {
        let cluster = make_feature_matrix(n, d, 0.0, 42);
        let noise = make_feature_matrix(n * 2, d, 5.0, 99);
        g.bench_with_input(BenchmarkId::new("isolation_distance", n), &n, |b, _| {
            b.iter(|| {
                black_box(analysis::sorting_quality::isolation_distance(
                    &cluster,
                    n,
                    &noise,
                    n * 2,
                    d,
                ))
            })
        });
        g.bench_with_input(BenchmarkId::new("silhouette_score", n), &n, |b, _| {
            let mut features = cluster.clone();
            features.extend_from_slice(&noise[..n * d]);
            let mut labels: Vec<i64> = vec![0; n];
            labels.extend(vec![1i64; n]);
            b.iter(|| {
                black_box(analysis::sorting_quality::silhouette_score(
                    &features,
                    n * 2,
                    d,
                    &labels,
                ))
            })
        });
    }
    let train_5k = make_binary_train(5000, 0.05, 42);
    g.bench_function("isi_violation_rate_5k", |b| {
        b.iter(|| {
            black_box(analysis::sorting_quality::isi_violation_rate(
                &train_5k, 0.001, 1.5,
            ))
        })
    });
    g.finish();
}

fn bench_dimensionality(c: &mut Criterion) {
    let mut g = c.benchmark_group("dimensionality");
    let n_neurons = 10;
    let mut trains = Vec::new();
    for i in 0..n_neurons {
        trains.push(make_binary_train(
            2000,
            0.04 + 0.005 * i as f64,
            42 + i as u64,
        ));
    }
    let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
    g.bench_function("pca_10n_2000t", |b| {
        b.iter(|| black_box(analysis::dimensionality::spike_train_pca(&refs, 3, 10)))
    });
    g.bench_function("factor_analysis_10n_2000t", |b| {
        b.iter(|| black_box(analysis::dimensionality::factor_analysis(&refs, 3, 10, 20)))
    });
    g.finish();
}

fn bench_gpfa(c: &mut Criterion) {
    let mut g = c.benchmark_group("gpfa");
    g.sample_size(10); // GPFA is slow, reduce iterations
    let n_neurons = 4;
    let mut trains = Vec::new();
    for i in 0..n_neurons {
        trains.push(make_binary_train(500, 0.05, 42 + i as u64));
    }
    let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
    g.bench_function("gpfa_4n_500t_5iter", |b| {
        b.iter(|| black_box(analysis::gpfa::gpfa(&refs, 2, 10.0, 0.001, 5, 1e-4, 42)))
    });
    g.finish();
}

fn bench_spade(c: &mut Criterion) {
    let mut g = c.benchmark_group("spade");
    g.sample_size(10); // SPADE with surrogates is slow
    let n_neurons = 3;
    let n = 500;
    let mut trains = Vec::new();
    for i in 0..n_neurons {
        let mut t = make_binary_train(n, 0.03, 42 + i as u64);
        // Add synchronous events
        for j in (0..n).step_by(20) {
            t[j] = 1;
        }
        trains.push(t);
    }
    let refs: Vec<&[i32]> = trains.iter().map(|t| t.as_slice()).collect();
    g.bench_function("spade_3n_500t_50surr", |b| {
        b.iter(|| {
            black_box(analysis::spade::spade_detect(
                &refs, 5.0, 0.001, 3, 3, 50, 0.05, 42,
            ))
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_basic,
    bench_rate,
    bench_variability,
    bench_correlation,
    bench_distance,
    bench_information,
    bench_causality,
    bench_decoding,
    bench_network,
    bench_surrogates,
    bench_temporal,
    bench_patterns,
    bench_spectral,
    bench_waveform,
    bench_point_process,
    bench_statistics,
    bench_stimulus,
    bench_lfp,
    bench_sorting_quality,
    bench_dimensionality,
    bench_gpfa,
    bench_spade,
);
criterion_main!(benches);
