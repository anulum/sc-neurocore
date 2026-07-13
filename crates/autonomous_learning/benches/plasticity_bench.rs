// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Plasticity Rule Benchmarks
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

use autonomous_learning::{BcmRule, EligentRule, PlasticityRule, RewardStdpRule, StdpRule};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

const DEFAULT_DT: f32 = 0.001;

fn bench_stdp_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("stdp_step");
    for steps in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |bench, &n| {
            bench.iter(|| {
                let mut rule = StdpRule::new(0.5, 0.1, 0.05, 20.0, 20.0);
                for i in 0..n {
                    rule.step(i % 3 == 0, i % 5 == 0, 0.0, DEFAULT_DT);
                }
                black_box(rule.weight())
            });
        });
    }
    group.finish();
}

fn bench_rstdp_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("rstdp_step");
    for steps in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |bench, &n| {
            bench.iter(|| {
                let mut rule = RewardStdpRule::new(0.5, 0.1, 0.05, 20.0, 20.0, 0.95);
                for i in 0..n {
                    let reward = if i % 10 == 0 { 1.0 } else { 0.0 };
                    rule.step(i % 3 == 0, i % 5 == 0, reward, DEFAULT_DT);
                }
                black_box(rule.weight())
            });
        });
    }
    group.finish();
}

fn bench_bcm_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("bcm_step");
    for steps in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |bench, &n| {
            bench.iter(|| {
                let mut rule = BcmRule::new(0.5, 0.01, 10.0);
                for i in 0..n {
                    rule.step(i % 3 == 0, i % 5 == 0, 0.0, DEFAULT_DT);
                }
                black_box(rule.weight())
            });
        });
    }
    group.finish();
}

fn bench_eligent_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("eligent_step");
    for steps in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |bench, &n| {
            bench.iter(|| {
                let mut rule = EligentRule {
                    threshold: 1.0,
                    target_rate: 0.1,
                    eta_intrinsic: 0.001,
                    eligibility_trace: 0.0,
                    tau_e: 0.95,
                    weight: 0.5,
                    sum_weights: 0.5,
                    target_sum_weights: 1.0,
                };
                for i in 0..n {
                    let reward = if i % 10 == 0 { 1.0 } else { 0.0 };
                    rule.step(i % 3 == 0, i % 5 == 0, reward, DEFAULT_DT);
                }
                black_box(rule.weight())
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_stdp_step,
    bench_rstdp_step,
    bench_bcm_step,
    bench_eligent_step,
);
criterion_main!(benches);
