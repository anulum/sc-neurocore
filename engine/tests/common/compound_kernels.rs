// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compound circuits against independent integer trajectories

//! Quantized graph, attention and phase RTL kernels.

use super::{ir_stream, random, signed};
use sc_neurocore_engine::ir::{builder::ScGraphBuilder, graph::*};

/// Keep numerical references separate from compiler quantization and layout helpers.
fn q16(value: f64) -> i128 {
    (value * 65536.0).round() as i128
}

fn constant_vector(b: &mut ScGraphBuilder, values: &[f64]) -> ValueId {
    b.constant(
        ScConst::F64Vec(values.to_vec()),
        ScType::Vec {
            element: Box::new(ScType::Rate),
            count: values.len(),
        },
    )
}

/// Compare all packed lanes, including outputs wider than an integer machine word.
fn kernel_run(label: &str, graph: &ScGraph, words: &[i128]) {
    let packed = words
        .iter()
        .rev()
        .map(|word| format!("{:06x}", *word as u128 & 0xffffff))
        .collect::<String>();
    let sum = words.iter().map(|v| signed(*v as u128, 24)).sum::<i128>() as u64;
    let bench = format!("module tb; reg clk=0,rst_n=1; wire [{}:0] result; wire [63:0] sum; kernel_runtime dut(.*); initial begin #2; $display(\"OBS %h %h\",result,sum); $finish; end endmodule\n",words.len()*24-1);
    ir_stream::run(label, graph, &bench, &[format!("OBS {packed} {sum:016x}")]);
}

/// Normalize once after accumulation; a zero signed degree retains unnormalized products.
#[test]
fn graph_kernel_handles_signed_degrees_and_cancellation() {
    let mut rng = 227_u128;
    let mut cases = vec![
        (2, 2, vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 1.0, 1.0, 0.0]),
        (2, 2, vec![1.0, -2.0, 3.0, 4.0], vec![1.0, -1.0, 0.0, 0.0]),
    ];
    for (nodes, columns) in [(1, 1), (2, 3), (3, 1), (4, 2)] {
        for _ in 0..2 {
            let x = (0..nodes * columns)
                .map(|_| (random(&mut rng, 7) as f64 - 64.0) / 16.0)
                .collect();
            let a = (0..nodes * nodes)
                .map(|_| [-1.0, 0.0, 0.5, 1.0][random(&mut rng, 2) as usize])
                .collect();
            cases.push((nodes, columns, x, a));
        }
    }
    for (case, (nodes, columns, x, a)) in cases.into_iter().enumerate() {
        let mut b = ScGraphBuilder::new("kernel_runtime");
        let mut vector = |values: &[f64]| {
            if case == 0 {
                b.constant(
                    ScConst::I64Vec(values.iter().map(|x| *x as i64).collect()),
                    ScType::Vec {
                        element: Box::new(ScType::Rate),
                        count: values.len(),
                    },
                )
            } else {
                constant_vector(&mut b, values)
            }
        };
        let features = vector(&x);
        let adjacency = vector(&a);
        let output = b.graph_forward(features, adjacency, nodes, columns);
        b.output("result", output);
        let sum = b.reduce(output, ReduceMode::Sum);
        b.output("sum", sum);
        let x: Vec<_> = x.into_iter().map(q16).collect();
        let a: Vec<_> = a.into_iter().map(q16).collect();
        let mut expected = Vec::new();
        for row in 0..nodes {
            let degree = signed(
                a[row * nodes..(row + 1) * nodes].iter().sum::<i128>() as u128,
                32,
            );
            for column in 0..columns {
                let sum = signed(
                    (0..nodes)
                        .map(|j| a[row * nodes + j] * x[j * columns + column])
                        .sum::<i128>() as u128,
                    56,
                );
                expected.push(if degree == 0 { sum >> 16 } else { sum / degree });
            }
        }
        kernel_run(&format!("compound-graph-{case}"), &b.build(), &expected);
    }
}

/// Stable score subtraction, finite exp grid and full output geometry survive lowering.
#[test]
fn attention_kernel_matches_quantized_normalization_and_weighted_values() {
    let mut rng = 991_u128;
    for (shape, (queries, keys, dim, columns)) in [
        (1, 1, 1, 2),
        (1, 2, 2, 1),
        (2, 3, 3, 2),
        (3, 2, 4, 3),
        (1, 4, 2, 2),
    ]
    .into_iter()
    .enumerate()
    {
        for (case, scale) in [0.0, 2.0, 127.0].into_iter().enumerate() {
            let mut values = |len, scale| {
                (0..len)
                    .map(|_| (random(&mut rng, 12) as f64 / 4096.0 * 2.0 - 1.0) * scale)
                    .collect::<Vec<_>>()
            };
            let q = values(queries * dim, scale);
            let k = values(keys * dim, scale);
            let v = values(keys * columns, 4.0);
            let mut b = ScGraphBuilder::new("kernel_runtime");
            let query = constant_vector(&mut b, &q);
            let key = constant_vector(&mut b, &k);
            let value = constant_vector(&mut b, &v);
            let output = b.softmax_attention(query, key, value, dim);
            b.output("result", output);
            let sum = b.reduce(output, ReduceMode::Sum);
            b.output("sum", sum);
            let q: Vec<_> = q.into_iter().map(q16).collect();
            let k: Vec<_> = k.into_iter().map(q16).collect();
            let v: Vec<_> = v.into_iter().map(q16).collect();
            let temperature = q16(1.0 / (dim as f64).sqrt());
            let mut expected = Vec::new();
            for row in 0..queries {
                let scores: Vec<_> = (0..keys)
                    .map(|j| {
                        let dot = signed(
                            (0..dim)
                                .map(|d| q[row * dim + d] * k[j * dim + d])
                                .sum::<i128>() as u128,
                            56,
                        );
                        signed((signed((dot * temperature) as u128, 80) >> 32) as u128, 24)
                    })
                    .collect();
                let max = *scores.iter().max().unwrap();
                let exp: Vec<_> = scores
                    .iter()
                    .map(|score| {
                        let difference = signed((score - max) as u128, 24);
                        let index = ((difference + 1048576) >> 13).clamp(0, 255);
                        (((-16.0 + index as f64 / 8.0).exp() * 65536.0).round() as i128)
                            .min(8388607)
                    })
                    .collect();
                let sum = signed(exp.iter().sum::<i128>() as u128, 32);
                let weights: Vec<_> = exp
                    .iter()
                    .map(|x| signed(((x << 16) / sum) as u128, 24))
                    .collect();
                for column in 0..columns {
                    let value = signed(
                        (0..keys)
                            .map(|j| weights[j] * v[j * columns + column])
                            .sum::<i128>() as u128,
                        56,
                    ) >> 16;
                    expected.push(value);
                }
            }
            kernel_run(
                &format!("compound-attention-{shape}-{case}"),
                &b.build(),
                &expected,
            );
        }
    }
}

/// Evaluate the documented physical phase contract without compiler lookup functions.
fn phase_reference(phases: &[f64], omega: &[f64], coupling: &[f64], dt: f64) -> Vec<i128> {
    let phases: Vec<_> = phases
        .iter()
        .map(|x| q16(x.rem_euclid(std::f64::consts::TAU)))
        .collect();
    let omega: Vec<_> = omega.iter().copied().map(q16).collect();
    let coupling: Vec<_> = coupling.iter().copied().map(q16).collect();
    let n = phases.len();
    let mut output = Vec::new();
    for i in 0..n {
        let mut sum = 0;
        for j in 0..n {
            let mut delta = signed((phases[j] - phases[i]) as u128, 24);
            if delta > 205887 {
                delta = signed((delta - 411775) as u128, 24);
            } else if delta < -205887 {
                delta = signed((delta + 411775) as u128, 24);
            }
            if delta >= 411775 {
                delta = signed((delta - 411775) as u128, 24);
            } else if delta < 0 {
                delta = signed((delta + 411775) as u128, 24);
            }
            let bin = (delta * 64 / 411775) & 63;
            let sine = q16((std::f64::consts::TAU * bin as f64 / 64.0).sin());
            sum = signed((sum + coupling[i * n + j] * sine) as u128, 56);
        }
        let term = signed((sum >> 16) as u128, 24);
        let velocity = signed((omega[i] + term) as u128, 24);
        let increment = signed((signed((velocity * q16(dt)) as u128, 48) >> 16) as u128, 24);
        let mut value = signed((phases[i] + increment) as u128, 24);
        if value >= 411775 {
            value = signed((value - 411775) as u128, 24);
        } else if value < 0 {
            value = signed((value + 411775) as u128, 24);
        }
        output.push(value);
    }
    output
}

/// Cover every sine bin, both wrap directions, negative dt and multi-node coupling.
#[test]
fn phase_kernel_matches_all_grid_bins_and_signed_steps() {
    let mut cases = Vec::new();
    for bin in 0..64 {
        cases.push((
            vec![0.0, (bin as f64 + 0.5) * std::f64::consts::TAU / 64.0],
            vec![0.1, -0.1],
            vec![0.0, 0.25, -0.5, 0.0],
            0.01,
        ));
    }
    cases.extend([
        (
            vec![0.01, 6.27],
            vec![-1.0, 1.0],
            vec![0.0, 0.2, 0.2, 0.0],
            0.2,
        ),
        (
            vec![-6.2, 7.1, 127.0],
            vec![0.2, -0.1, 0.3],
            vec![0.0, 0.2, -0.1, 0.1, 0.0, 0.3, -0.2, 0.1, 0.0],
            0.01,
        ),
        (vec![1.0], vec![1.0], vec![0.0], 0.0),
        (vec![1.0], vec![1.0], vec![0.0], -0.01),
    ]);
    for (case, (phases, omega, coupling, dt)) in cases.into_iter().enumerate() {
        let mut b = ScGraphBuilder::new("kernel_runtime");
        let p = constant_vector(&mut b, &phases);
        let o = constant_vector(&mut b, &omega);
        let c = constant_vector(&mut b, &coupling);
        let output = b.kuramoto_step(p, o, c, dt);
        b.output("result", output);
        let sum = b.reduce(output, ReduceMode::Sum);
        b.output("sum", sum);
        kernel_run(
            &format!("compound-kuramoto-{case}"),
            &b.build(),
            &phase_reference(&phases, &omega, &coupling, dt),
        );
    }
}
