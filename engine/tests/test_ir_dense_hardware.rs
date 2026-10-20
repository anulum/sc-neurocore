// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense IR hardware numerical fidelity

//! Compare actual native graph emission to independent integer/LFSR recurrence.

#[path = "common/ir_rtl.rs"]
mod ir_rtl;
use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder, emit_sv, graph::*, parser, printer, verify,
};
use std::path::Path;

fn fixed(width: u32, frac: u32) -> ScType {
    ScType::FixedPoint { width, frac }
}
fn vector(ty: ScType, count: usize) -> ScType {
    ScType::Vec {
        element: Box::new(ty),
        count,
    }
}
fn graph(
    p: DenseParams,
    input_type: ScType,
    weight_type: ScType,
    weights: Option<Vec<i64>>,
) -> ScGraph {
    let mut b = ScGraphBuilder::new("dense_rtl");
    let inputs = b.input("inputs", vector(input_type, p.n_inputs));
    let weights = match weights {
        Some(words) => b.constant(
            ScConst::I64Vec(words),
            vector(weight_type, p.n_inputs * p.n_neurons),
        ),
        None => b.input("weights", vector(weight_type, p.n_inputs * p.n_neurons)),
    };
    let leak = b.input("leak", fixed(p.data_width, 8));
    let gain = b.input("gain", fixed(p.data_width, 8));
    let spikes = b.dense_forward(inputs, weights, leak, gain, p);
    b.output("spikes", spikes);
    b.build()
}
fn lfsr(state: u16) -> u16 {
    let feedback = ((state >> 15) ^ (state >> 13) ^ (state >> 12) ^ (state >> 10)) & 1;
    (state << 1) | feedback
}
fn signed(value: i128, width: u32) -> i128 {
    let bits = value & ((1_i128 << width) - 1);
    if bits & (1_i128 << (width - 1)) != 0 {
        bits - (1_i128 << width)
    } else {
        bits
    }
}
fn words(width: u32, frac: u32, cycle: usize, count: usize) -> Vec<i64> {
    let one = 1_i128 << frac;
    let max = (1_i128 << (width - 1)) - 1;
    let cases = [-one / 2, 0, one / 8, one / 2, one.min(max), max];
    (0..count)
        .map(|i| cases[(i + cycle) % cases.len()] as i64)
        .collect()
}
fn packed(words: &[i64], width: u32) -> String {
    let mask = (1_i128 << width) - 1;
    format!(
        "{{{}}}",
        words
            .iter()
            .rev()
            .map(|v| format!("{width}'h{:x}", i128::from(*v) & mask))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn threshold(raw: i64, frac: u32) -> u16 {
    ((i128::from(raw).max(0) * 65535) / (1_i128 << frac)).min(65535) as u16
}

#[test]
fn dense_matrix_dynamic_and_constant_operands_match_each_clock_in_both_hdl_copies() {
    for (case, (dw, iw, ifrac, ww, wfrac, constant)) in [
        (10, 16, 8, 24, 16, false),
        (16, 32, 24, 16, 8, true),
        (24, 10, 1, 32, 24, false),
        (32, 64, 63, 64, 63, true),
        (64, 16, 8, 1, 0, false),
        (16, 16, 8, 16, 8, false),
    ]
    .into_iter()
    .enumerate()
    {
        let p = DenseParams {
            n_inputs: 3,
            n_neurons: 2,
            data_width: dw,
            stream_length: 7,
            input_seed_base: 0x1234 + case as u16,
            weight_seed_base: 0x5678 + case as u16,
            y_min: if case % 2 == 0 { -3 } else { 300 },
            y_max: if case % 2 == 0 { 300 } else { -4 },
        };
        let constant_weights = words(ww, wfrac, 1, 6);
        let g = graph(
            p.clone(),
            fixed(iw, ifrac),
            fixed(ww, wfrac),
            constant.then(|| constant_weights.clone()),
        );
        verify::verify(&g).unwrap();
        let text = printer::print(&g);
        let readback = parser::parse(&text).unwrap();
        assert_eq!(g, readback);
        let sv = emit_sv::emit(&readback).unwrap();
        assert_eq!(sv, emit_sv::emit(&g).unwrap());
        let mut drive = String::new();
        let mut expected = Vec::new();
        let mut inputs_rng: Vec<_> = (0..3)
            .map(|i| p.input_seed_base.wrapping_add(i * 7))
            .collect();
        let mut weights_rng: Vec<_> = (0..6)
            .map(|i| p.weight_seed_base.wrapping_add(i * 13))
            .collect();
        let mut input_bits = [0_u8; 3];
        let mut weight_bits = [0_u8; 6];
        let mut state = [0_i128; 2];
        for cycle in 0..256 {
            if cycle == 128 {
                drive.push_str("#1; rst_n=0; #1; rst_n=1; #1;\n");
                for (i, rng) in inputs_rng.iter_mut().enumerate() {
                    *rng = p.input_seed_base.wrapping_add(i as u16 * 7);
                }
                for (i, rng) in weights_rng.iter_mut().enumerate() {
                    *rng = p.weight_seed_base.wrapping_add(i as u16 * 13);
                }
                input_bits = [0; 3];
                weight_bits = [0; 6];
                state = [0; 2];
            }
            let input_words = words(iw, ifrac, cycle / 3, 3);
            let weight_words = if constant {
                constant_weights.clone()
            } else {
                words(ww, wfrac, cycle / 5, 6)
            };
            let leak = if cycle % 11 == 0 { -7 } else { 16 };
            let gain = if cycle % 13 == 0 { -128 } else { 256 };
            let mut spikes = 0;
            let mut currents = [0_i128; 2];
            for row in 0..2 {
                let count = (0..3)
                    .filter(|i| input_bits[*i] & weight_bits[row * 3 + *i] != 0)
                    .count() as i128;
                let current =
                    i128::from(p.y_min) + (i128::from(p.y_max) - i128::from(p.y_min)) * count / 3;
                let next = signed(
                    state[row] + ((-state[row] * leak) >> 8) + ((current * gain) >> 8),
                    dw,
                );
                if next >= 256 {
                    spikes |= 1 << row;
                    state[row] = 0;
                } else {
                    state[row] = next;
                }
            }
            for i in 0..3 {
                input_bits[i] = u8::from(inputs_rng[i] < threshold(input_words[i], ifrac));
                inputs_rng[i] = lfsr(inputs_rng[i]);
            }
            for i in 0..6 {
                weight_bits[i] = u8::from(weights_rng[i] < threshold(weight_words[i], wfrac));
                weights_rng[i] = lfsr(weights_rng[i]);
            }
            for row in 0..2 {
                let count = (0..3)
                    .filter(|i| input_bits[*i] & weight_bits[row * 3 + *i] != 0)
                    .count() as i128;
                currents[row] =
                    i128::from(p.y_min) + (i128::from(p.y_max) - i128::from(p.y_min)) * count / 3;
            }
            let bits = input_bits
                .iter()
                .enumerate()
                .map(|(i, b)| usize::from(*b) << i)
                .sum::<usize>();
            let wbits = weight_bits
                .iter()
                .enumerate()
                .map(|(i, b)| usize::from(*b) << i)
                .sum::<usize>();
            expected.push(vec![
                cycle as i128,
                spikes,
                state[0],
                state[1],
                currents[0],
                currents[1],
                bits as i128,
                wbits as i128,
            ]);
            drive.push_str(&format!("inputs={}; weights={}; leak={}; gain={}; #2; clk=1; #2; $display(\"DENSE %0d %0d %0d %0d %0d %0d %0d %0d\", {cycle}, spikes, $signed(dut.u_dense_0.NEURON[0].v_out_j), $signed(dut.u_dense_0.NEURON[1].v_out_j), $signed(dut.u_dense_0.NEURON[0].I_t_j), $signed(dut.u_dense_0.NEURON[1].I_t_j), dut.u_dense_0.pre_bits, {{dut.u_dense_0.NEURON[1].w_bits,dut.u_dense_0.NEURON[0].w_bits}}); clk=0; #1;\n", packed(&input_words,iw),packed(&weight_words,ww),packed(&[leak as i64],dw),packed(&[gain as i64],dw)));
        }
        let weight_port = if constant {
            String::new()
        } else {
            ".weights(weights),".into()
        };
        let tb=format!("module tb; reg clk=0,rst_n=0; reg [{}:0] inputs; reg [{}:0] weights; reg [{}:0] leak=0,gain=0; wire [1:0] spikes; dense_rtl dut(.clk(clk),.rst_n(rst_n),.inputs(inputs),{weight_port}.leak(leak),.gain(gain),.spikes(spikes)); initial begin #1; clk=1; #1; clk=0; #1; clk=1; #1; clk=0; rst_n=1; {drive} $finish; end endmodule",3*iw-1,6*ww-1,dw-1);
        for (copy, dir) in ["../hdl", "../src/sc_neurocore/hdl/primitives"]
            .into_iter()
            .enumerate()
        {
            let trace = ir_rtl::simulate(
                &format!("dense-{case}-{copy}"),
                &sv,
                &tb,
                &Path::new(env!("CARGO_MANIFEST_DIR")).join(dir),
            );
            let actual: Vec<Vec<i128>> = trace
                .lines()
                .filter_map(|line| line.strip_prefix("DENSE "))
                .map(|line| {
                    line.split_whitespace()
                        .map(|v| v.parse().unwrap())
                        .collect()
                })
                .collect();
            assert_eq!(actual, expected, "Dense case {case}, copy {copy}");
        }
    }
}

#[test]
fn dense_contract_refuses_malformed_shapes_state_geometry_and_seed_zero() {
    let base = DenseParams {
        n_inputs: 1,
        n_neurons: 1,
        ..DenseParams::default()
    };
    for p in [
        DenseParams {
            n_inputs: 0,
            ..base.clone()
        },
        DenseParams {
            n_neurons: 0,
            ..base.clone()
        },
        DenseParams {
            n_inputs: usize::MAX,
            ..base.clone()
        },
        DenseParams {
            data_width: 8,
            ..base.clone()
        },
        DenseParams {
            data_width: 65,
            ..base.clone()
        },
        DenseParams {
            stream_length: 0,
            ..base.clone()
        },
        DenseParams {
            stream_length: u32::MAX as usize + 1,
            ..base.clone()
        },
        DenseParams {
            y_min: -32769,
            ..base.clone()
        },
        DenseParams {
            y_max: 32768,
            ..base.clone()
        },
        DenseParams {
            input_seed_base: 0,
            ..base.clone()
        },
        DenseParams {
            weight_seed_base: 0,
            ..base.clone()
        },
        DenseParams {
            n_inputs: 2,
            input_seed_base: 65529,
            ..base.clone()
        },
        DenseParams {
            n_neurons: 2,
            weight_seed_base: 65523,
            ..base.clone()
        },
        DenseParams {
            n_neurons: 65536,
            ..base.clone()
        },
    ] {
        let g = graph(p, fixed(16, 8), fixed(16, 8), None);
        assert!(verify::verify(&g).is_err());
        assert!(emit_sv::emit(&g).is_err());
    }
    let good = graph(base.clone(), ScType::Rate, ScType::Rate, None);
    assert!(verify::verify(&good).is_ok());
    for ty in [
        ScType::Bool,
        ScType::SInt { width: 16 },
        fixed(0, 0),
        fixed(16, 16),
    ] {
        for input_bad in [true, false] {
            let g = graph(
                base.clone(),
                if input_bad { ty.clone() } else { ScType::Rate },
                if input_bad { ScType::Rate } else { ty.clone() },
                None,
            );
            assert!(verify::verify(&g).is_err());
            assert!(emit_sv::emit(&g).is_err());
        }
    }
    for (id, ty) in [
        (0, vector(ScType::Rate, 2)),
        (1, vector(ScType::Rate, 2)),
        (2, fixed(16, 7)),
        (3, ScType::UInt { width: 16 }),
    ] {
        let mut g = good.clone();
        if let ScOp::Input { ty: old, .. } = &mut g.ops[id] {
            *old = ty;
        }
        assert!(verify::verify(&g).is_err());
        assert!(emit_sv::emit(&g).is_err());
    }
}

#[test]
fn dense_accepts_raw_signed_q8_words_and_refuses_untyped_computed_operands() {
    for ty in [ScType::Rate, ScType::SInt { width: 16 }] {
        let p = DenseParams {
            n_inputs: 1,
            n_neurons: 1,
            ..DenseParams::default()
        };
        let mut g = graph(p, ScType::Rate, ScType::Rate, None);
        for index in [2, 3] {
            if let ScOp::Input { ty: word, .. } = &mut g.ops[index] {
                *word = ty.clone();
            }
        }
        verify::verify(&g).unwrap();
        emit_sv::emit(&g).unwrap();
    }
    let mut b = ScGraphBuilder::new("computed_dense");
    let inputs = b.input("inputs", vector(ScType::Rate, 1));
    let weights = b.input("weights", vector(ScType::Rate, 1));
    let leak = b.input("leak", fixed(16, 8));
    let gain = b.input("gain", fixed(16, 8));
    let computed = b.scale(inputs, 1.0);
    b.dense_forward(
        computed,
        weights,
        leak,
        gain,
        DenseParams {
            n_inputs: 1,
            n_neurons: 1,
            ..DenseParams::default()
        },
    );
    let graph = b.build();
    assert!(verify::verify(&graph).is_err());
    assert!(emit_sv::emit(&graph).is_err());
}
