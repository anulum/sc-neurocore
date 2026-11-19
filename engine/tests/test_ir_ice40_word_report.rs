// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — iCE40 calibrated graph admission and literal parity

//! Exercise calibrated and routed-only graph forms through the iCE40 report.

use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder,
    emit_sv::emit_systemverilog_for_ice40_up5k as emit,
    graph::{DenseParams, ScConst, ScGraph, ScOp, ScType, ValueId},
    ice40_estimate::{estimate, Ice40Estimate, OUTSIDE_CALIBRATED_FORM},
    word::{WordBinaryOp, WordOp},
};

fn binary(
    builder: &mut ScGraphBuilder,
    kind: WordBinaryOp,
    lhs: ValueId,
    rhs: ValueId,
    ty: ScType,
) -> ValueId {
    builder.word(WordOp::Binary { kind, lhs, rhs, ty })
}

fn uncalibrated(graph: &ScGraph) {
    assert!(matches!(estimate(graph), Ice40Estimate::Unavailable { .. }));
}

#[test]
fn equivalent_integer_literals_and_input_ports_get_their_calibration() {
    let ty = ScType::UInt { width: 11 };
    let literal_graph = |literal| {
        let mut b = ScGraphBuilder::new("literal");
        let a = b.input("a", ty.clone());
        let c = b.constant(literal, ty.clone());
        let result = binary(&mut b, WordBinaryOp::Add, a, c, ty.clone());
        b.output("result", result);
        b.build()
    };
    let signed_storage = emit(&literal_graph(ScConst::I64(13)), 12).unwrap();
    let unsigned_storage = emit(&literal_graph(ScConst::U64(13)), 12).unwrap();
    assert_eq!(signed_storage, unsigned_storage);
    assert!(matches!(
        unsigned_storage.1.estimate,
        Ice40Estimate::Calibrated {
            block: "word_add",
            ..
        }
    ));

    let ty = ScType::SInt { width: 32 };
    let mut b = ScGraphBuilder::new("signed_subtract");
    let constant = b.constant(ScConst::I64(i64::from(i32::MIN)), ty.clone());
    let input = b.input("input_word", ty.clone());
    let result = binary(&mut b, WordBinaryOp::Sub, constant, input, ty);
    b.output("result", result);
    let (_, report) = emit(&b.build(), 12).unwrap();
    assert!(matches!(
        report.estimate,
        Ice40Estimate::Calibrated {
            block: "word_sub",
            ..
        }
    ));

    let ty = ScType::UInt { width: 8 };
    let mut b = ScGraphBuilder::new("aliased_add");
    let input = b.input("input_word", ty.clone());
    let result = binary(&mut b, WordBinaryOp::Add, input, input, ty);
    b.output("result", result);
    let (_, report) = emit(&b.build(), 12).unwrap();
    assert_eq!(report.fits_logic_cells(), Some(true));
}

#[test]
fn valid_circuits_outside_the_calibrated_word_shape_still_emit() {
    let ty = ScType::UInt { width: 8 };
    let mut b = ScGraphBuilder::new("original_output");
    let a = b.input("a", ty.clone());
    let c = b.input("c", ty.clone());
    binary(&mut b, WordBinaryOp::Add, a, c, ty.clone());
    b.output("original", a);
    let (_, report) = emit(&b.build(), 12).unwrap();
    assert_eq!(report.fits_logic_cells(), None);

    let mut b = ScGraphBuilder::new("unused_input");
    let a = b.input("a", ty.clone());
    b.input("unused", ty.clone());
    let result = binary(&mut b, WordBinaryOp::Add, a, a, ty.clone());
    b.output("result", result);
    let (_, report) = emit(&b.build(), 12).unwrap();
    assert_eq!(report.fits_logic_cells(), None);

    for (kind, ty) in [
        (WordBinaryOp::Add, ScType::UInt { width: 33 }),
        (WordBinaryOp::Mul, ty.clone()),
    ] {
        let mut b = ScGraphBuilder::new("other_geometry_or_operator");
        let a = b.input("a", ty.clone());
        let c = b.input("c", ty.clone());
        let result = binary(&mut b, kind, a, c, ty);
        b.output("result", result);
        let (_, report) = emit(&b.build(), 12).unwrap();
        assert_eq!(report.fits_logic_cells(), None);
    }

    let mut b = ScGraphBuilder::new("folded_constants");
    let a = b.constant(ScConst::I64(1), ty.clone());
    let c = b.constant(ScConst::U64(3), ty.clone());
    let result = binary(&mut b, WordBinaryOp::Sub, a, c, ty);
    b.output("result", result);
    let (_, report) = emit(&b.build(), 12).unwrap();
    assert_eq!(report.fits_logic_cells(), None);
    let mut b = ScGraphBuilder::new("xor_stream");
    let lhs = b.input("lhs", ScType::Bitstream { length: 64 });
    let rhs = b.input("rhs", ScType::Bitstream { length: 64 });
    let result = b.bitwise_xor(lhs, rhs);
    b.output("result", result);
    let (_, report) = emit(&b.build(), 12).unwrap();
    assert_eq!(
        report.estimate,
        Ice40Estimate::Unavailable {
            reason: "the graph holds an operation the calibration does not cover".into(),
        }
    );
}

#[test]
fn malformed_words_and_out_of_range_literals_never_get_an_estimate() {
    let word = ScType::UInt { width: 8 };
    let wrong = ScType::UInt { width: 16 };
    let mut b = ScGraphBuilder::new("typed_ports");
    let lhs = b.input("lhs", word.clone());
    let rhs = b.input("rhs", wrong.clone());
    let result = binary(&mut b, WordBinaryOp::Add, lhs, rhs, word.clone());
    b.output("result", result);
    let invalid_ports = b.build();
    uncalibrated(&invalid_ports);
    assert!(emit(&invalid_ports, 12).is_err());

    for (ty, literal) in [
        (word.clone(), ScConst::U64(256)),
        (wrong, ScConst::U64(13)),
        (word.clone(), ScConst::F64(13.0)),
    ] {
        let mut b = ScGraphBuilder::new("literal_check");
        let lhs = b.input("lhs", word.clone());
        let rhs = b.constant(literal, ty);
        let result = binary(&mut b, WordBinaryOp::Add, lhs, rhs, word.clone());
        b.output("result", result);
        uncalibrated(&b.build());
    }

    let ty = ScType::FixedPoint { width: 16, frac: 8 };
    let mut b = ScGraphBuilder::new("wrong_arithmetic_type");
    let a = b.input("a", ty.clone());
    let result = binary(&mut b, WordBinaryOp::Add, a, a, ty);
    b.output("result", result);
    uncalibrated(&b.build());

    let mut b = ScGraphBuilder::new("undefined_word_operand");
    let a = b.input("a", word.clone());
    b.input("unrelated", word.clone());
    let result = binary(&mut b, WordBinaryOp::Add, a, ValueId(14), word);
    b.output("result", result);
    let graph = b.build();
    uncalibrated(&graph);
    assert!(emit(&graph, 12).is_err());
}

#[test]
fn incomplete_stream_chains_are_never_a_calibrated_synapse() {
    for operation in [
        "and_only",
        "decode_only",
        "two_encoders",
        "without_decoder",
        "mixed_lengths",
    ] {
        let mut b = ScGraphBuilder::new(operation);
        let result = match operation {
            "and_only" | "decode_only" => {
                let lhs = b.input("lhs", ScType::Bitstream { length: 64 });
                if operation == "and_only" {
                    let rhs = b.input("rhs", ScType::Bitstream { length: 64 });
                    b.bitwise_and(lhs, rhs)
                } else {
                    b.decode(lhs)
                }
            }
            _ => {
                let lhs = b.input("lhs", ScType::Rate);
                let rhs = b.input("rhs", ScType::Rate);
                let x = b.encode(lhs, 64, 0xACE1);
                let y = b.encode(
                    rhs,
                    if operation == "mixed_lengths" {
                        128
                    } else {
                        64
                    },
                    0xBEEF,
                );
                match operation {
                    "two_encoders" => x,
                    "without_decoder" => b.bitwise_and(x, y),
                    _ => {
                        let both = b.bitwise_and(x, x);
                        b.decode(both)
                    }
                }
            }
        };
        b.output("result", result);
        uncalibrated(&b.build());
    }

    let mut b = ScGraphBuilder::new("wrong_probability_port");
    let lhs = b.input("lhs", ScType::FixedPoint { width: 16, frac: 8 });
    let rhs = b.input("rhs", ScType::Rate);
    let lhs = b.encode(lhs, 64, 0xACE1);
    let rhs = b.encode(rhs, 64, 0xBEEF);
    let both = b.bitwise_and(lhs, rhs);
    let decoded = b.decode(both);
    b.output("result", decoded);
    uncalibrated(&b.build());
}

#[test]
fn dense_report_refuses_port_and_constant_forms_outside_its_calibration() {
    let vec_rate = |count| ScType::Vec {
        element: Box::new(ScType::Rate),
        count,
    };
    let fixed = |frac| ScType::FixedPoint { width: 16, frac };
    let mut b = ScGraphBuilder::new("dense_calibration");
    let inputs = b.input("inputs", vec_rate(2));
    let weights = b.constant(ScConst::F64Vec(vec![0.5; 4]), vec_rate(4));
    let leak = b.constant(ScConst::I64(16), fixed(8));
    let gain = b.constant(ScConst::I64(256), fixed(8));
    let out = b.dense_forward(
        inputs,
        weights,
        leak,
        gain,
        DenseParams {
            n_inputs: 2,
            n_neurons: 2,
            data_width: 16,
            stream_length: 7,
            input_seed_base: 4660,
            weight_seed_base: 4661,
            y_min: 0,
            y_max: 300,
        },
    );
    b.output("result", out);
    let baseline = b.build();
    assert!(matches!(
        estimate(&baseline),
        Ice40Estimate::Calibrated { .. }
    ));

    for (index, ty, value) in [
        (0, vec_rate(3), None),
        (1, vec_rate(5), None),
        (2, fixed(7), None),
        (3, fixed(8), Some(ScConst::I64(255))),
    ] {
        let mut alternate = baseline.clone();
        match &mut alternate.ops[index] {
            ScOp::Input { ty: actual, .. } => *actual = ty,
            ScOp::Constant {
                ty: actual,
                value: existing,
                ..
            } => {
                *actual = ty;
                if let Some(value) = value {
                    *existing = value;
                }
            }
            other => panic!("unexpected dense operand {other:?}"),
        }
        assert_eq!(
            estimate(&alternate),
            Ice40Estimate::Unavailable {
                reason: OUTSIDE_CALIBRATED_FORM.into()
            }
        );
    }
}
