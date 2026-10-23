// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Scalar word type and boundary refusal

//! Public verification and both emitters reject malformed circuits before RTL emission.

use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder, emit_mlir, emit_sv, graph::*, verify, word::*,
};

fn reject(graph: ScGraph, diagnostic: &str) {
    let errors = verify::verify(&graph)
        .unwrap_err()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(errors.contains(diagnostic), "{errors}");
    for error in [
        emit_sv::emit(&graph).unwrap_err(),
        emit_mlir::emit(&graph).unwrap_err(),
    ] {
        assert!(error.contains(diagnostic), "{error}");
    }
}

/// Invalid scalar geometry, signedness, windows and branch types are never inferred away.
#[test]
fn scalar_word_operations_reject_incompatible_types() {
    for (lhs_ty, rhs_ty, result_ty) in [
        (
            ScType::SInt { width: 8 },
            ScType::UInt { width: 8 },
            ScType::SInt { width: 8 },
        ),
        (
            ScType::UInt { width: 8 },
            ScType::UInt { width: 16 },
            ScType::UInt { width: 8 },
        ),
        (
            ScType::FixedPoint { width: 16, frac: 8 },
            ScType::FixedPoint { width: 16, frac: 8 },
            ScType::FixedPoint { width: 16, frac: 8 },
        ),
        (
            ScType::Bitstream { length: 8 },
            ScType::Bitstream { length: 8 },
            ScType::Bitstream { length: 8 },
        ),
        (
            ScType::UInt { width: 0 },
            ScType::UInt { width: 0 },
            ScType::UInt { width: 0 },
        ),
    ] {
        let diagnostic = if matches!(result_ty, ScType::FixedPoint { .. }) {
            "raw arithmetic"
        } else {
            "word"
        };
        let mut b = ScGraphBuilder::new("bad_word");
        let lhs = b.input("lhs", lhs_ty);
        let rhs = b.input("rhs", rhs_ty);
        b.word(WordOp::Binary {
            kind: WordBinaryOp::Add,
            lhs,
            rhs,
            ty: result_ty,
        });
        reject(b.build(), diagnostic);
    }
    let mut b = ScGraphBuilder::new("bad_shift");
    let x = b.input("x", ScType::UInt { width: 8 });
    b.word(WordOp::Binary {
        kind: WordBinaryOp::AShr,
        lhs: x,
        rhs: x,
        ty: ScType::UInt { width: 8 },
    });
    reject(b.build(), "signed integer");
    let mut b = ScGraphBuilder::new("bad_select");
    let x = b.input("x", ScType::UInt { width: 8 });
    b.word(WordOp::Select {
        condition: x,
        when_true: x,
        when_false: x,
        ty: ScType::UInt { width: 8 },
    });
    reject(b.build(), "requires bool");
    let mut b = ScGraphBuilder::new("bad_cast");
    let x = b.input("x", ScType::UInt { width: 8 });
    b.word(WordOp::Bitcast {
        input: x,
        ty: ScType::SInt { width: 16 },
    });
    reject(b.build(), "preserve physical");
}

/// Invalid rounding exponents and saturation limits cannot become Verilog dimensions/literals.
#[test]
fn scalar_word_operations_reject_invalid_numeric_attributes() {
    for op in [
        WordOp::Round {
            input: ValueId(0),
            shift: 8,
            mode: WordRounding::Floor,
            ty: ScType::SInt { width: 8 },
        },
        WordOp::Clamp {
            input: ValueId(0),
            min: -129,
            max: 0,
            ty: ScType::SInt { width: 8 },
        },
        WordOp::Clamp {
            input: ValueId(0),
            min: 0,
            max: 128,
            ty: ScType::SInt { width: 8 },
        },
        WordOp::Clamp {
            input: ValueId(0),
            min: 1,
            max: 0,
            ty: ScType::SInt { width: 8 },
        },
    ] {
        let mut b = ScGraphBuilder::new("invalid_numeric");
        b.input("x", ScType::SInt { width: 8 });
        let message = if matches!(op, WordOp::Round { .. }) {
            "shift"
        } else {
            "bounds"
        };
        b.word(op);
        reject(b.build(), message);
    }
}

/// Deep malformed recurrence cannot overflow the host stack during admission.
#[test]
fn large_scalar_graphs_are_verified_without_recursive_traversal() {
    let mut graph = ScGraph::new("large_words");
    for i in 0..20000 {
        graph.push(ScOp::Word {
            id: ValueId(i),
            op: WordOp::Resize {
                input: ValueId(i + 1),
                ty: ScType::SInt { width: 32 },
            },
        });
    }
    graph.push(ScOp::Input {
        id: ValueId(20000),
        name: "x".into(),
        ty: ScType::SInt { width: 32 },
    });
    reject(graph, "not defined before use");
}

/// Every cast, selector and numerical guard refuses malformed public graphs.
#[test]
fn scalar_casts_and_comparisons_reject_non_scalar_producers() {
    for input_type in [
        ScType::Bitstream { length: 4 },
        ScType::Vec {
            element: Box::new(ScType::UInt { width: 8 }),
            count: 2,
        },
        ScType::FixedPoint {
            width: 16,
            frac: 16,
        },
    ] {
        let mut b = ScGraphBuilder::new("bad_bitcast");
        let x = b.input("x", input_type);
        b.word(WordOp::Bitcast {
            input: x,
            ty: ScType::UInt { width: 16 },
        });
        reject(b.build(), "scalar type");
    }
    for op in [
        WordOp::Resize {
            input: ValueId(0),
            ty: ScType::UInt { width: 16 },
        },
        WordOp::Compare {
            predicate: WordCompareOp::Lt,
            lhs: ValueId(0),
            rhs: ValueId(0),
        },
    ] {
        let mut b = ScGraphBuilder::new("bad_raw_cast");
        b.input("x", ScType::Rate);
        b.word(op);
        reject(b.build(), "raw arithmetic");
    }
    let mut b = ScGraphBuilder::new("undefined_word");
    b.word(WordOp::Unary {
        kind: WordUnaryOp::Neg,
        input: ValueId(99),
        ty: ScType::SInt { width: 16 },
    });
    reject(b.build(), "no scalar word producer");
    let mut b = ScGraphBuilder::new("unsigned_bounds");
    let x = b.input("x", ScType::UInt { width: 8 });
    b.word(WordOp::Clamp {
        input: x,
        min: -1,
        max: 1,
        ty: ScType::UInt { width: 8 },
    });
    reject(b.build(), "bounds");
    let mut b = ScGraphBuilder::new("unsigned_bounds_high");
    let x = b.input("x", ScType::UInt { width: 8 });
    b.word(WordOp::Clamp {
        input: x,
        min: 0,
        max: 256,
        ty: ScType::UInt { width: 8 },
    });
    reject(b.build(), "bounds");
}

/// Clock duplication, generated-wire capture, source injection and duplicate ports fail identically.
#[test]
fn hardware_port_names_cannot_capture_scalar_temporaries() {
    for name in [
        "clk",
        "rst_n",
        "sc_clock",
        "sc_one",
        "v0",
        "v2_divide",
        "c0",
        "encode_0_probability",
        "dense_0_inputs",
        "dense_0_weights_product_with_unused_bits_0",
        "u_enc_0",
        "wire",
        "logic",
        "1bad",
        "x\"\nendmodule",
        "",
    ] {
        let mut b = ScGraphBuilder::new("port_names");
        let x = b.input(name, ScType::SInt { width: 8 });
        b.output("result", x);
        reject(b.build(), "identifier");
    }
    let mut b = ScGraphBuilder::new("duplicate_ports");
    let x = b.input("same", ScType::SInt { width: 8 });
    b.output("same", x);
    reject(b.build(), "duplicate HDL port");
    for name in ["module", "a-b", ""] {
        reject(ScGraphBuilder::new(name).build(), "module identifier");
    }
}

/// Type errors in either select branch and each conversion family remain attributable.
#[test]
fn every_scalar_family_checks_its_operand_geometry() {
    let u8t = ScType::UInt { width: 8 };
    let u16t = ScType::UInt { width: 16 };
    for op in [
        WordOp::Unary {
            kind: WordUnaryOp::Not,
            input: ValueId(0),
            ty: u16t.clone(),
        },
        WordOp::Unary {
            kind: WordUnaryOp::Neg,
            input: ValueId(0),
            ty: ScType::Rate,
        },
        WordOp::Round {
            input: ValueId(0),
            shift: 1,
            mode: WordRounding::Floor,
            ty: u16t.clone(),
        },
        WordOp::Compare {
            predicate: WordCompareOp::Eq,
            lhs: ValueId(0),
            rhs: ValueId(1),
        },
        WordOp::Resize {
            input: ValueId(0),
            ty: ScType::Rate,
        },
        WordOp::Clamp {
            input: ValueId(0),
            min: 0,
            max: 1,
            ty: u16t.clone(),
        },
        WordOp::Clamp {
            input: ValueId(0),
            min: 0,
            max: 1,
            ty: ScType::Rate,
        },
        WordOp::Bitcast {
            input: ValueId(0),
            ty: ScType::FixedPoint { width: 8, frac: 8 },
        },
        WordOp::Select {
            condition: ValueId(2),
            when_true: ValueId(1),
            when_false: ValueId(0),
            ty: u8t.clone(),
        },
        WordOp::Select {
            condition: ValueId(2),
            when_true: ValueId(0),
            when_false: ValueId(1),
            ty: u8t.clone(),
        },
    ] {
        let mut b = ScGraphBuilder::new("family_geometry");
        b.input("x", u8t.clone());
        b.input("y", u16t.clone());
        b.input("condition", ScType::Bool);
        b.word(op);
        reject(b.build(), "requires");
    }
}

/// Dense's probability conversion bus must not redeclare an external input port.
#[test]
fn dense_probability_bus_cannot_capture_a_public_port_name() {
    let mut b = ScGraphBuilder::new("dense_collision");
    let ty = ScType::Vec {
        element: Box::new(ScType::Rate),
        count: 1,
    };
    let x = b.input("dense_0_inputs", ty.clone());
    let w = b.constant(ScConst::I64Vec(vec![128]), ty);
    let leak = b.constant(ScConst::I64(16), ScType::SInt { width: 16 });
    let gain = b.constant(ScConst::I64(256), ScType::SInt { width: 16 });
    let y = b.dense_forward(
        x,
        w,
        leak,
        gain,
        DenseParams {
            n_inputs: 1,
            n_neurons: 1,
            ..Default::default()
        },
    );
    b.output("spikes", y);
    reject(b.build(), "reserved generated namespace");
}
