// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Declared type geometry through public construction and text

//! Unused declarations and parser recursion have the same hardware admission rules.

#[path = "common/ir_stream.rs"]
mod ir_stream;
use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder, emit_mlir, emit_sv, graph::*, parser, verify, word::*,
};

fn reject(graph: &ScGraph, message: &str) {
    let errors = verify::verify(graph)
        .unwrap_err()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(errors.contains(message), "{errors}");
    for error in [
        emit_sv::emit(graph).unwrap_err(),
        emit_mlir::emit(graph).unwrap_err(),
    ] {
        assert!(error.contains(message), "{error}");
    }
}

/// A type is checked even when no arithmetic or output consumes its value.
#[test]
fn unused_invalid_inputs_refuse_before_any_hardware_dimensions() {
    for ty in [
        ScType::UInt { width: 0 },
        ScType::SInt { width: 0 },
        ScType::UInt { width: u32::MAX },
        ScType::FixedPoint { width: 0, frac: 0 },
        ScType::FixedPoint {
            width: 16,
            frac: 16,
        },
        ScType::Bitstream { length: 0 },
        ScType::Vec {
            element: Box::new(ScType::Bool),
            count: 0,
        },
        ScType::Vec {
            element: Box::new(ScType::UInt { width: 64 }),
            count: usize::MAX,
        },
        ScType::Vec {
            element: Box::new(ScType::Bool),
            count: i32::MAX as usize + 1,
        },
    ] {
        let mut b = ScGraphBuilder::new("unused_invalid_input");
        b.input("unused", ty);
        let x = b.input("x", ScType::SInt { width: 8 });
        let y = b.word(WordOp::Binary {
            kind: WordBinaryOp::Add,
            lhs: x,
            rhs: x,
            ty: ScType::SInt { width: 8 },
        });
        b.output("value", y);
        reject(&b.build(), "declared");
    }
    let mut b = ScGraphBuilder::new("unused_invalid_constant");
    b.constant(ScConst::I64(0), ScType::UInt { width: 0 });
    reject(&b.build(), "declared");
    let mut b = ScGraphBuilder::new("invalid_word_result");
    let x = b.input("x", ScType::SInt { width: 8 });
    b.word(WordOp::Resize {
        input: x,
        ty: ScType::UInt { width: 0 },
    });
    reject(&b.build(), "declared");
}

/// Valid nested vectors retain physical bits; invalid nesting stops before type cloning or emission.
#[test]
fn vector_depth_boundary_has_real_hardware_semantics() {
    let mut ty = ScType::UInt { width: 8 };
    for _ in 0..64 {
        ty = ScType::Vec {
            element: Box::new(ty),
            count: 1,
        };
    }
    let mut b = ScGraphBuilder::new("nested_vector");
    let bits = b.input("bits", ty.clone());
    b.output("value", bits);
    let bench="module tb; reg clk=0,rst_n=1; reg [7:0] bits=8'd174; wire [7:0] value; nested_vector dut(.*); initial begin #1; $display(\"OBS %h\",value); $finish; end endmodule";
    ir_stream::run(
        "nested-vector-depth64",
        &b.build(),
        bench,
        &["OBS ae".into()],
    );
    let mut b = ScGraphBuilder::new("invalid_depth");
    b.input(
        "unused",
        ScType::Vec {
            element: Box::new(ty),
            count: 1,
        },
    );
    reject(&b.build(), "nesting exceeds 64");
}

/// Machine text is refused before recursively constructing an excessive boxed type tree.
#[test]
fn type_text_depth_is_bounded_before_recursive_parsing() {
    for depth in [65, 1000] {
        let ty = "vec<".repeat(depth) + "u8" + &",1>".repeat(depth);
        let text = format!("sc.graph @too_deep {{\n  %0 = sc.input \"x\" : {ty}\n}}\n");
        let error = parser::parse(&text).unwrap_err();
        assert_eq!(error.line, 2);
        assert!(error.message.contains("nesting exceeds 64"));
    }
}
