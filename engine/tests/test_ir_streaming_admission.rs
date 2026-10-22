// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Bitstream admission and serialization boundaries

//! Reject malformed public graphs before bus widths, counters or parser IDs overflow.

use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder, emit_mlir, emit_sv, graph::*, parser, verify,
};

/// Require all public admission surfaces to reject malformed circuit geometry.
fn reject(g: &ScGraph) {
    assert!(verify::verify(g).is_err());
    assert!(emit_sv::emit(g).is_err());
    assert!(emit_mlir::emit(g).is_err());
}

/// Validate indexing, nested vector products and total binary observations independently.
#[test]
fn packed_and_temporal_shape_overflows_are_errors() {
    let mut shapes = vec![
        ScType::UInt { width: u32::MAX },
        ScType::Vec {
            element: Box::new(ScType::UInt { width: 64 }),
            count: usize::MAX,
        },
        ScType::Vec {
            element: Box::new(ScType::Bool),
            count: i32::MAX as usize + 1,
        },
        ScType::Vec {
            element: Box::new(ScType::Rate),
            count: 2,
        },
    ];
    if usize::BITS == 64 {
        shapes.push(ScType::Vec {
            element: Box::new(ScType::Bitstream { length: usize::MAX }),
            count: 2,
        });
    }
    for ty in shapes {
        let mut b = ScGraphBuilder::new("overflow");
        let input = b.input("bits", ty);
        b.decode(input);
        reject(&b.build());
    }
}

/// Integer division rejects invalid buses and temporal or probability-valued operands.
#[test]
fn division_requires_a_finite_packed_integer_word() {
    for ty in [
        ScType::UInt { width: 0 },
        ScType::UInt { width: u32::MAX },
        ScType::Vec {
            element: Box::new(ScType::UInt { width: 64 }),
            count: usize::MAX,
        },
        ScType::Bitstream { length: 3 },
        ScType::Rate,
    ] {
        let mut b = ScGraphBuilder::new("invalid_division");
        let input = b.input("dividend", ty);
        let quotient = b.div_const(input, 3);
        b.output("quotient", quotient);
        reject(&b.build());
    }
}

/// Unknown producers and cycles return diagnostics without recursive primitive type lookup.
#[test]
fn undefined_duplicate_and_cyclic_stream_graphs_fail_admission() {
    let mut g = ScGraph::new("undefined");
    g.push(ScOp::Decode {
        id: ValueId(0),
        input: ValueId(99),
    });
    reject(&g);
    let mut g = ScGraph::new("cycle");
    g.push(ScOp::BitwiseAnd {
        id: ValueId(0),
        lhs: ValueId(1),
        rhs: ValueId(1),
    });
    g.push(ScOp::BitwiseXor {
        id: ValueId(1),
        lhs: ValueId(0),
        rhs: ValueId(0),
    });
    reject(&g);
    let mut g = ScGraph::new("duplicate");
    g.push(ScOp::Input {
        id: ValueId(0),
        name: "bits".into(),
        ty: ScType::Bool,
    });
    g.push(ScOp::Popcount {
        id: ValueId(0),
        input: ValueId(0),
    });
    reject(&g);
}

/// Equal physical width does not erase the difference between integer, bool and stream types.
#[test]
fn bitwise_operations_require_equal_shapes_types_and_epochs() {
    for (lhs, rhs) in [
        (ScType::Bool, ScType::UInt { width: 1 }),
        (ScType::UInt { width: 8 }, ScType::SInt { width: 8 }),
        (
            ScType::Bitstream { length: 3 },
            ScType::Bitstream { length: 4 },
        ),
        (
            ScType::Bool,
            ScType::Vec {
                element: Box::new(ScType::Bool),
                count: 0,
            },
        ),
    ] {
        for xor in [false, true] {
            let mut b = ScGraphBuilder::new("shape_mismatch");
            let a = b.input("a", lhs.clone());
            let z = b.input("b", rhs.clone());
            if xor {
                b.bitwise_xor(a, z);
            } else {
                b.bitwise_and(a, z);
            }
            reject(&b.build());
        }
    }
}

/// Seeds, windows and declared probability types are admitted before encoder lowering.
#[test]
fn encoders_require_positive_windows_and_nonabsorbing_seeds() {
    for (length, seed) in [(0, 0xACE1), (3, 0)] {
        let mut b = ScGraphBuilder::new("bad_encoder");
        let p = b.input("probability", ScType::Rate);
        b.encode(p, length, seed);
        reject(&b.build());
    }
    let mut b = ScGraphBuilder::new("untyped_probability");
    let p = b.input("probability", ScType::Rate);
    let p = b.scale(p, 1.0);
    b.encode(p, 3, 0xACE1);
    reject(&b.build());
}

/// Decoder text has an explicit Rate result and an SSA ID that can allocate its successor.
#[test]
fn decoder_text_rejects_bad_result_types_operands_and_overflowing_ids() {
    for line in [
        "%1 = sc.decode %0",
        "%1 = sc.decode %0 : u64",
        "%1 = sc.decode %0, %0 : rate",
        "%-1 = sc.decode %0 : rate",
        "%4294967295 = sc.decode %0 : rate",
        "%1 = sc.decode broken : rate",
    ] {
        let text =
            format!("sc.graph @bad_decode {{\n  %0 = sc.input \"bits\" : bool\n  {line}\n}}\n");
        assert!(parser::parse(&text).is_err(), "{line}");
    }
}
