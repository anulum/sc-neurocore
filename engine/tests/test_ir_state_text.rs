// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Register text preservation

//! Lossless state/store text and explicit refusal of malformed attributes.
use sc_neurocore_engine::ir::{builder::ScGraphBuilder, graph::*, parser, printer, verify};

#[test]
fn sparse_ids_and_every_store_attribute_round_trip() {
    let text = "sc.graph @sparse {\n%7 = sc.constant -3 : fixed<16,8>\n%19 = sc.input \"advance\" : bool\n%40 = sc.state %7 : fixed<16,8>\n%80 = sc.store %40, %40, enable=%19\n%81 = sc.store %42, %40, enable=none\n%42 = sc.state %7 : fixed<16,8>\n}";
    // Parsing preserves sparse identifiers; verification still refuses forward uses.
    let graph = parser::parse(text).unwrap();
    assert!(verify::verify(&graph).is_err());
    assert_eq!(parser::parse(&printer::print(&graph)).unwrap(), graph);
    for enable in [None, Some(ValueId(1))] {
        let mut b = ScGraphBuilder::new("roundtrip");
        let initial = b.constant(
            ScConst::F64(-0.125),
            ScType::FixedPoint { width: 16, frac: 8 },
        );
        b.input("advance", ScType::Bool);
        let state = b.state(initial, ScType::FixedPoint { width: 16, frac: 8 });
        b.store(state, state, enable);
        b.output("current", state);
        let graph = b.build();
        verify::verify(&graph).unwrap();
        assert_eq!(parser::parse(&printer::print(&graph)).unwrap(), graph);
    }
}

#[test]
fn malformed_reset_and_store_syntax_is_not_silently_defaulted() {
    for bad in [
        "%1= sc.state %0 : u8",
        "%invalid = sc.state %0 : u8",
        "%1 = sc.state %missing : u8",
        "%2 = sc.store %missing, %0, enable=none",
        "%2 = sc.store %1, %missing, enable=none",
        "%1 = sc.state %0",
        "%1 = sc.state %0 : u8, stale=1",
        "%1 = sc.state %0, %0 : u8",
        "%1 = sc.state %0 : unknown",
        "%2 = sc.store %1, %0",
        "%2 = sc.store %1, %0, enable=none, stale=1",
        "%2 = sc.store %1, %0, stale=none",
        "%2 = sc.store %1, %0, enable=bogus",
        "%4294967295 = sc.state %0 : u8",
        "%4294967295 = sc.store %1, %0, enable=none",
    ] {
        let text = format!("sc.graph @syntax {{\n%0 = sc.constant 1 : u8\n{bad}\n}}");
        let error = parser::parse(&text).unwrap_err();
        assert_eq!(error.line, 3, "{bad}");
    }
}

#[test]
fn raw_integer_and_float_reset_constructors_round_trip_without_reinterpretation() {
    use sc_neurocore_engine::ir::{emit_mlir, emit_sv};
    for ty in [
        ScType::Bool,
        ScType::Rate,
        ScType::UInt { width: 16 },
        ScType::SInt { width: 16 },
        ScType::FixedPoint { width: 16, frac: 8 },
    ] {
        for value in [ScConst::I64(1), ScConst::U64(1), ScConst::F64(0.5)] {
            let mut b = ScGraphBuilder::new("reset_words");
            let initial = b.constant(value, ty.clone());
            let state = b.state(initial, ty.clone());
            b.store(state, state, None);
            b.output("current", state);
            let graph = b.build();
            let restored = parser::parse(&printer::print(&graph)).unwrap();
            assert_eq!(restored, graph);
            assert_eq!(
                emit_sv::emit(&restored).unwrap(),
                emit_sv::emit(&graph).unwrap()
            );
            assert_eq!(
                emit_mlir::emit(&restored).unwrap(),
                emit_mlir::emit(&graph).unwrap()
            );
        }
    }
    let legacy =
        parser::parse("sc.graph @legacy {\n%0 = sc.constant 1 : rate\nsc.output \"current\" %0\n}")
            .unwrap();
    assert_eq!(
        legacy.ops[0],
        ScOp::Constant {
            id: ValueId(0),
            value: ScConst::F64(1.0),
            ty: ScType::Rate
        }
    );
    assert!(emit_sv::emit(&legacy).unwrap().contains("16'sd256"));
    for bad in [
        "i64(9223372036854775808)",
        "u64(-1)",
        "u64(18446744073709551616)",
        "i64(x)",
        "u64(1) stale=1",
    ] {
        assert!(parser::parse(&format!(
            "sc.graph @bad {{\n%0 = sc.constant {bad} : rate\n}}"
        ))
        .is_err());
    }
}
