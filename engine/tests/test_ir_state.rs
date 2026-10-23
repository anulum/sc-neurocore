// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Native register admission

//! Public native register admission, atomic drivers and terminator invariants.
use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder, emit_mlir, emit_sv, graph::*, parser, verify,
};

fn refuse(text: &str, message: &str) {
    let graph = parser::parse(text).unwrap();
    let errors = verify::verify(&graph)
        .unwrap_err()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(errors.contains(message), "{errors}");
    assert!(emit_sv::emit(&graph).is_err());
    assert!(emit_mlir::emit(&graph).is_err());
}

#[test]
fn public_register_builder_preserves_all_scalar_formats() {
    for ty in [
        ScType::Bool,
        ScType::Rate,
        ScType::UInt { width: 1 },
        ScType::UInt { width: 64 },
        ScType::SInt { width: 64 },
        ScType::FixedPoint { width: 16, frac: 7 },
    ] {
        let mut b = ScGraphBuilder::new("formats");
        let initial = b.constant(ScConst::U64(1), ty.clone());
        let state = b.state(initial, ty.clone());
        b.store(state, state, None);
        b.output("current", state);
        let graph = b.build();
        verify::verify(&graph).unwrap();
        let text = sc_neurocore_engine::ir::printer::print(&graph);
        assert_eq!(parser::parse(&text).unwrap(), graph);
        assert!(emit_sv::emit(&graph)
            .unwrap()
            .contains("always @(posedge clk or negedge rst_n)"));
        assert!(emit_mlir::emit(&graph).unwrap().contains("seq.firreg"));
    }
}

#[test]
fn register_requires_scalar_constant_reset_with_exact_type() {
    for (initial, ty, diagnostic) in [
        (
            "%0 = sc.input \"input_word\" : u8",
            "u8",
            "earlier constant",
        ),
        ("%0 = sc.constant 1 : u8", "i8", "exact state type"),
        (
            "%0 = sc.constant 1 : fixed<8,2>",
            "fixed<8,3>",
            "exact state type",
        ),
        (
            "%0 = sc.constant [1] : vec<u8,1>",
            "vec<u8,1>",
            "numeric scalar",
        ),
        ("%0 = sc.constant 1 : u65", "u65", "width must be 1..64"),
        ("%0 = sc.constant 1 : u8", "u0", "physical word width"),
    ] {
        refuse(&format!("sc.graph @bad {{\n{initial}\n%1 = sc.state %0 : {ty}\n%2 = sc.store %1, %1, enable=none\n}}"), diagnostic);
    }
    refuse("sc.graph @bad {\n%0 = sc.state %1 : u8\n%1 = sc.constant 1 : u8\n%2 = sc.store %0, %0, enable=none\n}", "not defined before use");
}

#[test]
fn register_store_requires_one_driver_real_target_exact_word_and_bool_enable() {
    let prefix = "sc.graph @bad {\n%0 = sc.constant 1 : u8\n%1 = sc.state %0 : u8\n";
    for (tail, diagnostic) in [
        ("}", "exactly one Store"),
        (
            "%2 = sc.store %1, %0, enable=none\n%3 = sc.store %1, %0, enable=none\n}",
            "multiple Store",
        ),
        ("%2 = sc.store %0, %0, enable=none\n}", "must be a State"),
        (
            "%2 = sc.constant 1 : i8\n%3 = sc.store %1, %2, enable=none\n}",
            "exact state type",
        ),
        ("%2 = sc.store %1, %0, enable=%0\n}", "Bool type"),
        (
            "%2 = sc.store %1, %99, enable=none\n}",
            "not defined before use",
        ),
        (
            "%2 = sc.store %1, %0, enable=%99\n}",
            "not defined before use",
        ),
        ("%1 = sc.store %1, %0, enable=none\n}", "already defined"),
    ] {
        refuse(&format!("{prefix}{tail}"), diagnostic);
    }
    refuse("sc.graph @bad {\n%0 = sc.constant 1 : fixed<8,2>\n%1 = sc.state %0 : fixed<8,2>\n%2 = sc.constant 1 : fixed<8,3>\n%3 = sc.store %1, %2, enable=none\n}", "exact state type");
}

#[test]
fn no_terminator_can_be_read_by_any_later_operation() {
    for producer in [
        "%2 = sc.store %1, %0, enable=none",
        "sc.output \"first\" %1",
    ] {
        for consumer in [
            "sc.output \"bad\" %2",
            "%3 = sc.word add %2, %0 : u8",
            "%3 = sc.store %1, %2, enable=none",
            "%3 = sc.store %1, %0, enable=%2",
            "%3 = sc.state %2 : u8",
        ] {
            refuse(&format!("sc.graph @bad {{\n%0 = sc.constant 1 : u8\n%1 = sc.state %0 : u8\n{producer}\n{consumer}\n}}"), "not defined before use");
        }
    }
}

#[test]
fn register_resource_estimates_count_storage_and_enabled_mux_widths() {
    use sc_neurocore_engine::ir::sv_target::SvTarget;
    for enabled in [false, true] {
        let mut b = ScGraphBuilder::new("resources");
        let enable = b.input("enabled", ScType::Bool);
        let initial = b.constant(ScConst::U64(3), ScType::UInt { width: 13 });
        let state = b.state(initial, ScType::UInt { width: 13 });
        b.store(state, state, enabled.then_some(enable));
        let (sv, report) =
            emit_sv::emit_systemverilog_with_target(&b.build(), SvTarget::Generic).unwrap();
        assert!(sv.contains("reg [12:0]"));
        assert_eq!(report.ff_estimated, 128 + 8 + 13);
        assert_eq!(
            report.lut_estimated,
            128 + 16 + if enabled { 13 } else { 0 }
        );
        assert_eq!(report.dsp_estimated, 0);
    }
}
