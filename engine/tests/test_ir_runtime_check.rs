// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Native runtime check contracts

//! Public graph construction, persistence, emission and malformed control admission.

use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder, emit_mlir, emit_sv, graph::*, parser, printer,
    runtime_check::RuntimeCheckKind, sv_target::SvTarget, verify,
};

#[test]
fn checks_preserve_reason_enable_source_and_nonvalue_identity() {
    let mut b = ScGraphBuilder::new("SYNTHESIS_MODEL");
    let condition = b.input("valid", ScType::Bool);
    let enable = b.input("enabled", ScType::Bool);
    b.runtime_check(condition, RuntimeCheckKind::Overflow, None)
        .unwrap();
    b.runtime_check(condition, RuntimeCheckKind::ArithmeticDomain, Some(enable))
        .unwrap();
    b.output("result", condition);
    b.bind_source("checked source".into()).unwrap();
    let graph = b.build();
    assert!(verify::verify(&graph).is_ok());
    let text = printer::print(&graph);
    assert!(text.contains("sc.check overflow, %0, enable=none"));
    assert!(text.contains("sc.check domain, %0, enable=%1"));
    assert_eq!(parser::parse(&text).unwrap(), graph);
    assert!(emit_sv::emit(&graph).unwrap().contains("always_comb"));
    assert!(emit_mlir::emit(&graph)
        .unwrap()
        .contains("sv.macro.decl @SYNTHESIS_MODEL__sc_synthesis [\"SYNTHESIS\"]"));
}

#[test]
fn check_terminators_cannot_feed_ports_or_another_check() {
    for consume_in_check in [false, true] {
        let mut b = ScGraphBuilder::new("check_consumer");
        let condition = b.input("valid", ScType::Bool);
        let check = b
            .runtime_check(condition, RuntimeCheckKind::Overflow, None)
            .unwrap();
        if consume_in_check {
            b.runtime_check(check, RuntimeCheckKind::Overflow, None)
                .unwrap();
            b.output("result", condition);
        } else {
            b.output("result", check);
        }
        let graph = b.build();
        assert!(verify::verify(&graph).is_err());
        assert!(emit_sv::emit(&graph).is_err());
        assert!(emit_mlir::emit(&graph).is_err());
    }
}

#[test]
fn check_condition_and_enable_require_boolean_types() {
    for invalid_enable in [false, true] {
        let mut b = ScGraphBuilder::new("check_type");
        let boolean = b.input("valid", ScType::Bool);
        let integer = b.input("word", ScType::UInt { width: 1 });
        b.runtime_check(
            if invalid_enable { boolean } else { integer },
            RuntimeCheckKind::Overflow,
            invalid_enable.then_some(integer),
        )
        .unwrap();
        b.output("result", boolean);
        let graph = b.build();
        let errors = verify::verify(&graph).unwrap_err();
        assert!(errors
            .iter()
            .any(|error| error.message.contains("requires bool")));
        assert!(emit_sv::emit(&graph).is_err());
        assert!(emit_mlir::emit(&graph).is_err());
    }
}

#[test]
fn malformed_check_text_is_refused_without_defaulting_controls() {
    for body in [
        "overflow, %0",
        "unknown, %0, enable=none",
        "overflow, %0, enable=none, extra",
        "overflow, %0, enabled=none",
        "overflow, bad, enable=none",
        "overflow, %0, enable=bad",
    ] {
        let text = format!("sc.graph @bad_check {{\n  %0 = sc.input \"valid\" : bool\n  %1 = sc.check {body}\n}}\n");
        let error = parser::parse(&text).unwrap_err();
        assert_eq!(error.line, 3);
    }
}

#[test]
fn checks_do_not_add_synthesized_state_or_data_resources() {
    let mut b = ScGraphBuilder::new("check_resources");
    let condition = b.input("valid", ScType::Bool);
    b.output("result", condition);
    let plain = b.build();
    let mut checked = plain.clone();
    let id = checked.try_fresh_id().unwrap();
    checked.push(ScOp::Check {
        id,
        condition,
        kind: RuntimeCheckKind::Overflow,
        enable: None,
    });
    let (_, before) = emit_sv::emit_systemverilog_with_target(&plain, SvTarget::Generic).unwrap();
    let (_, after) = emit_sv::emit_systemverilog_with_target(&checked, SvTarget::Generic).unwrap();
    assert_eq!(before, after);
}
