// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Native source provenance persistence and refusal

//! Public builders, text parsing and both emitters preserve or refuse real source bindings.

use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder, emit_mlir, emit_sv, graph::*, parser, printer, verify,
};

fn subject() -> ScGraphBuilder {
    let mut b = ScGraphBuilder::new("source_subject");
    let initial = b.constant(ScConst::I64(3), ScType::SInt { width: 16 });
    let state = b.state(initial, ScType::SInt { width: 16 });
    let input = b.input("sample", ScType::SInt { width: 16 });
    b.store(state, input, None);
    b.output("result", state);
    b
}

#[test]
fn source_preserves_utf8_bytes_without_changing_either_hardware_backend() {
    let plain = subject().build();
    for payload in [
        "abc",
        "{\"equation\":\"dv/dt = -v\",\"author\":\"Šotek\"}\n",
        "line1\r\nline2\t\0",
    ] {
        let mut b = subject();
        b.bind_source(payload.into()).unwrap();
        let graph = b.build();
        verify::verify(&graph).unwrap();
        let text = printer::print(&graph);
        let restored = parser::parse(&text).unwrap();
        assert_eq!(restored, graph);
        assert_eq!(printer::print(&restored), text);
        assert_eq!(graph.source.as_ref().unwrap().payload, payload);
        assert_eq!(
            emit_sv::emit(&restored).unwrap(),
            emit_sv::emit(&plain).unwrap()
        );
        assert_eq!(
            emit_mlir::emit(&restored).unwrap(),
            emit_mlir::emit(&plain).unwrap()
        );
        if payload == "abc" {
            assert_eq!(
                graph.source.as_ref().unwrap().source_sha256,
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            );
        }
    }
}

#[test]
fn changed_source_name_or_arithmetic_is_refused_by_parser_verifier_and_emitters() {
    let mut b = subject();
    b.bind_source("abc".into()).unwrap();
    let graph = b.build();
    for change in 0..7 {
        let mut altered = graph.clone();
        if change == 0 {
            altered.source.as_mut().unwrap().payload.push('x');
        } else if change == 1 {
            altered.name.push_str("_changed");
        } else if change == 2 {
            if let ScOp::Constant { value, .. } = &mut altered.ops[0] {
                *value = ScConst::I64(4);
            } else {
                panic!("subject must retain its actual reset constant");
            }
        } else if change == 3 {
            altered.source.as_mut().unwrap().payload.clear();
        } else if change == 4 {
            altered.source.as_mut().unwrap().source_sha256 = "0".into();
        } else if change == 5 {
            altered.source.as_mut().unwrap().graph_sha256 = "A".repeat(64);
        } else {
            altered.source.as_mut().unwrap().graph_sha256 = "0".repeat(64);
        }
        assert!(verify::verify(&altered).is_err());
        assert!(emit_sv::emit(&altered).is_err());
        assert!(emit_mlir::emit(&altered).is_err());
        assert!(parser::parse(&printer::print(&altered)).is_err());
    }
}

#[test]
fn binding_refusals_preserve_existing_operations_and_source() {
    let mut b = subject();
    assert!(b.bind_source(String::new()).is_err());
    assert_eq!(b.build(), subject().build());
    let mut b = subject();
    b.bind_source("abc".into()).unwrap();
    assert!(b.bind_source("replacement".into()).is_err());
    let graph = b.build();
    assert_eq!(graph.source.as_ref().unwrap().payload, "abc");
    verify::verify(&graph).unwrap();
    let mut b = subject();
    b.bind_source("abc".into()).unwrap();
    b.input("late_input", ScType::Bool);
    assert!(verify::verify(&b.build()).is_err());
}

#[test]
fn source_text_refuses_unknown_fields_versions_checksums_and_non_utf8_payloads() {
    let mut b = subject();
    b.bind_source("abc".into()).unwrap();
    let text = printer::print(&b.build());
    let line = text.lines().nth(1).unwrap();
    let prefix = line.split("payload=").next().unwrap();
    let malformed = [
        line.replace("sc.source v1", "sc.source v2"),
        format!("{line} unknown=1"),
        line.replace("source=ba", "source=BA"),
        line.replace("source=ba", "source=ca"),
        line.replace("graph=", "other="),
        line.replace("source=", "other="),
        format!("{prefix}payload=6"),
        format!("{prefix}payload=gg"),
        format!("{prefix}payload=ff"),
        format!("{prefix}payload="),
        format!("{prefix}payload=616264"),
        line.replace("payload=", "other="),
    ];
    for replacement in malformed {
        assert!(
            parser::parse(&text.replace(line, &replacement)).is_err(),
            "{replacement}"
        );
    }
    assert!(parser::parse(&text.replace(line, &format!("{line}\n{line}"))).is_err());
    let moved = text
        .replace(&format!("{line}\n"), "")
        .replace("}\n", &format!("{line}\n}}\n"));
    assert!(parser::parse(&moved).is_err());
}

#[test]
fn legacy_graph_text_remains_source_free_and_byte_stable() {
    let graph = subject().build();
    let text = printer::print(&graph);
    assert!(!text.contains("sc.source"));
    assert!(graph.source.is_none());
    assert_eq!(printer::print(&parser::parse(&text).unwrap()), text);
}
