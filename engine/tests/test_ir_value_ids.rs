// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public graph identifier allocation contracts

//! Exercise allocation and exhaustion through the public graph and text APIs.

use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder,
    emit_sv,
    graph::{ScGraph, ScOp, ScType, ValueId, ValueIdExhausted},
    parser, printer, verify,
};

fn sparse_graph(id: u32) -> ScGraph {
    parser::parse(&format!(
        "sc.graph @sparse_ids {{\n  %{id} = sc.input \"sample\" : u8\n}}\n"
    ))
    .expect("sparse public text must parse")
}

#[test]
fn fallible_and_legacy_allocators_share_one_sequence() {
    let mut graph = ScGraph::new("allocated_ids");
    assert_eq!(graph.try_fresh_id(), Ok(ValueId(0)));
    assert_eq!(graph.fresh_id(), ValueId(1));
    assert_eq!(graph.try_fresh_id(), Ok(ValueId(2)));
    assert!(graph.is_empty(), "reserving IDs must not append operations");
}

#[test]
fn parsed_boundary_allocates_last_output_then_refuses_without_mutation() {
    let mut graph = sparse_graph(u32::MAX - 2);
    let output = graph.try_fresh_id().expect("last identifier is available");
    assert_eq!(output, ValueId(u32::MAX - 1));
    graph.push(ScOp::Output {
        id: output,
        name: "result".into(),
        source: ValueId(u32::MAX - 2),
    });
    verify::verify(&graph).expect("sparse IDs are valid SSA");
    let text = printer::print(&graph);
    let before = graph.clone();
    for _ in 0..3 {
        assert_eq!(graph.try_fresh_id(), Err(ValueIdExhausted));
        assert_eq!(graph, before);
        assert_eq!(printer::print(&graph), text);
    }
    let mut parsed = parser::parse(&text).expect("last output ID round-trips");
    assert_eq!(parsed, graph);
    assert_eq!(parsed.try_fresh_id(), Err(ValueIdExhausted));
    assert_eq!(
        emit_sv::emit(&parsed).expect("refused allocation must leave valid RTL"),
        emit_sv::emit(&before).unwrap()
    );
}

#[test]
fn legacy_exhaustion_panics_before_mutation_with_a_stable_diagnostic() {
    let mut graph = sparse_graph(u32::MAX - 1);
    let before = graph.clone();
    let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| graph.fresh_id()))
        .expect_err("legacy API must not wrap to an existing identifier");
    let message = error.downcast_ref::<String>().map(String::as_str);
    assert_eq!(
        message,
        Some("SC IR value ID space exhausted: ValueIdExhausted")
    );
    assert_eq!(graph, before);
    assert_eq!(graph.try_fresh_id(), Err(ValueIdExhausted));
    let error: &dyn std::error::Error = &ValueIdExhausted;
    assert_eq!(error.to_string(), "SC IR value ID space exhausted");
    assert!(error.source().is_none());
}

#[test]
fn ordinary_builder_keeps_its_identifiers_and_hardware_contract() {
    let mut builder = ScGraphBuilder::new("builder_ids");
    let input = builder.input("sample", ScType::UInt { width: 8 });
    let output = builder.output("result", input);
    assert_eq!((input, output), (ValueId(0), ValueId(1)));
    let graph = builder.build();
    verify::verify(&graph).unwrap();
    let parsed = parser::parse(&printer::print(&graph)).unwrap();
    assert_eq!(parsed, graph);
    assert_eq!(
        emit_sv::emit(&parsed).unwrap(),
        emit_sv::emit(&graph).unwrap()
    );
}
