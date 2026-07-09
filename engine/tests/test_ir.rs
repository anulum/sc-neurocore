// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

use sc_neurocore_engine::ir::builder::ScGraphBuilder;
use sc_neurocore_engine::ir::graph::*;
use sc_neurocore_engine::ir::parser;
use sc_neurocore_engine::ir::printer;
use sc_neurocore_engine::ir::verify;

#[test]
fn empty_graph_verifies() {
    let g = ScGraphBuilder::new("empty").build();
    assert!(verify::verify(&g).is_ok());
}

#[test]
fn single_encode_pipeline() {
    let mut b = ScGraphBuilder::new("single_encode");
    let x = b.input("x_in", ScType::Rate);
    let bs = b.encode(x, 1024, 0xACE1);
    let pc = b.popcount(bs);
    b.output("result", pc);
    let g = b.build();

    assert_eq!(g.len(), 4);
    assert!(verify::verify(&g).is_ok());
}

#[test]
fn synapse_pipeline() {
    let mut b = ScGraphBuilder::new("synapse");
    let x = b.input("x_in", ScType::Rate);
    let w = b.constant(ScConst::F64(0.5), ScType::Rate);
    let x_bs = b.encode(x, 1024, 0xACE1);
    let w_bs = b.encode(w, 1024, 0xBEEF);
    let syn = b.bitwise_and(x_bs, w_bs);
    let pc = b.popcount(syn);
    b.output("synapse_count", pc);
    let g = b.build();

    assert_eq!(g.len(), 7);
    assert!(verify::verify(&g).is_ok());
}

#[test]
fn dense_layer_graph() {
    let mut b = ScGraphBuilder::new("dense_net");
    let inputs = b.input(
        "inputs",
        ScType::Vec {
            element: Box::new(ScType::Rate),
            count: 3,
        },
    );
    let weights = b.input(
        "weights",
        ScType::Vec {
            element: Box::new(ScType::Rate),
            count: 21, // 7 * 3
        },
    );
    let leak = b.constant(ScConst::I64(20), ScType::FixedPoint { width: 16, frac: 8 });
    let gain = b.constant(ScConst::I64(256), ScType::FixedPoint { width: 16, frac: 8 });
    let spikes = b.dense_forward(
        inputs,
        weights,
        leak,
        gain,
        DenseParams {
            n_inputs: 3,
            n_neurons: 7,
            stream_length: 1024,
            ..DenseParams::default()
        },
    );
    b.output("spikes", spikes);
    let g = b.build();

    assert_eq!(g.len(), 6);
    assert!(verify::verify(&g).is_ok());
}

#[test]
fn duplicate_id_rejected() {
    let mut g = ScGraph::new("bad_ssa");
    g.push(ScOp::Input {
        id: ValueId(0),
        name: "a".to_string(),
        ty: ScType::Rate,
    });
    g.push(ScOp::Input {
        id: ValueId(0), // duplicate
        name: "b".to_string(),
        ty: ScType::Rate,
    });

    let result = verify::verify(&g);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .iter()
        .any(|e| e.message.contains("already defined")));
}

#[test]
fn undefined_operand_rejected() {
    let mut g = ScGraph::new("bad_ref");
    g.push(ScOp::Input {
        id: ValueId(0),
        name: "x".to_string(),
        ty: ScType::Rate,
    });
    g.push(ScOp::Popcount {
        id: ValueId(1),
        input: ValueId(99), // not defined
    });

    let result = verify::verify(&g);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .iter()
        .any(|e| e.message.contains("not defined")));
}

#[test]
fn printer_round_trip() {
    let mut b = ScGraphBuilder::new("roundtrip");
    let x = b.input("x_in", ScType::Rate);
    let w = b.constant(ScConst::F64(0.5), ScType::Rate);
    let x_bs = b.encode(x, 1024, 0xACE1);
    let w_bs = b.encode(w, 1024, 0xBEEF);
    let syn = b.bitwise_and(x_bs, w_bs);
    let pc = b.popcount(syn);
    b.output("result", pc);
    let g = b.build();

    let text = printer::print(&g);
    let g2 = parser::parse(&text).expect("parse should succeed");
    let text2 = printer::print(&g2);

    assert_eq!(g.name, g2.name);
    assert_eq!(g.len(), g2.len());
    assert_eq!(text, text2);
    assert_eq!(g, g2);
}

#[test]
fn type_display() {
    assert_eq!(ScType::Rate.to_string(), "rate");
    assert_eq!(
        ScType::Bitstream { length: 1024 }.to_string(),
        "bitstream<1024>"
    );
    assert_eq!(
        ScType::FixedPoint { width: 16, frac: 8 }.to_string(),
        "fixed<16,8>"
    );
    assert_eq!(
        ScType::Vec {
            element: Box::new(ScType::Bool),
            count: 7,
        }
        .to_string(),
        "vec<bool,7>"
    );
}

#[test]
fn value_id_display() {
    assert_eq!(ValueId(0).to_string(), "%0");
    assert_eq!(ValueId(42).to_string(), "%42");
}

#[test]
fn op_name_coverage() {
    // Ensure every op variant has a unique textual name.
    let names = [
        "sc.input",
        "sc.output",
        "sc.constant",
        "sc.encode",
        "sc.and",
        "sc.popcount",
        "sc.lif_step",
        "sc.dense_forward",
        "sc.xor",
        "sc.reduce",
        "sc.graph_forward",
        "sc.softmax_attention",
        "sc.kuramoto_step",
        "sc.scale",
        "sc.offset",
        "sc.div_const",
    ];
    let unique: std::collections::HashSet<&&str> = names.iter().collect();
    assert_eq!(names.len(), unique.len());
}

#[test]
fn xor_round_trip() {
    let mut b = ScGraphBuilder::new("xor_test");
    let a = b.input("a", ScType::Bitstream { length: 512 });
    let c = b.input("b", ScType::Bitstream { length: 512 });
    let x = b.bitwise_xor(a, c);
    b.output("out", x);
    let g = b.build();

    let text = printer::print(&g);
    let g2 = parser::parse(&text).expect("parse xor");
    assert_eq!(g, g2);
}

#[test]
fn reduce_round_trip() {
    let mut b = ScGraphBuilder::new("reduce_test");
    let x = b.input("x", ScType::Rate);
    let s = b.reduce(x, ReduceMode::Sum);
    let m = b.reduce(x, ReduceMode::Max);
    b.output("sum", s);
    b.output("max", m);
    let g = b.build();

    let text = printer::print(&g);
    let g2 = parser::parse(&text).expect("parse reduce");
    assert_eq!(g, g2);
}

#[test]
fn graph_forward_round_trip() {
    let mut b = ScGraphBuilder::new("gnn_test");
    let feat = b.input("features", ScType::Rate);
    let adj = b.input("adjacency", ScType::Rate);
    let out = b.graph_forward(feat, adj, 8, 4);
    b.output("gnn_out", out);
    let g = b.build();

    let text = printer::print(&g);
    let g2 = parser::parse(&text).expect("parse graph_forward");
    assert_eq!(g, g2);
}

#[test]
fn softmax_attention_round_trip() {
    let mut b = ScGraphBuilder::new("attn_test");
    let q = b.input("q", ScType::Rate);
    let k = b.input("k", ScType::Rate);
    let v = b.input("v", ScType::Rate);
    let a = b.softmax_attention(q, k, v, 64);
    b.output("attn", a);
    let g = b.build();

    let text = printer::print(&g);
    let g2 = parser::parse(&text).expect("parse softmax_attention");
    assert_eq!(g, g2);
}

#[test]
fn kuramoto_step_round_trip() {
    let mut b = ScGraphBuilder::new("kuramoto_test");
    let ph = b.input("phases", ScType::Rate);
    let om = b.input("omega", ScType::Rate);
    let kk = b.input("coupling", ScType::Rate);
    let out = b.kuramoto_step(ph, om, kk, 0.001);
    b.output("next_phases", out);
    let g = b.build();

    let text = printer::print(&g);
    let g2 = parser::parse(&text).expect("parse kuramoto_step");
    assert_eq!(g, g2);
}

#[test]
fn new_ops_verify() {
    let mut b = ScGraphBuilder::new("all_new_ops");
    let a = b.input("a", ScType::Rate);
    let c = b.input("b", ScType::Rate);
    let d = b.input("c", ScType::Rate);
    let _ = b.bitwise_xor(a, c);
    let _ = b.reduce(a, ReduceMode::Sum);
    let _ = b.graph_forward(a, c, 4, 2);
    let _ = b.softmax_attention(a, c, d, 32);
    let _ = b.kuramoto_step(a, c, d, 0.01);
    let g = b.build();
    assert!(verify::verify(&g).is_ok());
}

// --- Parse/print round-trip regressions (found by the roundtrip_ir fuzz target) ---

/// A whole-number `f64` scalar constant must survive `parse . print`.
///
/// Regression: `format!("{}", 5.0)` is `"5"`, so a `F64(5.0)` with a non-`rate` type used
/// to re-parse to an integer variant (`U64(5)`), silently changing the graph.
#[test]
fn whole_number_float_scalar_constant_round_trips() {
    let text = "sc.graph @whole_scalar {\n  %0 = sc.constant 5.0 : u64\n}\n";
    let g = parser::parse(text).expect("parse whole-number float constant");
    let g2 = parser::parse(&printer::print(&g)).expect("re-parse printed graph");
    assert_eq!(
        g, g2,
        "parse . print changed a whole-number float scalar constant"
    );
    match &g2.ops[0] {
        ScOp::Constant {
            value: ScConst::F64(v),
            ..
        } => assert_eq!(*v, 5.0),
        other => panic!("expected an F64 constant after round trip, got {other:?}"),
    }
}

/// An all-whole-number `f64` vector constant must survive `parse . print`.
///
/// Regression: `[5.0, 6.0]` printed as `"[5, 6]"`, which re-parsed to `I64Vec`.
#[test]
fn whole_number_float_vector_constant_round_trips() {
    let text = "sc.graph @whole_vec {\n  %0 = sc.constant [1.0, 2.0, 3.0] : rate\n}\n";
    let g = parser::parse(text).expect("parse whole-number float vector");
    let g2 = parser::parse(&printer::print(&g)).expect("re-parse printed graph");
    assert_eq!(
        g, g2,
        "parse . print changed a whole-number float vector constant"
    );
    match &g2.ops[0] {
        ScOp::Constant {
            value: ScConst::F64Vec(v),
            ..
        } => assert_eq!(v, &vec![1.0, 2.0, 3.0]),
        other => panic!("expected an F64Vec constant after round trip, got {other:?}"),
    }
}

/// Non-finite float literals are rejected at parse time.
///
/// `NaN` breaks the round trip (`NaN != NaN`) and is meaningless as an SC constant; `inf`
/// is likewise rejected. This keeps the text format total under `parse . print`.
#[test]
fn non_finite_float_constants_are_rejected() {
    for literal in ["NaN", "inf", "-inf"] {
        let text = format!("sc.graph @nf {{\n  %0 = sc.constant {literal} : rate\n}}\n");
        assert!(
            parser::parse(&text).is_err(),
            "parser must reject the non-finite constant '{literal}'"
        );
    }
    // The same guard applies to float-valued op parameters (e.g. sc.scale factor).
    let scale_nan =
        "sc.graph @nf2 {\n  %0 = sc.input \"x\" : rate\n  %1 = sc.scale %0, factor=NaN : rate\n}\n";
    assert!(
        parser::parse(scale_nan).is_err(),
        "parser must reject a non-finite scale factor"
    );
}
