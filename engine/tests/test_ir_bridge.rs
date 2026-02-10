//! Tests for IR bridge types exposed via lib.rs.
//! These verify that the PyO3 wrappers correctly delegate to the IR module.

use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder, emit_sv::emit, parser::parse as ir_parse, printer::print as ir_print,
    verify::verify,
};

#[test]
fn ir_bridge_build_and_verify() {
    let mut b = ScGraphBuilder::new("bridge_test");
    let x = b.input("x", sc_neurocore_engine::ir::graph::ScType::Rate);
    let enc = b.encode(x, 1024, 0xACE1);
    let count = b.popcount(enc);
    b.output("count", count);
    let g = b.build();
    assert!(verify(&g).is_ok());
}

#[test]
fn ir_bridge_print_parse_roundtrip() {
    let mut b = ScGraphBuilder::new("roundtrip");
    let x = b.input("x", sc_neurocore_engine::ir::graph::ScType::Rate);
    let enc = b.encode(x, 512, 0xACE1);
    let count = b.popcount(enc);
    b.output("count", count);
    let g = b.build();

    let text = ir_print(&g);
    let parsed = ir_parse(&text).expect("parse failed");
    let text2 = ir_print(&parsed);
    assert_eq!(text, text2);
}

#[test]
fn ir_bridge_emit_sv() {
    let mut b = ScGraphBuilder::new("sv_bridge");
    let x = b.input("x", sc_neurocore_engine::ir::graph::ScType::Rate);
    let w = b.input("w", sc_neurocore_engine::ir::graph::ScType::Rate);
    let x_enc = b.encode(x, 1024, 0xACE1);
    let w_enc = b.encode(w, 1024, 0xBEEF);
    let product = b.bitwise_and(x_enc, w_enc);
    b.output("out", product);
    let g = b.build();

    let sv = emit(&g);
    assert!(sv.contains("module"));
    assert!(sv.contains("sv_bridge"));
    assert!(sv.contains("endmodule"));
}
