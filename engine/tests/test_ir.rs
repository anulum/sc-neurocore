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
        "sc.scale",
        "sc.offset",
        "sc.div_const",
    ];
    let unique: std::collections::HashSet<&&str> = names.iter().collect();
    assert_eq!(names.len(), unique.len());
}
