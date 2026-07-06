// SPDX-License-Identifier: AGPL-3.0-or-later
use sc_neurocore_engine::ir::builder::ScGraphBuilder;
use sc_neurocore_engine::ir::emit_sv;
use sc_neurocore_engine::ir::graph::*;
use sc_neurocore_engine::ir::verify;

#[test]
fn emit_single_synapse() {
    let mut b = ScGraphBuilder::new("test_synapse");
    let x = b.input("x_in", ScType::Rate);
    let w = b.constant(ScConst::F64(0.5), ScType::Rate);
    let x_bs = b.encode(x, 1024, 0xACE1);
    let w_bs = b.encode(w, 1024, 0xBEEF);
    let syn = b.bitwise_and(x_bs, w_bs);
    let pc = b.popcount(syn);
    b.output("result", pc);
    let g = b.build();

    assert!(verify::verify(&g).is_ok());
    let sv = emit_sv::emit(&g).unwrap();

    // Structural checks.
    assert!(sv.contains("module test_synapse"));
    assert!(sv.contains("sc_bitstream_encoder"));
    assert!(sv.contains("sc_bitstream_synapse"));
    assert!(sv.contains("SEED_INIT(16'hACE1)"));
    assert!(sv.contains("SEED_INIT(16'hBEEF)"));
    assert!(sv.contains("endmodule"));
}

#[test]
fn emit_dense_layer() {
    let mut b = ScGraphBuilder::new("test_dense");
    let inputs = b.input(
        "x_fp",
        ScType::Vec {
            element: Box::new(ScType::FixedPoint { width: 16, frac: 8 }),
            count: 3,
        },
    );
    let weights = b.input(
        "w_fp",
        ScType::Vec {
            element: Box::new(ScType::FixedPoint { width: 16, frac: 8 }),
            count: 3,
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
    b.output("spike_out", spikes);
    let g = b.build();

    assert!(verify::verify(&g).is_ok());
    let sv = emit_sv::emit(&g).unwrap();

    assert!(sv.contains("module test_dense"));
    assert!(sv.contains("sc_dense_layer_core"));
    assert!(sv.contains("N_INPUTS(3)"));
    assert!(sv.contains("N_NEURONS(7)"));
    assert!(sv.contains("endmodule"));
}

#[test]
fn emit_lif_neuron() {
    let mut b = ScGraphBuilder::new("test_lif");
    let current = b.input("I_t", ScType::FixedPoint { width: 16, frac: 8 });
    let leak = b.constant(ScConst::I64(20), ScType::FixedPoint { width: 16, frac: 8 });
    let gain = b.constant(ScConst::I64(256), ScType::FixedPoint { width: 16, frac: 8 });
    let noise = b.constant(ScConst::I64(0), ScType::FixedPoint { width: 16, frac: 8 });
    let lif = b.lif_step(current, leak, gain, noise, LifParams::default());
    b.output("spike", lif);
    let g = b.build();

    assert!(verify::verify(&g).is_ok());
    let sv = emit_sv::emit(&g).unwrap();

    assert!(sv.contains("module test_lif"));
    assert!(sv.contains("sc_lif_neuron"));
    assert!(sv.contains("V_THRESHOLD(256)"));
    assert!(sv.contains("REFRACTORY_PERIOD(2)"));
    assert!(sv.contains("endmodule"));
}

#[test]
fn emit_kuramoto_step() {
    let mut b = ScGraphBuilder::new("test_kuramoto");
    let fp = ScType::Vec {
        element: Box::new(ScType::FixedPoint {
            width: 24,
            frac: 16,
        }),
        count: 2,
    };
    let phases = b.constant(ScConst::F64Vec(vec![0.0, 1.5]), fp.clone());
    let omega = b.constant(ScConst::F64Vec(vec![0.3, -0.2]), fp);
    let coupling = b.constant(
        ScConst::F64Vec(vec![0.0, 0.5, 0.5, 0.0]),
        ScType::Vec {
            element: Box::new(ScType::FixedPoint {
                width: 24,
                frac: 16,
            }),
            count: 4,
        },
    );
    let next = b.kuramoto_step(phases, omega, coupling, 0.01);
    b.output("phases_next", next);
    let g = b.build();

    assert!(verify::verify(&g).is_ok());
    let sv = emit_sv::emit(&g).expect("KuramotoStep must emit synthesizable RTL");

    assert!(sv.contains("module test_kuramoto"));
    assert!(sv.contains("sc_kuramoto_step"));
    assert!(sv.contains(".N_OSC(2)"));
    assert!(sv.contains(".DATA_WIDTH(24)"));
    assert!(sv.contains(".FRACTION(16)"));
    assert!(sv.contains(".LUT_SIZE(64)"));
    // dt=0.01 -> round(0.01 * 2^16) = 655; 2*pi and pi moduli in Q8.16.
    assert!(sv.contains(".DT_FIXED(24'sd655)"));
    assert!(sv.contains(".PHASE_MODULUS(24'sd411775)"));
    assert!(sv.contains(".HALF_PHASE_MODULUS(24'sd205887)"));
    assert!(sv.contains("u_kuramoto_"));
    // Result bus is the packed 2*24-bit next-phase vector wired to the output.
    assert!(sv.contains("wire signed [47:0] v"));
    assert!(sv.contains("output wire [47:0] phases_next"));
    // The retired placeholder must be gone.
    assert!(!sv.contains("no synthesizable RTL implementation yet"));
    // The omega constant vector holds -0.2, so its packed wire must use a
    // well-formed signed literal (sign outside the sized base, not `16'sd-51`).
    assert!(sv.contains("-16'sd51"));
    assert!(!sv.contains("16'sd-"));
    assert!(sv.contains("endmodule"));
}

#[test]
fn kuramoto_step_rejects_mismatched_coupling() {
    let mut b = ScGraphBuilder::new("bad_kuramoto");
    let fp = ScType::Vec {
        element: Box::new(ScType::FixedPoint {
            width: 24,
            frac: 16,
        }),
        count: 2,
    };
    let phases = b.constant(ScConst::F64Vec(vec![0.0, 1.5]), fp.clone());
    let omega = b.constant(ScConst::F64Vec(vec![0.3, -0.2]), fp);
    // A 2-oscillator step needs a 4-entry coupling matrix; supply 3.
    let coupling = b.constant(
        ScConst::F64Vec(vec![0.1, 0.2, 0.3]),
        ScType::Vec {
            element: Box::new(ScType::FixedPoint {
                width: 24,
                frac: 16,
            }),
            count: 3,
        },
    );
    let next = b.kuramoto_step(phases, omega, coupling, 0.01);
    b.output("phases_next", next);
    let g = b.build();

    let err = emit_sv::emit(&g).expect_err("mismatched coupling must be rejected");
    assert!(err.contains("coupling length 3 is not 2×2"), "{err}");
}

#[test]
fn emit_graph_forward() {
    let mut b = ScGraphBuilder::new("test_graph");
    let feat_ty = ScType::Vec {
        element: Box::new(ScType::FixedPoint {
            width: 24,
            frac: 16,
        }),
        count: 4,
    };
    // Two nodes, two features; a negative feature exercises the signed literal path.
    let features = b.constant(ScConst::F64Vec(vec![0.1, 0.2, -0.3, 0.4]), feat_ty.clone());
    let adjacency = b.constant(ScConst::F64Vec(vec![1.0, 1.0, 1.0, 1.0]), feat_ty);
    let agg = b.graph_forward(features, adjacency, 2, 2);
    b.output("agg_out", agg);
    let g = b.build();

    assert!(verify::verify(&g).is_ok());
    let sv = emit_sv::emit(&g).expect("GraphForward must emit synthesizable RTL");

    assert!(sv.contains("module test_graph"));
    assert!(sv.contains("sc_graph_forward"));
    assert!(sv.contains(".N_NODES(2)"));
    assert!(sv.contains(".N_FEATURES(2)"));
    assert!(sv.contains(".DATA_WIDTH(24)"));
    assert!(sv.contains(".FRACTION(16)"));
    assert!(sv.contains("u_graph_"));
    // Result bus is the packed 2*2*24-bit aggregate wired to the output.
    assert!(sv.contains("wire signed [95:0] v"));
    assert!(sv.contains("output wire [95:0] agg_out"));
    // The retired placeholder must be gone.
    assert!(!sv.contains("no synthesizable RTL implementation yet"));
    // The feature constant holds -0.3 -> trunc(-0.3*256) = -76 in Q8.8, so its packed
    // constant must use a well-formed signed literal (sign outside the sized base).
    assert!(sv.contains("-16'sd76"));
    assert!(!sv.contains("16'sd-"));
    assert!(sv.contains("endmodule"));
}

#[test]
fn graph_forward_rejects_mismatched_adjacency() {
    let mut b = ScGraphBuilder::new("bad_graph");
    let feat_ty = ScType::Vec {
        element: Box::new(ScType::FixedPoint {
            width: 24,
            frac: 16,
        }),
        count: 4,
    };
    let features = b.constant(ScConst::F64Vec(vec![0.1, 0.2, 0.3, 0.4]), feat_ty);
    // A 2-node graph needs a 4-entry adjacency; supply 3.
    let adjacency = b.constant(
        ScConst::F64Vec(vec![1.0, 0.0, 1.0]),
        ScType::Vec {
            element: Box::new(ScType::FixedPoint {
                width: 24,
                frac: 16,
            }),
            count: 3,
        },
    );
    let agg = b.graph_forward(features, adjacency, 2, 2);
    b.output("agg_out", agg);
    let g = b.build();

    let err = emit_sv::emit(&g).expect_err("mismatched adjacency must be rejected");
    assert!(err.contains("adjacency length 3 is not 2×2"), "{err}");
}

#[test]
fn emitted_sv_has_timescale() {
    let mut b = ScGraphBuilder::new("ts_check");
    let x = b.input("x", ScType::Bool);
    b.output("y", x);
    let g = b.build();
    let sv = emit_sv::emit(&g).unwrap();
    assert!(sv.contains("`timescale 1ns / 1ps"));
}

#[test]
fn emitted_sv_has_header_comment() {
    let mut b = ScGraphBuilder::new("hdr_check");
    let x = b.input("x", ScType::Bool);
    b.output("y", x);
    let g = b.build();
    let sv = emit_sv::emit(&g).unwrap();
    assert!(sv.contains("Auto-generated by SC-NeuroCore IR Compiler"));
    assert!(sv.contains("Source graph: hdr_check"));
}
