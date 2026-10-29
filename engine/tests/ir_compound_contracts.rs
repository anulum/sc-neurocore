// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public compound graph admission and composition regressions

//! Exercise the real public parser, verifier and both hardware emitters.

use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder, emit_mlir, emit_sv, graph::*, parser, printer, verify, word::WordOp,
};

/// Construct a complete externally driven LIF without relying on any private helper.
fn lif_graph() -> ScGraph {
    let mut b = ScGraphBuilder::new("lif_contract");
    let ty = ScType::FixedPoint { width: 16, frac: 8 };
    let current = b.input("current", ty.clone());
    let leak = b.input("leak", ty.clone());
    let gain = b.input("gain", ty.clone());
    let noise = b.input("noise", ty);
    let spike = b.lif_step(current, leak, gain, noise, LifParams::default());
    b.output("result", spike);
    b.build()
}

/// Require consistent refusal at graph admission and at both real emission entry points.
fn refused(graph: &ScGraph, reason: &str) {
    let errors = verify::verify(graph).expect_err("invalid hardware graph passed admission");
    assert!(
        errors.iter().any(|error| error.message.contains(reason)),
        "{errors:?}"
    );
    for result in [emit_sv::emit(graph), emit_mlir::emit(graph)] {
        let error = result.expect_err("invalid hardware graph was emitted");
        assert!(error.contains(reason), "{error}");
    }
}

/// Require real public admission, both lowerers and exact source/text immutability.
fn admitted(graph: &ScGraph) {
    let original = printer::print(graph);
    verify::verify(graph).unwrap();
    emit_sv::emit(graph).unwrap();
    emit_mlir::emit(graph).unwrap();
    assert_eq!(printer::print(graph), original);
    let restored = parser::parse(&original).unwrap();
    assert_eq!(printer::print(&restored), original);
    verify::verify(&restored).unwrap();
}

#[test]
/// A physical width match cannot conceal a different declared binary point.
fn lif_requires_matching_binary_points_before_either_emitter() {
    let mut graph = lif_graph();
    let ScOp::Input { ty, .. } = &mut graph.ops[0] else {
        unreachable!()
    };
    *ty = ScType::FixedPoint { width: 16, frac: 4 };
    refused(&graph, "Q8");
}

#[test]
/// Invalid widths, fractions and parameter words fail before any hardware dimensions form.
fn lif_rejects_invalid_state_geometry_and_unrepresentable_parameters() {
    for (width, fraction, threshold, reason) in [
        (0, 0, 0, "1..64"),
        (65, 8, 256, "1..64"),
        (16, 16, 256, "smaller fraction"),
        (8, 4, 256, "must fit"),
    ] {
        let mut graph = lif_graph();
        let ScOp::LifStep { params, .. } = &mut graph.ops[4] else {
            unreachable!()
        };
        params.data_width = width;
        params.fraction = fraction;
        params.v_threshold = threshold;
        refused(&graph, reason);
    }
}

#[test]
/// One source-bound spike may feed integer arithmetic and a delayed synapse simultaneously.
fn lif_spike_composes_as_boolean_without_changing_the_legacy_port_contract() {
    let mut b = ScGraphBuilder::new("spike_composition");
    let ty = ScType::FixedPoint { width: 16, frac: 8 };
    let current = b.input("current", ty.clone());
    let leak = b.constant(ScConst::I64(16), ty.clone());
    let gain = b.constant(ScConst::I64(256), ty.clone());
    let noise = b.constant(ScConst::I64(0), ty);
    let spike = b.lif_step(current, leak, gain, noise, LifParams::default());
    let word = b.word(WordOp::Resize {
        input: spike,
        ty: ScType::UInt { width: 16 },
    });
    let scaled = b.scale(spike, 0.5);
    let weights = b.constant(
        ScConst::I64Vec(vec![256]),
        ScType::Vec {
            element: Box::new(ScType::Rate),
            count: 1,
        },
    );
    let centre = b.constant(ScConst::I64(0), ScType::Rate);
    let sigma = b.constant(ScConst::I64(256), ScType::Rate);
    let delayed = b.dcls_layer(
        spike,
        weights,
        centre,
        sigma,
        DclsParams {
            n_taps: 1,
            delay_depth: 4,
            ptr_width: 2,
            tap_offsets: vec![1],
            ..DclsParams::default()
        },
    );
    b.output("legacy_spike", spike);
    b.output("word", word);
    b.output("scaled", scaled);
    b.output("delayed", delayed);
    b.bind_source("{\"case\":\"typed-spike\"}".into()).unwrap();
    admitted(&b.build());
}

#[test]
/// Packed neuron events remain typed numeric lanes when reduced and registered.
fn dense_spike_vector_composes_with_numeric_reduction() {
    let mut b = ScGraphBuilder::new("dense_reduction");
    let inputs = b.input(
        "values",
        ScType::Vec {
            element: Box::new(ScType::Rate),
            count: 2,
        },
    );
    let weights = b.constant(
        ScConst::F64Vec(vec![0.0, 0.0, 1.0, 1.0]),
        ScType::Vec {
            element: Box::new(ScType::Rate),
            count: 4,
        },
    );
    let leak = b.constant(ScConst::I64(16), ScType::Rate);
    let gain = b.constant(ScConst::I64(256), ScType::Rate);
    let spikes = b.dense_forward(
        inputs,
        weights,
        leak,
        gain,
        DenseParams {
            n_inputs: 2,
            n_neurons: 2,
            ..DenseParams::default()
        },
    );
    let count = b.reduce(spikes, ReduceMode::Sum);
    let delayed = b.word(WordOp::Delay {
        input: count,
        ty: ScType::UInt { width: 64 },
    });
    b.output("spike_count", count);
    b.output("delayed_count", delayed);
    admitted(&b.build());
}

#[test]
/// Temporal spikes are valid delay inputs; every configured tap must address real storage.
fn dcls_accepts_temporal_spikes_and_checks_all_delay_addresses() {
    let text = "sc.graph @dcls_addresses {\n  %0 = sc.input \"spike\" : bitstream<7>\n  %1 = sc.input \"weights\" : vec<fixed<16,8>,1>\n  %2 = sc.input \"centre\" : fixed<16,8>\n  %3 = sc.input \"sigma\" : fixed<16,8>\n  %4 = sc.dcls_layer %0, weights=%1, centre=%2, sigma=%3, taps=1, depth=7, dw=16, frac=8, ptr=3, offsets=[6] : fixed<16,8>\n  sc.output \"result\" %4\n}\n";
    let graph = parser::parse(text).unwrap();
    admitted(&graph);
    for (depth, pointer, offsets) in [
        (0, 3, vec![0]),
        (7, 2, vec![0]),
        (7, 33, vec![0]),
        (7, 3, vec![7]),
        (7, 3, vec![]),
    ] {
        let mut invalid = graph.clone();
        let ScOp::DclsLayer { params, .. } = &mut invalid.ops[4] else {
            unreachable!()
        };
        params.delay_depth = depth;
        params.ptr_width = pointer;
        params.tap_offsets = offsets;
        refused(&invalid, "tap offsets");
    }
}

#[test]
/// Zero, overflowing and mismatched graph dimensions are admission failures rather than emitter panics.
fn malformed_kernel_dimensions_are_refused_before_hardware_port_construction() {
    for (nodes, features) in [(0, 2), (2, 0), (usize::MAX, 2), (2, 3)] {
        let mut b = ScGraphBuilder::new("invalid_graph_geometry");
        let input = b.constant(
            ScConst::F64Vec(vec![1.0; 4]),
            ScType::Vec {
                element: Box::new(ScType::Rate),
                count: 4,
            },
        );
        let adjacency = b.constant(
            ScConst::F64Vec(vec![0.0; 4]),
            ScType::Vec {
                element: Box::new(ScType::Rate),
                count: 4,
            },
        );
        let output = b.graph_forward(input, adjacency, nodes, features);
        b.output("result", output);
        refused(&b.build(), "GraphForward");
    }
}

#[test]
/// Attention row geometry and phase matrix geometry have the same public refusal boundary.
fn attention_and_phase_shape_errors_are_public_admission_errors() {
    let attention = parser::parse("sc.graph @bad_attention {\n  %0 = sc.constant [1.0, 2.0] : vec<rate,2>\n  %1 = sc.constant [1.0, 2.0] : vec<rate,2>\n  %2 = sc.constant [1.0] : vec<rate,1>\n  %3 = sc.softmax_attention %0, %1, %2, dim_k=0 : rate\n  sc.output \"result\" %3\n}\n").unwrap();
    refused(&attention, "SoftmaxAttention");
    let phase = parser::parse("sc.graph @bad_phase {\n  %0 = sc.constant [0.0, 0.1] : vec<rate,2>\n  %1 = sc.constant [1.0] : vec<rate,1>\n  %2 = sc.constant [0.0] : vec<rate,1>\n  %3 = sc.kuramoto_step %0, omega=%1, K=%2, dt=0.01 : rate\n  sc.output \"result\" %3\n}\n").unwrap();
    refused(&phase, "KuramotoStep");
}

#[test]
/// Nonfinite coefficients and vector/scalar ambiguity cannot pass graph verification.
fn invalid_scalar_coefficients_and_vector_scaling_are_refused_at_admission() {
    for coefficient in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
        let mut b = ScGraphBuilder::new("invalid_coefficient");
        let input = b.input("value", ScType::Rate);
        let output = b.scale(input, coefficient);
        b.output("result", output);
        refused(&b.build(), "coefficient");
    }
    let mut b = ScGraphBuilder::new("invalid_vector_scale");
    let input = b.input(
        "value",
        ScType::Vec {
            element: Box::new(ScType::Rate),
            count: 2,
        },
    );
    let output = b.scale(input, 1.0);
    b.output("result", output);
    refused(&b.build(), "numeric scalar");
}

#[test]
/// A self-reference produces an ordinary SSA error rather than recursive inference failure.
fn malformed_reduce_cycle_is_reported_without_recursive_type_inference() {
    let graph = parser::parse("sc.graph @cycle {\n  %0 = sc.input \"value\" : vec<rate,2>\n  %1 = sc.reduce %1, mode=sum : rate\n  sc.output \"result\" %1\n}\n").unwrap();
    refused(&graph, "not defined before use");
}

#[test]
/// Dynamic DCLS operands must have the declared physical meaning before bus construction.
fn dcls_invalid_formats_and_operand_roles_are_rejected() {
    for case in 0..12 {
        let ty = ScType::FixedPoint { width: 16, frac: 8 };
        let mut spike_ty = ScType::Bool;
        let mut weight_ty = ScType::Vec {
            element: Box::new(ty.clone()),
            count: 1,
        };
        let mut centre_ty = ty.clone();
        let mut sigma_ty = ty;
        let mut params = DclsParams {
            n_taps: 1,
            tap_offsets: vec![0],
            ..DclsParams::default()
        };
        let reason = match case {
            0 => {
                params.data_width = 1;
                "2..64"
            }
            1 => {
                params.n_taps = 0;
                "nonempty taps"
            }
            2 => {
                params.fraction = 16;
                "fixed-point word"
            }
            3 => {
                spike_ty = ScType::UInt { width: 16 };
                "Boolean"
            }
            4 => {
                weight_ty = ScType::Rate;
                "packed vector"
            }
            5 => {
                weight_ty = ScType::Vec {
                    element: Box::new(ScType::UInt { width: 16 }),
                    count: 1,
                };
                "weight lanes"
            }
            6 => {
                weight_ty = ScType::Vec {
                    element: Box::new(ScType::FixedPoint { width: 16, frac: 4 }),
                    count: 1,
                };
                "weight lanes"
            }
            7 => {
                weight_ty = ScType::Vec {
                    element: Box::new(ScType::Rate),
                    count: 2,
                };
                "weight lanes"
            }
            8 => {
                centre_ty = ScType::UInt { width: 16 };
                "signed 16-bit"
            }
            9 => {
                sigma_ty = ScType::SInt { width: 32 };
                "signed 16-bit"
            }
            10 => {
                params.delay_depth = i32::MAX as usize + 1;
                "tap offsets"
            }
            11 => {
                params.ptr_width = 0;
                "pointer width"
            }
            _ => unreachable!(),
        };
        let mut b = ScGraphBuilder::new("invalid_dcls_format");
        let spike = b.input("spike", spike_ty);
        let weights = b.input("weights", weight_ty);
        let centre = b.input("centre", centre_ty);
        let sigma = b.input("sigma", sigma_ty);
        let output = b.dcls_layer(spike, weights, centre, sigma, params);
        b.output("result", output);
        refused(&b.build(), reason);
    }
}

#[test]
/// Encoded signed words retain raw units; unsigned and temporal scalar ambiguity is refused.
fn signed_raw_state_operands_are_explicit_and_numeric_reductions_refuse_streams() {
    let mut graph = lif_graph();
    for op in &mut graph.ops[..4] {
        let ScOp::Input { ty, .. } = op else {
            unreachable!()
        };
        *ty = ScType::SInt { width: 16 };
    }
    admitted(&graph);
    for ty in [
        ScType::Bool,
        ScType::UInt { width: 16 },
        ScType::Bitstream { length: 7 },
    ] {
        let mut invalid = graph.clone();
        let ScOp::Input { ty: input_ty, .. } = &mut invalid.ops[0] else {
            unreachable!()
        };
        *input_ty = ty;
        refused(&invalid, "signed 16-bit");
    }
    for reduce in [false, true] {
        let mut b = ScGraphBuilder::new("invalid_temporal_arithmetic");
        let input = b.input("value", ScType::Bitstream { length: 7 });
        let output = if reduce {
            b.reduce(input, ReduceMode::Max)
        } else {
            b.offset(input, 1.0)
        };
        b.output("result", output);
        refused(&b.build(), "numeric arithmetic requires scalar lanes");
    }
}

#[test]
/// Constant-only kernels refuse live matrices, incomplete rows and unrepresentable real units.
fn kernel_constants_and_row_geometry_have_public_diagnostics() {
    for (q, k, v, dim) in [
        (vec![1.0], vec![1.0, 2.0], vec![1.0], 2),
        (vec![1.0], vec![], vec![1.0], 1),
        (vec![1.0], vec![1.0], vec![], 1),
        (vec![1.0], vec![1.0, 2.0], vec![1.0, 2.0, 3.0], 1),
    ] {
        let mut b = ScGraphBuilder::new("invalid_attention_rows");
        let mut vector = |values: Vec<f64>| {
            b.constant(
                ScConst::F64Vec(values.clone()),
                ScType::Vec {
                    element: Box::new(ScType::Rate),
                    count: values.len(),
                },
            )
        };
        let q = vector(q);
        let k = vector(k);
        let v = vector(v);
        let result = b.softmax_attention(q, k, v, dim);
        b.output("result", result);
        let graph = b.build();
        let reason = if graph.ops.iter().any(|op| {
            matches!(
                op,
                ScOp::Constant {
                    ty: ScType::Vec { count: 0, .. },
                    ..
                }
            )
        }) {
            "word count must be positive"
        } else {
            "SoftmaxAttention"
        };
        refused(&graph, reason);
    }
    for value in [f64::NAN, f64::INFINITY, 128.0, -129.0] {
        let mut b = ScGraphBuilder::new("invalid_kernel_constant");
        let features = b.constant(
            ScConst::F64Vec(vec![value]),
            ScType::Vec {
                element: Box::new(ScType::FixedPoint { width: 64, frac: 8 }),
                count: 1,
            },
        );
        let adjacency = b.constant(
            ScConst::I64Vec(vec![1]),
            ScType::Vec {
                element: Box::new(ScType::Rate),
                count: 1,
            },
        );
        let result = b.graph_forward(features, adjacency, 1, 1);
        b.output("result", result);
        refused(&b.build(), "GraphForward");
    }
    let mut b = ScGraphBuilder::new("live_graph_matrix");
    let ty = ScType::Vec {
        element: Box::new(ScType::Rate),
        count: 1,
    };
    let features = b.input("features", ty.clone());
    let adjacency = b.input("adjacency", ty);
    let result = b.graph_forward(features, adjacency, 1, 1);
    b.output("result", result);
    refused(&b.build(), "constant numeric vectors");
    for dt in [f64::NAN, f64::INFINITY, 128.0] {
        let mut b = ScGraphBuilder::new("invalid_phase_dt");
        let ty = ScType::Vec {
            element: Box::new(ScType::Rate),
            count: 1,
        };
        let phase = b.constant(ScConst::I64Vec(vec![0]), ty.clone());
        let omega = b.constant(ScConst::I64Vec(vec![1]), ty.clone());
        let coupling = b.constant(ScConst::I64Vec(vec![0]), ty);
        let result = b.kuramoto_step(phase, omega, coupling, dt);
        b.output("result", result);
        refused(&b.build(), "dt is not representable");
    }
}

#[test]
/// Scalar phase storage cannot be reinterpreted as a constant oscillator vector.
fn phase_kernel_refuses_scalar_phase_operands() {
    let mut b = ScGraphBuilder::new("scalar_phase_operand");
    let phase = b.constant(ScConst::I64(0), ScType::Rate);
    let ty = ScType::Vec {
        element: Box::new(ScType::Rate),
        count: 1,
    };
    let omega = b.constant(ScConst::I64Vec(vec![1]), ty.clone());
    let coupling = b.constant(ScConst::I64Vec(vec![0]), ty);
    let output = b.kuramoto_step(phase, omega, coupling, 0.01);
    b.output("result", output);
    refused(&b.build(), "constant numeric vectors");
}
