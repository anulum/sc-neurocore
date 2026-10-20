// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neuron text serialization fidelity tests

//! Exercise real graph construction, text decoding and hardware emission.

use sc_neurocore_engine::ir::{
    builder::ScGraphBuilder, emit_sv, graph::*, parser, printer, verify,
};

fn graph_with(params: ScOp) -> ScGraph {
    let fixed = |width, frac| ScType::FixedPoint { width, frac };
    let vector = |width, frac, count| ScType::Vec {
        element: Box::new(fixed(width, frac)),
        count,
    };
    let types = match &params {
        ScOp::LifStep { params: p, .. } => [
            fixed(p.data_width, p.fraction),
            fixed(p.data_width, p.fraction),
            fixed(p.data_width, p.fraction),
            fixed(p.data_width, p.fraction),
        ],
        ScOp::DenseForward { params: p, .. } => [
            vector(p.data_width, 8, p.n_inputs),
            vector(p.data_width, 8, p.n_inputs * p.n_neurons),
            fixed(p.data_width, 8),
            fixed(p.data_width, 8),
        ],
        ScOp::DclsLayer { params: p, .. } => [
            ScType::Bool,
            vector(p.data_width, p.fraction, p.n_taps),
            fixed(p.data_width, p.fraction),
            fixed(p.data_width, p.fraction),
        ],
        _ => panic!("expected a compound neuron operation"),
    };
    let mut b = ScGraphBuilder::new("neuron_text");
    for (index, ty) in types.into_iter().enumerate() {
        b.input(format!("input_{index}"), ty);
    }
    let mut g = b.build();
    g.push(params);
    g.next_id = 5;
    let id = g.fresh_id();
    g.push(ScOp::Output {
        id,
        name: "result".into(),
        source: ValueId(4),
    });
    g
}

fn lif(params: LifParams) -> ScGraph {
    graph_with(ScOp::LifStep {
        id: ValueId(4),
        current: ValueId(0),
        leak: ValueId(1),
        gain: ValueId(2),
        noise: ValueId(3),
        params,
    })
}

fn dense(params: DenseParams) -> ScGraph {
    graph_with(ScOp::DenseForward {
        id: ValueId(4),
        inputs: ValueId(0),
        weights: ValueId(1),
        leak: ValueId(2),
        gain: ValueId(3),
        params,
    })
}

fn dcls(params: DclsParams) -> ScGraph {
    graph_with(ScOp::DclsLayer {
        id: ValueId(4),
        spike: ValueId(0),
        weights: ValueId(1),
        centre: ValueId(2),
        sigma: ValueId(3),
        params,
    })
}

fn assert_roundtrip(graph: &ScGraph) {
    verify::verify(graph).expect("source graph verifies");
    let text = printer::print(graph);
    let parsed = parser::parse(&text).expect("printer output parses");
    verify::verify(&parsed).expect("read-back graph verifies");
    assert_eq!(graph, &parsed, "typed operation parameters changed");
    assert_eq!(text, printer::print(&parsed), "text is canonical");
    for target in [
        sc_neurocore_engine::ir::sv_target::SvTarget::Generic,
        sc_neurocore_engine::ir::sv_target::SvTarget::Zynq7 {
            device: "xc7z020clg484-1".into(),
            clock_mhz: 100,
        },
        sc_neurocore_engine::ir::sv_target::SvTarget::zynq_ultrascale_plus(
            sc_neurocore_engine::ir::sv_target::SkuKind::Zu3eg,
            100,
        ),
    ] {
        assert_eq!(
            emit_sv::emit_systemverilog_with_target(graph, target.clone()).unwrap(),
            emit_sv::emit_systemverilog_with_target(&parsed, target).unwrap(),
            "serialization changed emitted RTL or target resource report"
        );
    }
}

#[test]
fn lif_roundtrip_preserves_rest_reset_and_every_configuration_field() {
    for width in [8, 16, 24, 32] {
        for fraction in [0, 4] {
            for (rest, reset, threshold, refractory) in [(-7, 3, 10, 0), (5, -11, 27, 17)] {
                assert_roundtrip(&lif(LifParams {
                    data_width: width,
                    fraction,
                    v_rest: rest,
                    v_reset: reset,
                    v_threshold: threshold,
                    refractory_period: refractory,
                }));
            }
        }
    }
}

#[test]
fn dense_roundtrip_preserves_seed_geometry_and_current_mapping() {
    for width in [8, 16, 24, 32] {
        for (inputs, neurons, length) in [(1, 1, 64), (3, 7, 1024)] {
            assert_roundtrip(&dense(DenseParams {
                n_inputs: inputs,
                n_neurons: neurons,
                data_width: width,
                stream_length: length,
                input_seed_base: 0x1234,
                weight_seed_base: 0x5678,
                y_min: -100,
                y_max: 300,
            }));
        }
    }
}

#[test]
fn dcls_roundtrip_preserves_ordered_repeated_taps_and_pointer_geometry() {
    for offsets in [vec![0], vec![7, 0, 7], vec![63, 2, 32, 1]] {
        assert_roundtrip(&dcls(DclsParams {
            n_taps: offsets.len(),
            data_width: 16,
            fraction: 8,
            delay_depth: 64,
            ptr_width: 6,
            tap_offsets: offsets,
        }));
    }
    let graph = dcls(DclsParams {
        n_taps: 0,
        tap_offsets: vec![],
        ..DclsParams::default()
    });
    assert_eq!(graph, parser::parse(&printer::print(&graph)).unwrap());
}

#[test]
fn legacy_lif_and_dense_text_keep_historical_defaults() {
    for graph in [lif(LifParams::default()), dense(DenseParams::default())] {
        let text = printer::print(&graph)
            .replace(", vr=0, reset=0", "")
            .replace(
                ", dw=16, input_seed=44257, weight_seed=48879, y_min=0, y_max=256",
                "",
            );
        assert_eq!(graph, parser::parse(&text).unwrap());
    }
    let text = printer::print(&lif(LifParams::default()))
        .replace(", dw=16, frac=8, vr=0, reset=0, vt=256, rp=2", "");
    assert_eq!(lif(LifParams::default()), parser::parse(&text).unwrap());
    let text = printer::print(&dense(DenseParams::default())).replace(
        ", ni=3, nn=7, len=1024, dw=16, input_seed=44257, weight_seed=48879, y_min=0, y_max=256",
        "",
    );
    assert_eq!(dense(DenseParams::default()), parser::parse(&text).unwrap());
}

#[test]
fn explicit_invalid_parameters_never_fall_back_to_defaults() {
    let graphs = [
        lif(LifParams::default()),
        dense(DenseParams::default()),
        dcls(DclsParams::default()),
    ];
    for graph in graphs {
        let text = printer::print(&graph);
        let op = text.lines().find(|line| line.contains("%4 =")).unwrap();
        let attributes = op.split_once(", ").unwrap().1.split_once(" : ").unwrap().0;
        for attribute in attributes.split(", ") {
            let key = attribute.split_once('=').unwrap().0;
            let invalid = text.replace(attribute, &format!("{key}=broken"));
            let err = parser::parse(&invalid).expect_err("explicit malformed attribute must fail");
            assert_eq!(err.line, 6);
            for replacement in [
                format!("{attribute}, {attribute}"),
                format!("unknown_{attribute}"),
            ] {
                assert!(parser::parse(&text.replace(attribute, &replacement)).is_err());
            }
        }
    }
}

#[test]
fn missing_operands_invalid_unsigned_values_and_unrecoverable_taps_fail() {
    let text = printer::print(&dcls(DclsParams::default()));
    for (from, to) in [
        (", offsets=[0;1;2]", ""),
        ("offsets=[0;1;2]", "offsets=0;1;2"),
        ("offsets=[0;1;2]", "offsets=[0;-1;2]"),
        ("ptr=5", "ptr=-1"),
        ("taps=3", "taps=18446744073709551616"),
        (", weights=%1", ""),
        (", centre=%2", ""),
        (", sigma=%3", ""),
        ("%4 = sc.dcls_layer %0", "%4294967295 = sc.dcls_layer %0"),
        ("%4 = sc.dcls_layer %0", "%4 = sc.dcls_layer invalid"),
        ("offsets=[0;1;2] : fixed<16,8>", "offsets=[0;1;2]"),
        ("%4 = sc.dcls_layer %0, weights=%1, centre=%2, sigma=%3, taps=3, depth=31, dw=16, frac=8, ptr=5, offsets=[0;1;2]", "%4 = sc.dcls_layer %0"),
        ("%4 = sc.dcls_layer", "%bad = sc.dcls_layer"),
        ("weights=%1", "weights"),
    ] {
        assert!(
            parser::parse(&text.replace(from, to)).is_err(),
            "accepted {from} -> {to}"
        );
    }
    let text = printer::print(&lif(LifParams::default()));
    for field in [", leak=%1", ", gain=%2", ", noise=%3"] {
        assert!(parser::parse(&text.replace(field, "")).is_err());
    }
    let text = printer::print(&dense(DenseParams::default()));
    for field in [", weights=%1", ", leak=%2", ", gain=%3"] {
        assert!(parser::parse(&text.replace(field, "")).is_err());
    }
}
