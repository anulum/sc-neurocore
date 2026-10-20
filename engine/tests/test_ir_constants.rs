// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Declared constant words through real RTL

//! Exercise public native builders, verification, and real emitted constant buses.

#[path = "common/ir_rtl.rs"]
mod ir_rtl;
use sc_neurocore_engine::ir::{builder::ScGraphBuilder, emit_sv, graph::*, verify};
use std::path::Path;

fn check(label: &str, value: ScConst, ty: ScType, width: u32, words: &[u64]) {
    let mut b = ScGraphBuilder::new("constant_words");
    let id = b.constant(value, ty);
    b.output("value", id);
    let graph = b.build();
    verify::verify(&graph).unwrap();
    let sv = emit_sv::emit(&graph).unwrap();
    let mut displays = String::new();
    for i in 0..words.len() {
        displays.push_str(&format!(
            "$display(\"WORD %h\", value[{} +: {width}]);\n",
            i * width as usize
        ));
    }
    let tb = format!("module tb; wire [{}:0] value; constant_words dut(.clk(1'b0), .rst_n(1'b1), .value(value)); initial begin #1; {displays} $finish; end endmodule", width as usize * words.len() - 1);
    let trace = ir_rtl::simulate(
        label,
        &sv,
        &tb,
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../hdl"),
    );
    let actual: Vec<_> = trace
        .lines()
        .filter_map(|line| line.strip_prefix("WORD "))
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect();
    assert_eq!(actual, words, "{label}");
}

#[test]
fn declared_signed_unsigned_boolean_and_fixed_constants_reach_rtl_exactly() {
    for width in [1, 2, 8, 16, 24, 32, 64] {
        let min = -(1_i128 << (width - 1));
        let max = (1_i128 << (width - 1)) - 1;
        let mask = (1_u128 << width) - 1;
        for (idx, value) in [min, 0, max].into_iter().enumerate() {
            check(
                &format!("signed-{width}-{idx}"),
                ScConst::I64(value as i64),
                ScType::SInt { width },
                width,
                &[(value as u128 & mask) as u64],
            );
        }
        check(
            &format!("unsigned-{width}"),
            ScConst::U64(mask as u64),
            ScType::UInt { width },
            width,
            &[mask as u64],
        );
        let frac = width / 2;
        let values = vec![-0.375, 0.0, 0.625];
        let expected: Vec<_> = values
            .iter()
            .map(|v| ((*v * 2_f64.powi(frac as i32)).trunc() as i128 as u128 & mask) as u64)
            .collect();
        check(
            &format!("fixed-float-{width}"),
            ScConst::F64Vec(values),
            ScType::Vec {
                element: Box::new(ScType::FixedPoint { width, frac }),
                count: 3,
            },
            width,
            &expected,
        );
        check(
            &format!("fixed-integer-{width}"),
            ScConst::I64Vec(vec![min as i64, max as i64]),
            ScType::Vec {
                element: Box::new(ScType::FixedPoint { width, frac }),
                count: 2,
            },
            width,
            &[(min as u128 & mask) as u64, max as u64],
        );
    }
    check("rate", ScConst::F64(0.625), ScType::Rate, 16, &[160]);
    check("bool", ScConst::I64(1), ScType::Bool, 1, &[1]);
    check(
        "unsigned-float",
        ScConst::F64(63.75),
        ScType::UInt { width: 8 },
        8,
        &[63],
    );
    check(
        "signed-float",
        ScConst::F64(-7.75),
        ScType::SInt { width: 8 },
        8,
        &[249],
    );
}

#[test]
fn malformed_constant_values_and_types_are_rejected_before_emission() {
    let cases = [
        (ScConst::F64(f64::NAN), ScType::Rate),
        (
            ScConst::F64Vec(vec![f64::NAN]),
            ScType::Vec {
                element: Box::new(ScType::Rate),
                count: 1,
            },
        ),
        (
            ScConst::I64Vec(vec![128]),
            ScType::Vec {
                element: Box::new(ScType::SInt { width: 8 }),
                count: 1,
            },
        ),
        (
            ScConst::I64Vec(vec![0]),
            ScType::Vec {
                element: Box::new(ScType::SInt { width: 64 }),
                count: usize::MAX,
            },
        ),
        (ScConst::F64(-1.0), ScType::UInt { width: 8 }),
        (ScConst::F64(f64::INFINITY), ScType::Rate),
        (ScConst::F64(f64::NEG_INFINITY), ScType::Rate),
        (ScConst::F64(128.0), ScType::Rate),
        (ScConst::F64(-129.0), ScType::Rate),
        (ScConst::F64(2_f64.powi(63)), ScType::SInt { width: 64 }),
        (ScConst::F64(2_f64.powi(64)), ScType::UInt { width: 64 }),
        (ScConst::I64(-1), ScType::UInt { width: 8 }),
        (ScConst::I64(128), ScType::SInt { width: 8 }),
        (ScConst::U64(128), ScType::SInt { width: 8 }),
        (ScConst::U64(256), ScType::UInt { width: 8 }),
        (ScConst::I64(2), ScType::Bool),
        (ScConst::I64(1), ScType::SInt { width: 0 }),
        (ScConst::I64(1), ScType::UInt { width: 65 }),
        (ScConst::F64(0.5), ScType::FixedPoint { width: 8, frac: 8 }),
        (ScConst::F64(0.5), ScType::Bitstream { length: 8 }),
        (ScConst::F64Vec(vec![0.5]), ScType::Rate),
        (
            ScConst::I64(1),
            ScType::Vec {
                element: Box::new(ScType::Rate),
                count: 1,
            },
        ),
        (
            ScConst::F64Vec(vec![]),
            ScType::Vec {
                element: Box::new(ScType::Rate),
                count: 0,
            },
        ),
        (
            ScConst::I64Vec(vec![1]),
            ScType::Vec {
                element: Box::new(ScType::SInt { width: 8 }),
                count: 2,
            },
        ),
        (
            ScConst::I64Vec(vec![1]),
            ScType::Vec {
                element: Box::new(ScType::Vec {
                    element: Box::new(ScType::Bool),
                    count: 1,
                }),
                count: 1,
            },
        ),
    ];
    for (value, ty) in cases {
        let mut b = ScGraphBuilder::new("invalid_constant");
        b.constant(value, ty);
        let graph = b.build();
        assert!(verify::verify(&graph).is_err(), "{graph:?}");
        assert!(emit_sv::emit(&graph).is_err(), "{graph:?}");
    }
}
