// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compound circuits against independent integer trajectories

//! Scalar and packed reduction RTL arithmetic.

use super::{ir_stream, observation, random, signed};
use sc_neurocore_engine::ir::{builder::ScGraphBuilder, graph::*};

/// Signedness and lane widths survive arithmetic, including unsigned buses above 64 bits.
#[test]
fn arithmetic_uses_every_declared_lane_and_binary_point() {
    for (width, lanes, is_signed, operation, coefficient) in [
        (16, 1, true, "scale", -0.375),
        (16, 1, true, "offset", -0.5),
        (80, 1, false, "scale", -1.125),
        (80, 1, false, "offset", 0.5),
        (12, 4, true, "sum", 0.0),
        (12, 4, true, "max", 0.0),
        (80, 3, false, "max", 0.0),
        (16, 1, true, "sum", 0.0),
    ] {
        let element = if is_signed {
            ScType::SInt { width }
        } else {
            ScType::UInt { width }
        };
        let ty = if lanes == 1 {
            element
        } else {
            ScType::Vec {
                element: Box::new(element),
                count: lanes,
            }
        };
        let mut b = ScGraphBuilder::new("arithmetic_runtime");
        let input = b.input("value", ty);
        let output = match operation {
            "scale" => b.scale(input, coefficient),
            "offset" => b.offset(input, coefficient),
            "sum" => b.reduce(input, ReduceMode::Sum),
            "max" => b.reduce(input, ReduceMode::Max),
            _ => unreachable!(),
        };
        b.output("result", output);
        let mut bench = format!("module tb; reg clk=0,rst_n=1; reg [{}:0] value; wire [63:0] result; arithmetic_runtime dut(.*); initial begin\n", width as usize*lanes-1);
        let mut expected = Vec::new();
        let mut rng = 83_u128;
        for step in 0..52 {
            let mask = u128::MAX >> (128 - width);
            let raw: Vec<_> = (0..lanes)
                .map(|i| match step {
                    0 => 0,
                    1 => mask,
                    2 => 1_u128 << (width - 1),
                    3 => mask - i as u128,
                    _ => random(&mut rng, width),
                })
                .collect();
            let values: Vec<_> = raw
                .iter()
                .map(|v| {
                    if is_signed {
                        signed(*v, width)
                    } else {
                        *v as i128
                    }
                })
                .collect();
            let coefficient = (coefficient * 256.0) as i128;
            let result = match operation {
                "scale" => (values[0] * coefficient) >> 8,
                "offset" => values[0] + coefficient,
                "sum" => values.iter().sum(),
                "max" => *values.iter().max().unwrap(),
                _ => unreachable!(),
            };
            expected.push(observation(result as u128, 64));
            let packed = raw
                .iter()
                .rev()
                .map(|v| format!("{v:0digits$x}", digits = width.div_ceil(4) as usize))
                .collect::<String>();
            bench.push_str(&format!(
                "value={}'h{packed}; #1; $display(\"OBS %h\",result);\n",
                width as usize * lanes
            ));
        }
        bench.push_str("$finish; end endmodule\n");
        ir_stream::run(
            &format!("compound-{operation}-{width}-{lanes}"),
            &b.build(),
            &bench,
            &expected,
        );
    }
}
