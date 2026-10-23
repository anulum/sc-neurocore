// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Full-width pipeline register traces

//! Native word-delay builders, graph persistence and real clock/reset traces.

#[path = "common/ir_stream.rs"]
mod ir_stream;

use sc_neurocore_engine::ir::{builder::ScGraphBuilder, graph::ScType, sv_target::SvTarget};

fn hexadecimal(width: usize, bit: impl Fn(usize) -> bool) -> String {
    let mut digits = String::new();
    for nibble in (0..width.div_ceil(4)).rev() {
        let mut digit = 0;
        for offset in 0..4 {
            let index = nibble * 4 + offset;
            if index < width && bit(index) {
                digit |= 1 << offset;
            }
        }
        digits.push(char::from_digit(digit, 16).unwrap());
    }
    digits
}

#[test]
fn two_real_delay_stages_preserve_every_word_bit_and_asynchronous_reset() {
    for width in [1usize, 16, 64, 65, 128, 131] {
        for signed in [false, true] {
            let ty = if signed {
                ScType::SInt {
                    width: width as u32,
                }
            } else {
                ScType::UInt {
                    width: width as u32,
                }
            };
            let mut b = ScGraphBuilder::new("word_pipeline");
            let input = b.input("sample", ty.clone());
            let first = b.word_delay(input, ty.clone());
            let second = b.word_delay(first, ty);
            b.output("first", first);
            b.output("second", second);
            let zero = hexadecimal(width, |_| false);
            let ones = hexadecimal(width, |_| true);
            let mut previous = zero.clone();
            let mut older = zero.clone();
            let mut expected = vec![format!("OBS {previous} {older}")];
            let mut bench = format!(
                "module tb; reg clk=0,rst_n=1; reg [{}:0] sample=0; wire [{}:0] first,second; word_pipeline dut(.*); initial begin #1; rst_n=0; #1; $display(\"OBS %h %h\",first,second); #1; rst_n=1;\n",
                width - 1, width - 1
            );
            for step in 0..9 {
                let value = hexadecimal(width, |index| match step {
                    0 => false,
                    1 => true,
                    2 => index == width - 1,
                    3 => index == 0,
                    4 => index == 63 || index == 64,
                    _ => (index * 13 + step * 7) % 17 < 8,
                });
                bench.push_str(&format!(
                    "sample={width}'h{value}; #1; $display(\"OBS %h %h\",first,second); clk=1; #1; $display(\"OBS %h %h\",first,second); clk=0; #1;\n"
                ));
                expected.push(format!("OBS {previous} {older}"));
                older = previous;
                previous = value;
                expected.push(format!("OBS {previous} {older}"));
                if step == 4 {
                    bench.push_str(
                        "#1; rst_n=0; #1; $display(\"OBS %h %h\",first,second); #1; rst_n=1;\n",
                    );
                    previous = zero.clone();
                    older = zero.clone();
                    expected.push(format!("OBS {previous} {older}"));
                }
            }
            bench.push_str(&format!(
                "sample={width}'h{ones}; rst_n=0; #1; clk=1; #1; $display(\"OBS %h %h\",first,second); rst_n=1; #1; $display(\"OBS %h %h\",first,second); clk=0; #1; clk=1; #1; $display(\"OBS %h %h\",first,second); $finish; end endmodule"
            ));
            expected.push(format!("OBS {zero} {zero}"));
            expected.push(format!("OBS {zero} {zero}"));
            expected.push(format!("OBS {ones} {zero}"));
            ir_stream::run(
                &format!(
                    "word_delay_{}_{}",
                    if signed { "signed" } else { "unsigned" },
                    width
                ),
                &b.build(),
                &bench,
                &expected,
            );
        }
    }
}

#[test]
fn planning_counts_register_bits_without_inventing_combinational_gates() {
    for width in [1, 64, 128, 131] {
        let ty = ScType::SInt { width };
        let mut baseline = ScGraphBuilder::new("delay_baseline");
        let direct = baseline.input("sample", ty.clone());
        baseline.output("second", direct);
        let baseline = SvTarget::Generic.estimate_graph(&baseline.build());
        let mut b = ScGraphBuilder::new("delay_resources");
        let input = b.input("sample", ty.clone());
        let first = b.word_delay(input, ty.clone());
        let second = b.word_delay(first, ty);
        b.output("second", second);
        let estimate = SvTarget::Generic.estimate_graph(&b.build());
        assert_eq!(estimate.ff_estimated, baseline.ff_estimated + 2 * width);
        assert_eq!(estimate.lut_estimated, baseline.lut_estimated);
        assert_eq!(estimate.dsp_estimated, baseline.dsp_estimated);
    }
}
