// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compound circuits against independent integer trajectories

//! Stateful compound neuron RTL trajectories.

use super::{ir_stream, observation, random, signed};
use sc_neurocore_engine::ir::{builder::ScGraphBuilder, graph::*, word::WordOp};

/// Four dynamic operands, independent state wrapping, resets and refractory clocks.
#[test]
fn lif_runtime_matches_integer_state_transitions() {
    for (width, fraction, rest, reset, threshold, period) in [
        (1, 0, 0, 0, 0, 0),
        (16, 8, 0, 0, 256, 2),
        (12, 4, -30, -70, 35, 3),
        (8, 0, -40, -50, 80, 0),
        (64, 32, -100, -50, 1_i64 << 40, 1),
    ] {
        let mut b = ScGraphBuilder::new("lif_runtime");
        let ty = ScType::FixedPoint {
            width,
            frac: fraction,
        };
        let current = b.input("current", ty.clone());
        let leak = b.input("leak", ty.clone());
        let gain = b.input("gain", ty.clone());
        let noise = b.input("noise", ty);
        let output = b.lif_step(
            current,
            leak,
            gain,
            noise,
            LifParams {
                data_width: width,
                fraction,
                v_rest: rest,
                v_reset: reset,
                v_threshold: threshold,
                refractory_period: period,
            },
        );
        b.output("result", output);
        let mut bench = format!("module tb; reg clk=0,rst_n=1; reg [{}:0] current,leak,gain,noise; wire [{}:0] result; lif_runtime dut(.*); initial begin\n",width-1,width-1);
        let mut expected = Vec::new();
        let (mut membrane, mut refractory, mut rng) = (i128::from(rest), 0, 47_u128);
        for step in 0..96 {
            let active = ![0, 7, 31, 62].contains(&step);
            let mut raw = [0_u128; 4];
            for value in &mut raw {
                *value = random(&mut rng, width);
            }
            if step < 12 {
                raw = [threshold.max(1) as u128, 0, 1_u128 << fraction, 0];
            }
            let [current, leak, gain, noise] = raw.map(|v| signed(v, width));
            let spike = if !active {
                membrane = i128::from(rest);
                refractory = 0;
                false
            } else if refractory > 0 {
                membrane = i128::from(rest);
                refractory -= 1;
                false
            } else {
                let leak_product = signed(
                    (i128::from(rest) - membrane).wrapping_mul(leak) as u128,
                    2 * width,
                );
                let input_product = signed(current.wrapping_mul(gain) as u128, 2 * width);
                let potential = signed(
                    (membrane
                        + signed((leak_product >> fraction) as u128, width)
                        + signed((input_product >> fraction) as u128, width)
                        + noise) as u128,
                    width,
                );
                let spike = potential >= i128::from(threshold);
                membrane = if spike { i128::from(reset) } else { potential };
                refractory = if spike { period } else { 0 };
                spike
            };
            expected.push(observation(u128::from(spike), width));
            bench.push_str(&format!("clk=0; rst_n={}; current={width}'h{:x}; leak={width}'h{:x}; gain={width}'h{:x}; noise={width}'h{:x}; #2; clk=1; #1; $display(\"OBS %h\",result); #2;\n",u8::from(active),raw[0],raw[1],raw[2],raw[3]));
        }
        bench.push_str("$finish; end endmodule\n");
        ir_stream::run(
            &format!("compound-lif-{width}-{fraction}"),
            &b.build(),
            &bench,
            &expected,
        );
    }
}

/// Score tap history independently of ring pointers, including 63 fractional bits.
#[test]
fn dcls_dynamic_taps_match_history_and_wide_integer_tents() {
    for (width, fraction, depth, offsets) in [
        (2, 1, 3, vec![0, 2]),
        (16, 8, 64, vec![7, 0, 3]),
        (16, 8, 64, vec![63, 0, 1, 31]),
        (12, 4, 7, vec![0, 6, 3]),
        (32, 16, 1, vec![0]),
        (64, 63, 7, vec![0, 6]),
        (64, 8, 4, vec![0, 3]),
    ] {
        let mut b = ScGraphBuilder::new("dcls_runtime");
        let ty = ScType::FixedPoint {
            width,
            frac: fraction,
        };
        let spike = b.input("spike", ScType::Bool);
        let weights = b.input(
            "weights",
            ScType::Vec {
                element: Box::new(if width == 16 && offsets.len() == 3 {
                    ScType::SInt { width }
                } else {
                    ty.clone()
                }),
                count: offsets.len(),
            },
        );
        let centre = b.input("centre", ty.clone());
        let sigma = b.input("sigma", ty);
        let output = b.dcls_layer(
            spike,
            weights,
            centre,
            sigma,
            DclsParams {
                n_taps: offsets.len(),
                delay_depth: depth,
                data_width: width,
                fraction,
                ptr_width: (usize::BITS - (depth - 1).leading_zeros()).max(1),
                tap_offsets: offsets.clone(),
            },
        );
        b.output("result", output);
        let mut bench = format!("module tb; reg clk=0,rst_n=1,spike; reg [{}:0] weights; reg [{}:0] centre,sigma; wire [{}:0] result; dcls_runtime dut(.*); initial begin\n",width as usize*offsets.len()-1,width-1,width-1);
        let mut history = vec![false; depth];
        let mut rng = 144_u128;
        let mut expected = Vec::new();
        let limit = (1_i128 << (width - 1)) - 1;
        for step in 0..160 {
            let active = ![0, 73, 147].contains(&step);
            let spike = random(&mut rng, 1) != 0;
            if active {
                history.rotate_right(1);
                history[0] = spike;
            } else {
                history.fill(false);
            }
            let raw: Vec<_> = offsets.iter().map(|_| random(&mut rng, width)).collect();
            let centre = if step < 120 {
                [
                    -((1_i128) << fraction),
                    0,
                    ((offsets.len() - 1) as i128) << fraction,
                ][step % 3]
            } else {
                random(&mut rng, width) as i128
            };
            let centre = signed(centre as u128, width);
            let unit = (1_i128 << fraction).min(limit);
            let sigma = [0, -1, 1, unit, (2 * unit - 1).min(limit)][step % 5];
            let mut sum = 0_i128;
            if sigma > 0 {
                for (tap, offset) in offsets.iter().enumerate() {
                    let selected = if *offset == 0 {
                        spike
                    } else {
                        history[*offset as usize - 1]
                    };
                    let distance = (((tap as i128) << fraction) - centre).abs();
                    if selected && distance < sigma {
                        let gate =
                            (((sigma - distance) << fraction) / sigma).min(1_i128 << fraction);
                        sum += signed(raw[tap], width) * gate;
                    }
                }
            }
            let result = (sum >> fraction).clamp(-(1_i128 << (width - 1)), limit);
            expected.push(observation(result as u128, width));
            let packed = raw
                .iter()
                .rev()
                .map(|v| format!("{v:0digits$b}", digits = width as usize))
                .collect::<String>();
            let mask = u128::MAX >> (128 - width);
            bench.push_str(&format!("clk=0; rst_n={}; spike={}; weights={}'b{packed}; centre={width}'h{:x}; sigma={width}'h{:x}; #2; clk=1; #1; $display(\"OBS %h\",result); #2;\n",u8::from(active),u8::from(spike),width as usize*offsets.len(),centre as u128&mask,sigma as u128&mask));
        }
        assert!(expected.iter().any(|v| *v != observation(0, width)));
        bench.push_str("$finish; end endmodule\n");
        ir_stream::run(
            &format!("compound-dcls-{width}-{fraction}-{depth}-{}", offsets.len()),
            &b.build(),
            &bench,
            &expected,
        );
    }
}

/// A registered spike fans out into arithmetic and another clocked operation.
#[test]
fn lif_boolean_fanout_preserves_state_and_external_port_padding() {
    let mut b = ScGraphBuilder::new("lif_composition");
    let input = b.input("current", ScType::FixedPoint { width: 16, frac: 8 });
    let zero = b.constant(ScConst::I64(0), ScType::Rate);
    let gain = b.constant(ScConst::I64(256), ScType::Rate);
    let spike = b.lif_step(
        input,
        zero,
        gain,
        zero,
        LifParams {
            refractory_period: 1,
            ..LifParams::default()
        },
    );
    let scaled = b.scale(spike, 2.0);
    let word = b.word(WordOp::Resize {
        input: spike,
        ty: ScType::UInt { width: 16 },
    });
    let delayed = b.word(WordOp::Delay {
        input: word,
        ty: ScType::UInt { width: 16 },
    });
    let weights = b.constant(
        ScConst::I64Vec(vec![256, 256]),
        ScType::Vec {
            element: Box::new(ScType::Rate),
            count: 2,
        },
    );
    let sigma = b.constant(ScConst::I64(512), ScType::Rate);
    let synapse = b.dcls_layer(
        spike,
        weights,
        zero,
        sigma,
        DclsParams {
            n_taps: 2,
            delay_depth: 4,
            ptr_width: 2,
            tap_offsets: vec![0, 1],
            ..DclsParams::default()
        },
    );
    b.output("spike", spike);
    b.output("scaled", scaled);
    b.output("delayed", delayed);
    b.output("synapse", synapse);
    let mut bench = String::from("module tb; reg clk=0,rst_n=1; reg [15:0] current; wire [15:0] spike,delayed,synapse; wire [63:0] scaled; lif_composition dut(.*); initial begin\n");
    let (mut membrane, mut refractory, mut previous) = (0_i32, 0, false);
    let mut expected = Vec::new();
    for step in 0..48 {
        let active = ![0, 19, 35].contains(&step);
        let current = [128, 256, 0, -128, 768][step % 5];
        let delayed = active && previous;
        let spike = if !active {
            membrane = 0;
            refractory = 0;
            false
        } else if refractory > 0 {
            membrane = 0;
            refractory -= 1;
            false
        } else {
            membrane = i32::from((membrane + current) as i16);
            let spike = membrane >= 256;
            if spike {
                membrane = 0;
                refractory = 1;
            }
            spike
        };
        previous = spike;
        expected.push(format!(
            "OBS {:04x} {:016x} {:04x} {:04x}",
            u8::from(spike),
            u8::from(spike) * 2,
            u8::from(delayed),
            u16::from(spike) * 256 + u16::from(delayed) * 128
        ));
        bench.push_str(&format!("clk=0; rst_n={}; current=16'h{:x}; #2; clk=1; #1; $display(\"OBS %h %h %h %h\",spike,scaled,delayed,synapse); #2;\n",u8::from(active),current as u16));
    }
    bench.push_str("$finish; end endmodule\n");
    ir_stream::run("compound-lif-composition", &b.build(), &bench, &expected);
}

/// Dynamic row-major weights feed old encoder samples into all neurons and a delayed count.
#[test]
fn dense_runtime_composes_with_reduction_and_registered_count() {
    for (dw, pw, fraction, ni, nn, min, max, input_seed, weight_seed) in [
        (16, 16, 8, 2, 2, 0, 256, 44257_u16, 48879_u16),
        (10, 12, 4, 3, 2, 511, -512, 123, 577),
        (32, 24, 16, 1, 1, -128, 512, 61111, 39919),
        (64, 64, 63, 2, 2, i64::MIN, i64::MAX, 54577, 44419),
    ] {
        let mut b = ScGraphBuilder::new("dense_runtime");
        let ty = ScType::FixedPoint {
            width: pw,
            frac: fraction,
        };
        let input = b.input(
            "value",
            ScType::Vec {
                element: Box::new(ty.clone()),
                count: ni,
            },
        );
        let weights = b.input(
            "weights",
            ScType::Vec {
                element: Box::new(ty),
                count: ni * nn,
            },
        );
        let leak = b.input("leak", ScType::FixedPoint { width: dw, frac: 8 });
        let gain = b.input("gain", ScType::FixedPoint { width: dw, frac: 8 });
        let spikes = b.dense_forward(
            input,
            weights,
            leak,
            gain,
            DenseParams {
                n_inputs: ni,
                n_neurons: nn,
                data_width: dw,
                stream_length: 2,
                input_seed_base: input_seed,
                weight_seed_base: weight_seed,
                y_min: min,
                y_max: max,
            },
        );
        let count = b.reduce(spikes, ReduceMode::Sum);
        let delay = b.word(WordOp::Delay {
            input: count,
            ty: ScType::UInt { width: 64 },
        });
        b.output("spikes", spikes);
        b.output("count", count);
        b.output("delayed", delay);
        let seeds: Vec<_> = (0..ni)
            .map(|i| input_seed.wrapping_add(7 * i as u16))
            .chain((0..ni * nn).map(|i| weight_seed.wrapping_add(13 * i as u16)))
            .collect();
        let mut states = seeds.clone();
        let mut bits = vec![false; states.len()];
        let mut membrane = vec![0_i128; nn];
        let mut previous = 0_u32;
        let mut rng = 199_u128;
        let mut expected = Vec::new();
        let mut bench=format!("module tb; reg clk=0,rst_n=1; reg [{}:0] value; reg [{}:0] weights; reg [{}:0] leak,gain; wire [{}:0] spikes; wire [63:0] count,delayed; dense_runtime dut(.*); initial begin\n",pw as usize*ni-1,pw as usize*ni*nn-1,dw-1,nn-1);
        for step in 0..320 {
            let active = ![0, 37, 157, 257].contains(&step);
            let mut raw: Vec<_> = (0..states.len()).map(|_| random(&mut rng, pw)).collect();
            let mut leak = random(&mut rng, dw);
            let mut gain = random(&mut rng, dw);
            if step < 16 {
                let unit = (1_u128 << fraction).min((1_u128 << (pw - 1)) - 1);
                raw.fill(unit);
                raw[ni..2 * ni].fill(0);
                leak = 0;
                gain = 256;
            }
            let delayed = if active { previous } else { 0 };
            let mut output = 0_u128;
            if !active {
                states.clone_from(&seeds);
                bits.fill(false);
                membrane.fill(0);
            } else {
                for neuron in 0..nn {
                    let count = (0..ni)
                        .filter(|i| bits[*i] && bits[ni + neuron * ni + *i])
                        .count();
                    let current = i128::from(min)
                        + (i128::from(max) - i128::from(min)) * count as i128 / ni as i128;
                    let leaking = signed(
                        (-membrane[neuron]).wrapping_mul(signed(leak, dw)) as u128,
                        2 * dw,
                    );
                    let driven = signed(current.wrapping_mul(signed(gain, dw)) as u128, 2 * dw);
                    let potential = signed(
                        (membrane[neuron]
                            + signed((leaking >> 8) as u128, dw)
                            + signed((driven >> 8) as u128, dw)) as u128,
                        dw,
                    );
                    let spike = potential >= 256;
                    membrane[neuron] = if spike { 0 } else { potential };
                    output |= u128::from(spike) << neuron;
                }
                for (i, probability) in raw.iter().enumerate() {
                    let probability = signed(*probability, pw);
                    let threshold = if probability <= 0 {
                        0
                    } else {
                        (probability * 65535 / (1_i128 << fraction)).min(65535)
                    };
                    bits[i] = i128::from(states[i]) < threshold;
                    let feedback = ((states[i] >> 15)
                        ^ (states[i] >> 13)
                        ^ (states[i] >> 12)
                        ^ (states[i] >> 10))
                        & 1;
                    states[i] = (states[i] << 1) | feedback;
                }
            }
            previous = output.count_ones();
            expected.push(format!(
                "{} {:016x} {delayed:016x}",
                observation(output, nn as u32),
                previous
            ));
            let packed = |values: &[u128]| {
                values
                    .iter()
                    .rev()
                    .map(|v| format!("{v:0digits$x}", digits = pw.div_ceil(4) as usize))
                    .collect::<String>()
            };
            bench.push_str(&format!("clk=0; rst_n={}; value={}'h{}; weights={}'h{}; leak={dw}'h{leak:x}; gain={dw}'h{gain:x}; #2; clk=1; #1; $display(\"OBS %h %h %h\",spikes,count,delayed); #2;\n",u8::from(active),pw as usize*ni,packed(&raw[..ni]),pw as usize*ni*nn,packed(&raw[ni..])));
        }
        assert!(expected.iter().any(|v| !v.starts_with("OBS 0 ")));
        bench.push_str("$finish; end endmodule\n");
        ir_stream::run(
            &format!("compound-dense-{dw}-{pw}-{fraction}"),
            &b.build(),
            &bench,
            &expected,
        );
    }
}
