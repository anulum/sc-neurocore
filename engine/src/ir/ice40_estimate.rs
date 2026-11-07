// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — iCE40 UP5K calibrated static estimate

//! Calibrated static estimate for the iCE40 UltraPlus UP5K.
//!
//! The estimate is an upper bound measured with Yosys and nextpnr-ice40 (see
//! [`crate::ir::ice40_calibration::TOOLCHAIN`]). It exists only for a graph
//! that is exactly one calibrated structural block — a dense layer, a LIF
//! neuron or a synapse stream — inside that block's measured domain. Any other
//! graph gets no number, only the reason, and synthesis with place and route
//! decides.

use crate::ir::graph::{DenseParams, LifParams, ScConst, ScGraph, ScOp, ScType, ValueId};
use crate::ir::ice40_calibration as cal;

/// iCE40 UP5K logic cells (Lattice FPGA-DS-02008).
pub const LOGIC_CELL_BUDGET: u32 = 5_280;
/// iCE40 UP5K 4 Kbit embedded block RAMs.
pub const RAM_BLOCK_BUDGET: u32 = 30;
/// iCE40 UP5K SB_MAC16 DSP blocks.
pub const DSP_BUDGET: u32 = 8;
/// Device part of the target.
pub const DEVICE_PART: &str = "iCE40UP5K-SG48";
/// Target name, as accepted by the Python binding.
pub const TARGET_NAME: &str = "ice40:UP5K";

/// The static estimate of one graph, or why there is none.
#[derive(Debug, Clone, PartialEq)]
pub enum Ice40Estimate {
    /// Upper bounds for a calibrated block inside its domain.
    Calibrated {
        /// Which calibrated family the graph is: `dense`, `lif` or `synapse`.
        block: &'static str,
        /// Logic-cell upper bound of the block.
        logic_cells: u32,
        /// Critical-path upper bound in nanoseconds.
        critical_path_ns: f64,
    },
    /// No static estimate; the reason names what the calibration does not cover.
    Unavailable {
        /// What the calibration does not cover in this graph.
        reason: String,
    },
}

/// Report of a graph emitted for the iCE40 UP5K.
#[derive(Debug, Clone, PartialEq)]
pub struct Ice40Report {
    /// Target clock in MHz.
    pub clock_mhz: u32,
    /// The calibrated estimate, or why there is none.
    pub estimate: Ice40Estimate,
}

impl Ice40Report {
    /// Clock period in nanoseconds.
    pub fn clock_period_ns(&self) -> f64 {
        1_000.0 / f64::from(self.clock_mhz)
    }

    /// Whether the logic-cell bound fits the device; `None` without an estimate.
    pub fn fits_logic_cells(&self) -> Option<bool> {
        match &self.estimate {
            Ice40Estimate::Calibrated { logic_cells, .. } => {
                Some(*logic_cells <= LOGIC_CELL_BUDGET)
            }
            Ice40Estimate::Unavailable { .. } => None,
        }
    }

    /// Whether the critical-path bound fits the clock period; `None` without an estimate.
    pub fn meets_timing(&self) -> Option<bool> {
        match &self.estimate {
            Ice40Estimate::Calibrated {
                critical_path_ns, ..
            } => Some(*critical_path_ns <= self.clock_period_ns()),
            Ice40Estimate::Unavailable { .. } => None,
        }
    }

    /// Header comment the emitted SystemVerilog carries.
    pub fn header_comment(&self) -> String {
        format!(
            "// Target: iCE40 UltraPlus UP5K, part={DEVICE_PART}, clock={} MHz\n\n",
            self.clock_mhz
        )
    }
}

/// Build the iCE40 UP5K report of a graph at a clock.
///
/// # Errors
/// A zero clock has no period to judge and is refused.
pub fn report(graph: &ScGraph, clock_mhz: u32) -> Result<Ice40Report, String> {
    if clock_mhz == 0 {
        return Err("clock_mhz must be positive".to_string());
    }
    Ok(Ice40Report {
        clock_mhz,
        estimate: estimate(graph),
    })
}

/// Reason given when a block's operands or parameters differ from the measured designs.
pub const OUTSIDE_CALIBRATED_FORM: &str =
    "operand source, type or parameter outside the calibration";

/// Dense leak constant of every calibration design.
pub const CALIBRATED_DENSE_LEAK: i64 = 16;
/// Dense gain constant of every calibration design.
pub const CALIBRATED_DENSE_GAIN: i64 = 256;
/// Dense parameters of every calibration design, apart from the geometry.
pub const CALIBRATED_DENSE_STREAM: (usize, u16, u16, i64, i64) = (7, 4660, 4661, 0, 300);
/// LIF parameters of every calibration design, apart from the width:
/// fraction, rest, reset, threshold and refractory period.
pub const CALIBRATED_LIF: (u32, i64, i64, i64, u32) = (8, 0, 0, 256, 7);
/// Encoder seeds of every calibration synapse stream.
pub const CALIBRATED_SYNAPSE_SEEDS: (u16, u16) = (0xACE1, 0xBEEF);

fn defining(graph: &ScGraph, id: ValueId) -> Option<&ScOp> {
    graph.ops.iter().find(|op| op.result_id() == id)
}

fn is_input(graph: &ScGraph, id: ValueId, expected: &ScType) -> bool {
    matches!(defining(graph, id), Some(ScOp::Input { ty, .. }) if ty == expected)
}

fn fixed(width: u32) -> ScType {
    ScType::FixedPoint { width, frac: 8 }
}

fn rate_vec(count: usize) -> ScType {
    ScType::Vec {
        element: Box::new(ScType::Rate),
        count,
    }
}

fn is_fixed_constant(graph: &ScGraph, id: ValueId, width: u32, expected: i64) -> bool {
    matches!(
        defining(graph, id),
        Some(ScOp::Constant { value: ScConst::I64(v), ty, .. }) if *v == expected && *ty == fixed(width)
    )
}

/// The measured dense form: input port of rates, constant rate weights, the
/// calibration's leak and gain constants and its stream parameters.
fn dense_in_calibrated_form(
    graph: &ScGraph,
    [inputs, weights, leak, gain]: [ValueId; 4],
    p: &DenseParams,
) -> bool {
    let weights_constant = matches!(
        defining(graph, weights),
        Some(ScOp::Constant { value: ScConst::F64Vec(_), ty, .. })
            if *ty == rate_vec(p.n_inputs * p.n_neurons)
    );
    is_input(graph, inputs, &rate_vec(p.n_inputs))
        && weights_constant
        && is_fixed_constant(graph, leak, p.data_width, CALIBRATED_DENSE_LEAK)
        && is_fixed_constant(graph, gain, p.data_width, CALIBRATED_DENSE_GAIN)
        && (
            p.stream_length,
            p.input_seed_base,
            p.weight_seed_base,
            p.y_min,
            p.y_max,
        ) == CALIBRATED_DENSE_STREAM
}

/// The measured LIF form: four input ports of the neuron's word and its parameters.
fn lif_in_calibrated_form(graph: &ScGraph, operands: [ValueId; 4], p: &LifParams) -> bool {
    operands
        .iter()
        .all(|id| is_input(graph, *id, &fixed(p.data_width)))
        && (
            p.fraction,
            p.v_rest,
            p.v_reset,
            p.v_threshold,
            p.refractory_period,
        ) == CALIBRATED_LIF
}

/// Return the calibrated estimate of a graph, or why it has none.
///
/// A block has a number only in the form the calibration measured: its
/// operands' sources and types and its parameter values. Synthesis follows
/// constants, so any other form gets [`OUTSIDE_CALIBRATED_FORM`].
pub fn estimate(graph: &ScGraph) -> Ice40Estimate {
    let mut blocks = Vec::new();
    let mut calibrated_form = true;
    let (mut encodes, mut and_ops, mut decodes) = (Vec::new(), 0_usize, 0_usize);
    for op in &graph.ops {
        match op {
            ScOp::Input { .. } | ScOp::Output { .. } | ScOp::Constant { .. } => {}
            ScOp::DenseForward {
                inputs,
                weights,
                leak,
                gain,
                params,
                ..
            } => {
                calibrated_form &=
                    dense_in_calibrated_form(graph, [*inputs, *weights, *leak, *gain], params);
                blocks.push(Block::Dense {
                    n_inputs: params.n_inputs,
                    n_neurons: params.n_neurons,
                    data_width: params.data_width,
                });
            }
            ScOp::LifStep {
                current,
                leak,
                gain,
                noise,
                params,
                ..
            } => {
                calibrated_form &=
                    lif_in_calibrated_form(graph, [*current, *leak, *gain, *noise], params);
                blocks.push(Block::Lif {
                    data_width: params.data_width,
                });
            }
            ScOp::Encode {
                prob, length, seed, ..
            } => {
                calibrated_form &= is_input(graph, *prob, &ScType::Rate);
                encodes.push((*length, *seed));
            }
            ScOp::BitwiseAnd { .. } => and_ops += 1,
            ScOp::Decode { .. } => decodes += 1,
            ScOp::Word { .. } | ScOp::State { .. } | ScOp::Store { .. } => {
                return unavailable("word-level operations have no static estimate on this target")
            }
            _ => return unavailable("the graph holds an operation the calibration does not cover"),
        }
    }
    if !encodes.is_empty() || and_ops > 0 || decodes > 0 {
        let synapse =
            encodes.len() == 2 && and_ops == 1 && decodes == 1 && encodes[0].0 == encodes[1].0;
        if !synapse {
            return unavailable(
                "only a two-encoder synapse stream is calibrated among stream operations",
            );
        }
        let seeds = (encodes[0].1, encodes[1].1);
        calibrated_form &=
            seeds == CALIBRATED_SYNAPSE_SEEDS || (seeds.1, seeds.0) == CALIBRATED_SYNAPSE_SEEDS;
        blocks.push(Block::Synapse {
            stream_length: encodes[0].0,
        });
    }
    match blocks.as_slice() {
        [_] if !calibrated_form => unavailable(OUTSIDE_CALIBRATED_FORM),
        [block] => block.estimate(),
        [] => unavailable("the graph holds no calibrated block"),
        _ => unavailable("composition of blocks is not calibrated"),
    }
}

fn unavailable(reason: &str) -> Ice40Estimate {
    Ice40Estimate::Unavailable {
        reason: reason.to_string(),
    }
}

enum Block {
    Dense {
        n_inputs: usize,
        n_neurons: usize,
        data_width: u32,
    },
    Lif {
        data_width: u32,
    },
    Synapse {
        stream_length: usize,
    },
}

fn within<T: PartialOrd>(value: T, (low, high): (T, T)) -> bool {
    low <= value && value <= high
}

fn dot<const N: usize>(features: [f64; N], coefficients: [f64; N]) -> f64 {
    features.iter().zip(coefficients).map(|(f, c)| f * c).sum()
}

fn cells(value: f64) -> u32 {
    value.ceil().min(f64::from(u32::MAX)) as u32
}

impl Block {
    fn estimate(&self) -> Ice40Estimate {
        let outside = || unavailable("outside the calibrated domain");
        match *self {
            Self::Dense {
                n_inputs,
                n_neurons,
                data_width,
            } => {
                if !(within(n_inputs, cal::DENSE_INPUTS)
                    && within(n_neurons, cal::DENSE_NEURONS)
                    && within(data_width, cal::DENSE_WIDTH))
                {
                    return outside();
                }
                let (i, n, w) = (n_inputs as f64, n_neurons as f64, f64::from(data_width));
                Ice40Estimate::Calibrated {
                    block: "dense",
                    logic_cells: cells(
                        cal::DENSE_LOGIC_CELL_FACTOR
                            * dot([i, n, i * n, n * w], cal::DENSE_LOGIC_CELLS),
                    ),
                    critical_path_ns: cal::DENSE_PATH_FACTOR
                        * dot([1.0, i, i * i], cal::DENSE_PATH),
                }
            }
            Self::Lif { data_width } => {
                if !within(data_width, cal::LIF_WIDTH) {
                    return outside();
                }
                let w = f64::from(data_width);
                Ice40Estimate::Calibrated {
                    block: "lif",
                    logic_cells: cells(
                        cal::LIF_LOGIC_CELL_FACTOR * dot([w, w * w], cal::LIF_LOGIC_CELLS),
                    ),
                    critical_path_ns: cal::LIF_PATH_FACTOR * dot([1.0, w], cal::LIF_PATH),
                }
            }
            Self::Synapse { stream_length } => {
                // Only power-of-two lengths are calibrated: the decoder's
                // division becomes a shift, and other lengths measured up to
                // 1.7x larger and 5x slower.
                if !(stream_length.is_power_of_two() && within(stream_length, cal::SYNAPSE_LENGTH))
                {
                    return outside();
                }
                let log_length = (stream_length as f64).log2();
                Ice40Estimate::Calibrated {
                    block: "synapse",
                    logic_cells: cells(
                        cal::SYNAPSE_LOGIC_CELL_FACTOR
                            * dot([1.0, log_length], cal::SYNAPSE_LOGIC_CELLS),
                    ),
                    critical_path_ns: cal::SYNAPSE_PATH_FACTOR * dot([1.0], cal::SYNAPSE_PATH),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::ScGraphBuilder;
    use crate::ir::graph::{DenseParams, LifParams, ScConst, ScType};

    fn fixed(width: u32) -> ScType {
        ScType::FixedPoint { width, frac: 8 }
    }

    fn dense(n_inputs: usize, n_neurons: usize, data_width: u32) -> ScGraph {
        let mut b = ScGraphBuilder::new("dense");
        let inputs = b.input(
            "inputs",
            ScType::Vec {
                element: Box::new(ScType::Rate),
                count: n_inputs,
            },
        );
        let weights = b.constant(
            ScConst::F64Vec(vec![0.25; n_inputs * n_neurons]),
            ScType::Vec {
                element: Box::new(ScType::Rate),
                count: n_inputs * n_neurons,
            },
        );
        let leak = b.constant(ScConst::I64(16), fixed(data_width));
        let gain = b.constant(ScConst::I64(256), fixed(data_width));
        let out = b.dense_forward(
            inputs,
            weights,
            leak,
            gain,
            dense_params(n_inputs, n_neurons, data_width),
        );
        b.output("spikes", out);
        b.build()
    }

    fn dense_params(n_inputs: usize, n_neurons: usize, data_width: u32) -> DenseParams {
        let (stream_length, input_seed_base, weight_seed_base, y_min, y_max) =
            CALIBRATED_DENSE_STREAM;
        DenseParams {
            n_inputs,
            n_neurons,
            data_width,
            stream_length,
            input_seed_base,
            weight_seed_base,
            y_min,
            y_max,
        }
    }

    fn lif_params(data_width: u32) -> LifParams {
        let (fraction, v_rest, v_reset, v_threshold, refractory_period) = CALIBRATED_LIF;
        LifParams {
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        }
    }

    /// A 2x2 w16 dense layer whose weights, or leak and gain, are input ports.
    fn dense_with_ports(port_weights: bool, port_leak_gain: bool) -> ScGraph {
        let mut b = ScGraphBuilder::new("dense_ports");
        let inputs = b.input("inputs", rate_vec(2));
        let weights = if port_weights {
            b.input("weights", rate_vec(4))
        } else {
            b.constant(ScConst::F64Vec(vec![0.25; 4]), rate_vec(4))
        };
        let (leak, gain) = if port_leak_gain {
            (b.input("leak", fixed(16)), b.input("gain", fixed(16)))
        } else {
            (
                b.constant(ScConst::I64(CALIBRATED_DENSE_LEAK), fixed(16)),
                b.constant(ScConst::I64(CALIBRATED_DENSE_GAIN), fixed(16)),
            )
        };
        let out = b.dense_forward(inputs, weights, leak, gain, dense_params(2, 2, 16));
        b.output("spikes", out);
        b.build()
    }

    fn lif(data_width: u32) -> ScGraph {
        let mut b = ScGraphBuilder::new("lif");
        let ops: Vec<_> = ["current", "leak", "gain", "noise"]
            .iter()
            .map(|n| b.input(*n, fixed(data_width)))
            .collect();
        let spike = b.lif_step(ops[0], ops[1], ops[2], ops[3], lif_params(data_width));
        b.output("spike", spike);
        b.build()
    }

    fn synapse(length: usize) -> ScGraph {
        let mut b = ScGraphBuilder::new("synapse");
        let x = b.input("x", ScType::Rate);
        let w = b.input("w", ScType::Rate);
        let xs = b.encode(x, length, 0xACE1);
        let ws = b.encode(w, length, 0xBEEF);
        let product = b.bitwise_and(xs, ws);
        let rate = b.decode(product);
        b.output("rate", rate);
        b.build()
    }

    fn reason(estimate: Ice40Estimate) -> String {
        match estimate {
            Ice40Estimate::Unavailable { reason } => reason,
            other => panic!("expected no static estimate, got {other:?}"),
        }
    }

    #[test]
    fn calibrated_blocks_inside_their_domain_have_a_bound() {
        for (graph, block) in [
            (dense(4, 3, 16), "dense"),
            (lif(16), "lif"),
            (synapse(256), "synapse"),
        ] {
            match estimate(&graph) {
                Ice40Estimate::Calibrated {
                    block: found,
                    logic_cells,
                    critical_path_ns,
                } => {
                    assert_eq!(found, block);
                    assert!(logic_cells > 0 && critical_path_ns > 0.0);
                }
                other => panic!("{block}: {other:?}"),
            }
        }
    }

    #[test]
    fn the_dense_bound_grows_with_size_and_its_path_with_the_input_count() {
        let small = estimate(&dense(2, 2, 12));
        let large = estimate(&dense(12, 8, 24));
        match (small, large) {
            (
                Ice40Estimate::Calibrated {
                    logic_cells: a,
                    critical_path_ns: pa,
                    ..
                },
                Ice40Estimate::Calibrated {
                    logic_cells: b,
                    critical_path_ns: pb,
                    ..
                },
            ) => {
                assert!(a < b);
                assert!(pa < pb);
                let path = |i: f64| cal::DENSE_PATH_FACTOR * dot([1.0, i, i * i], cal::DENSE_PATH);
                assert_eq!(pa, path(2.0));
                assert_eq!(pb, path(12.0));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn each_domain_edge_is_inclusive_and_one_step_outside_has_no_estimate() {
        for graph in [
            dense(2, 2, 12),
            dense(12, 8, 24),
            lif(12),
            lif(32),
            synapse(64),
            synapse(4096),
        ] {
            assert!(matches!(estimate(&graph), Ice40Estimate::Calibrated { .. }));
        }
        for graph in [
            dense(1, 2, 16),
            dense(13, 2, 16),
            dense(4, 1, 16),
            dense(4, 9, 16),
            dense(4, 3, 11),
            dense(4, 3, 25),
            lif(10),
            synapse(32),
            synapse(8192),
            synapse(700),
        ] {
            assert_eq!(reason(estimate(&graph)), "outside the calibrated domain");
        }
    }

    #[test]
    fn word_operations_compositions_and_partial_streams_have_no_estimate() {
        let mut b = ScGraphBuilder::new("word");
        let state_init = b.constant(ScConst::I64(0), fixed(16));
        let state = b.state(state_init, fixed(16));
        b.output("state", state);
        assert_eq!(
            reason(estimate(&b.build())),
            "word-level operations have no static estimate on this target"
        );

        let mut b = ScGraphBuilder::new("two");
        let ops: Vec<_> = ["c", "l", "g", "n"]
            .iter()
            .map(|n| b.input(*n, fixed(16)))
            .collect();
        let first = b.lif_step(ops[0], ops[1], ops[2], ops[3], LifParams::default());
        let second = b.lif_step(ops[0], ops[1], ops[2], ops[3], LifParams::default());
        b.output("a", first);
        b.output("b", second);
        assert_eq!(
            reason(estimate(&b.build())),
            "composition of blocks is not calibrated"
        );

        let mut b = ScGraphBuilder::new("encode_only");
        let x = b.input("x", ScType::Rate);
        let stream = b.encode(x, 256, 1);
        let rate = b.decode(stream);
        b.output("rate", rate);
        assert_eq!(
            reason(estimate(&b.build())),
            "only a two-encoder synapse stream is calibrated among stream operations"
        );

        let empty = ScGraphBuilder::new("empty").build();
        assert_eq!(
            reason(estimate(&empty)),
            "the graph holds no calibrated block"
        );
    }

    #[test]
    fn the_report_judges_fit_and_timing_only_with_an_estimate() {
        let fast = report(&dense(4, 3, 16), 12).expect("positive clock");
        assert_eq!(fast.fits_logic_cells(), Some(true));
        assert_eq!(fast.meets_timing(), Some(true));
        let slow = report(&dense(4, 3, 16), 100).expect("positive clock");
        assert_eq!(slow.meets_timing(), Some(false));
        let large = report(&dense(12, 8, 24), 12).expect("positive clock");
        assert_eq!(large.fits_logic_cells(), Some(false));
        let none = report(&lif(10), 12).expect("positive clock");
        assert_eq!(none.fits_logic_cells(), None);
        assert_eq!(none.meets_timing(), None);
        assert_eq!(
            report(&lif(16), 0).expect_err("zero clock"),
            "clock_mhz must be positive"
        );
        assert_eq!(
            fast.header_comment(),
            "// Target: iCE40 UltraPlus UP5K, part=iCE40UP5K-SG48, clock=12 MHz\n\n"
        );
        assert_eq!(fast.clock_period_ns(), 1_000.0 / 12.0);
    }

    #[test]
    fn port_operands_and_other_constants_have_no_estimate() {
        assert!(matches!(
            estimate(&dense_with_ports(false, false)),
            Ice40Estimate::Calibrated { .. }
        ));
        for graph in [dense_with_ports(true, false), dense_with_ports(false, true)] {
            assert_eq!(reason(estimate(&graph)), OUTSIDE_CALIBRATED_FORM);
        }

        let mut b = ScGraphBuilder::new("dense_leak");
        let inputs = b.input("inputs", rate_vec(2));
        let weights = b.constant(ScConst::F64Vec(vec![0.25; 4]), rate_vec(4));
        let leak = b.constant(ScConst::I64(CALIBRATED_DENSE_LEAK + 1), fixed(16));
        let gain = b.constant(ScConst::I64(CALIBRATED_DENSE_GAIN), fixed(16));
        let out = b.dense_forward(inputs, weights, leak, gain, dense_params(2, 2, 16));
        b.output("spikes", out);
        assert_eq!(reason(estimate(&b.build())), OUTSIDE_CALIBRATED_FORM);

        let mut b = ScGraphBuilder::new("dense_y_max");
        let inputs = b.input("inputs", rate_vec(2));
        let weights = b.constant(ScConst::F64Vec(vec![0.25; 4]), rate_vec(4));
        let leak = b.constant(ScConst::I64(CALIBRATED_DENSE_LEAK), fixed(16));
        let gain = b.constant(ScConst::I64(CALIBRATED_DENSE_GAIN), fixed(16));
        let params = DenseParams {
            y_max: 301,
            ..dense_params(2, 2, 16)
        };
        let out = b.dense_forward(inputs, weights, leak, gain, params);
        b.output("spikes", out);
        assert_eq!(reason(estimate(&b.build())), OUTSIDE_CALIBRATED_FORM);
    }

    #[test]
    fn lif_and_synapse_need_their_measured_ports_and_parameters() {
        let mut b = ScGraphBuilder::new("lif_constant_noise");
        let current = b.input("current", fixed(16));
        let leak = b.input("leak", fixed(16));
        let gain = b.input("gain", fixed(16));
        let noise = b.constant(ScConst::I64(0), fixed(16));
        let spike = b.lif_step(current, leak, gain, noise, lif_params(16));
        b.output("spike", spike);
        assert_eq!(reason(estimate(&b.build())), OUTSIDE_CALIBRATED_FORM);

        let mut b = ScGraphBuilder::new("lif_refractory");
        let ops: Vec<_> = ["current", "leak", "gain", "noise"]
            .iter()
            .map(|n| b.input(*n, fixed(16)))
            .collect();
        let params = LifParams {
            refractory_period: 2,
            ..lif_params(16)
        };
        let spike = b.lif_step(ops[0], ops[1], ops[2], ops[3], params);
        b.output("spike", spike);
        assert_eq!(reason(estimate(&b.build())), OUTSIDE_CALIBRATED_FORM);

        for (constant_x, seed) in [(true, CALIBRATED_SYNAPSE_SEEDS.0), (false, 0x1234)] {
            let mut b = ScGraphBuilder::new("synapse_other");
            let x = if constant_x {
                b.constant(ScConst::F64(0.5), ScType::Rate)
            } else {
                b.input("x", ScType::Rate)
            };
            let w = b.input("w", ScType::Rate);
            let xs = b.encode(x, 256, seed);
            let ws = b.encode(w, 256, CALIBRATED_SYNAPSE_SEEDS.1);
            let product = b.bitwise_and(xs, ws);
            let rate = b.decode(product);
            b.output("rate", rate);
            assert_eq!(reason(estimate(&b.build())), OUTSIDE_CALIBRATED_FORM);
        }
    }
}
