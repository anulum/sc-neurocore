//! Text-format printer for SC IR graphs.
//!
//! # Format
//!
//! ```text
//! sc.graph @module_name {
//!   %0 = sc.input "x_in" : rate
//!   %1 = sc.constant 0.5 : rate
//!   %2 = sc.encode %0, length=1024, seed=0xACE1 : bitstream<1024>
//!   %3 = sc.encode %1, length=1024, seed=0xBEEF : bitstream<1024>
//!   %4 = sc.and %2, %3 : bitstream<1024>
//!   %5 = sc.popcount %4 : u64
//!   sc.output "result" %5
//! }
//! ```

use crate::ir::graph::*;

/// Print a graph to its text representation.
pub fn print(graph: &ScGraph) -> String {
    let mut out = String::new();
    out.push_str(&format!("sc.graph @{} {{\n", graph.name));

    for op in &graph.ops {
        out.push_str("  ");
        match op {
            ScOp::Input { id, name, ty } => {
                out.push_str(&format!("{} = sc.input \"{}\" : {}\n", id, name, ty));
            }
            ScOp::Output { name, source, .. } => {
                out.push_str(&format!("sc.output \"{}\" {}\n", name, source));
            }
            ScOp::Constant { id, value, ty } => {
                let val_str = match value {
                    ScConst::F64(v) => format!("{v}"),
                    ScConst::I64(v) => format!("{v}"),
                    ScConst::U64(v) => format!("{v}"),
                    ScConst::F64Vec(v) => format!(
                        "[{}]",
                        v.iter()
                            .map(|x| format!("{x}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    ScConst::I64Vec(v) => format!(
                        "[{}]",
                        v.iter()
                            .map(|x| format!("{x}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                };
                out.push_str(&format!("{} = sc.constant {} : {}\n", id, val_str, ty));
            }
            ScOp::Encode {
                id,
                prob,
                length,
                seed,
            } => {
                out.push_str(&format!(
                    "{} = sc.encode {}, length={}, seed=0x{:04X} : bitstream<{}>\n",
                    id, prob, length, seed, length
                ));
            }
            ScOp::BitwiseAnd { id, lhs, rhs } => {
                out.push_str(&format!("{} = sc.and {}, {} : bitstream\n", id, lhs, rhs));
            }
            ScOp::Popcount { id, input } => {
                out.push_str(&format!("{} = sc.popcount {} : u64\n", id, input));
            }
            ScOp::LifStep {
                id,
                current,
                leak,
                gain,
                noise,
                params,
            } => {
                out.push_str(&format!(
                    "{} = sc.lif_step {}, leak={}, gain={}, noise={}, \
                     dw={}, frac={}, vt={}, rp={} : (bool, fixed<{},{}>)\n",
                    id,
                    current,
                    leak,
                    gain,
                    noise,
                    params.data_width,
                    params.fraction,
                    params.v_threshold,
                    params.refractory_period,
                    params.data_width,
                    params.fraction
                ));
            }
            ScOp::DenseForward {
                id,
                inputs,
                weights,
                leak,
                gain,
                params,
            } => {
                out.push_str(&format!(
                    "{} = sc.dense_forward {}, weights={}, leak={}, gain={}, \
                     ni={}, nn={}, len={} : vec<bool,{}>\n",
                    id,
                    inputs,
                    weights,
                    leak,
                    gain,
                    params.n_inputs,
                    params.n_neurons,
                    params.stream_length,
                    params.n_neurons
                ));
            }
            ScOp::Scale { id, input, factor } => {
                out.push_str(&format!(
                    "{} = sc.scale {}, factor={} : rate\n",
                    id, input, factor
                ));
            }
            ScOp::Offset { id, input, offset } => {
                out.push_str(&format!(
                    "{} = sc.offset {}, offset={} : rate\n",
                    id, input, offset
                ));
            }
            ScOp::DivConst { id, input, divisor } => {
                out.push_str(&format!(
                    "{} = sc.div_const {}, divisor={} : u64\n",
                    id, input, divisor
                ));
            }
        }
    }

    out.push_str("}\n");
    out
}
