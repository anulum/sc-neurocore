// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Signed scalar and packed reduction RTL

//! Match the numeric lane contract used by the typed CIRCT backend.

use super::value_to_wire;
use crate::ir::{
    arithmetic_contract::{coefficient, lanes},
    graph::*,
};

/// Emit explicit extension or truncation without relying on expression context.
fn resize(wire: &str, width: usize, target: usize, signed: bool) -> String {
    if target < width {
        format!("{wire}[{}:0]", target - 1)
    } else if target > width {
        let bit = if signed {
            format!("{wire}[{}]", width - 1)
        } else {
            "1'b0".into()
        };
        format!("{{{{{}{{{bit}}}}}, {wire}}}", target - width)
    } else {
        wire.into()
    }
}

/// Emit modular raw-word arithmetic, preserving the input binary point.
pub(super) fn emit(text: &mut String, graph: &ScGraph, op: &ScOp) -> Result<bool, String> {
    let (id, input) = match op {
        ScOp::Scale { id, input, .. }
        | ScOp::Offset { id, input, .. }
        | ScOp::Reduce { id, input, .. } => (*id, *input),
        _ => return Ok(false),
    };
    let layout = lanes(graph, input)?;
    let wire = value_to_wire(graph, input);
    match op {
        ScOp::Scale { factor, .. } | ScOp::Offset { offset: factor, .. } => {
            if layout.count != 1 {
                return Err("Scale and Offset require a numeric scalar".into());
            }
            let scale = matches!(op, ScOp::Scale { .. });
            let width = if scale { 128 } else { 64 };
            let raw = coefficient(*factor)?;
            let literal = if raw < 0 {
                format!("-{width}'sd{}", -i128::from(raw))
            } else {
                format!("{width}'sd{raw}")
            };
            let input = resize(&wire, layout.width, width, layout.signed);
            let operator = if scale { "*" } else { "+" };
            text.push_str(&format!(
                "    wire signed [{}:0] v{}_arithmetic = $signed({input}) {operator} {literal};\n",
                width - 1,
                id.0
            ));
            let result = if scale {
                format!("v{}_arithmetic[71:8]", id.0)
            } else {
                format!("v{}_arithmetic", id.0)
            };
            text.push_str(&format!("    assign v{} = {result};\n", id.0));
        }
        ScOp::Reduce { mode, .. } => {
            let width = if *mode == ReduceMode::Max {
                layout.width.max(64)
            } else {
                64
            };
            let sign = if layout.signed { "signed " } else { "" };
            for i in 0..layout.count {
                let lane = format!("v{}_lane_{i}", id.0);
                let selection = if layout.count == 1 {
                    wire.clone()
                } else {
                    format!("{wire}[{} +: {}]", i * layout.width, layout.width)
                };
                text.push_str(&format!(
                    "    wire [{}:0] {lane} = {selection};\n",
                    layout.width - 1
                ));
                let resized = resize(&lane, layout.width, width, layout.signed);
                text.push_str(&format!(
                    "    wire {sign}[{}:0] v{}_reduce_{i} = {resized};\n",
                    width - 1,
                    id.0
                ));
                if i > 0 {
                    let previous = if i == 1 {
                        format!("v{}_reduce_0", id.0)
                    } else {
                        format!("v{}_acc_{}", id.0, i - 1)
                    };
                    let current = format!("v{}_reduce_{i}", id.0);
                    let rhs = if *mode == ReduceMode::Sum {
                        format!("{previous} + {current}")
                    } else {
                        format!("({previous} > {current}) ? {previous} : {current}")
                    };
                    text.push_str(&format!(
                        "    wire {sign}[{}:0] v{}_acc_{i} = {rhs};\n",
                        width - 1,
                        id.0
                    ));
                }
            }
            let result = if layout.count == 1 {
                format!("v{}_reduce_0", id.0)
            } else {
                format!("v{}_acc_{}", id.0, layout.count - 1)
            };
            text.push_str(&format!("    assign v{} = {result};\n", id.0));
        }
        _ => unreachable!(),
    }
    Ok(true)
}
