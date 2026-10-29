// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Typed CIRCT scaling, offset and numeric reduction

//! Compute actual lane reductions and signed scalar arithmetic without implicit widths.

use super::{
    ops::{Net, Value},
    value_wire,
};
use crate::ir::{
    arithmetic_contract::{coefficient, lanes},
    graph::*,
};

/// Lower scalar Q8.8 coefficients or homogeneous packed reductions.
pub(super) fn emit(text: &mut String, graph: &ScGraph, op: &ScOp) -> Result<bool, String> {
    let (id, input) = match op {
        ScOp::Scale { id, input, .. }
        | ScOp::Offset { id, input, .. }
        | ScOp::Reduce { id, input, .. } => (*id, *input),
        _ => return Ok(false),
    };
    let layout = lanes(graph, input)?;
    let source = Value {
        name: value_wire(graph, input),
        width: layout.width * layout.count,
        signed: layout.signed,
    };
    let mut n = Net::new(text, id.0);
    let result = match op {
        ScOp::Scale { factor, .. } | ScOp::Offset { offset: factor, .. } => {
            if layout.count != 1 {
                return Err("Scale and Offset require a numeric scalar".into());
            }
            let scale = matches!(op, ScOp::Scale { .. });
            let width = if scale { 128 } else { 64 };
            let input = n.resize(&source, width, true);
            let constant = n.constant(coefficient(*factor)?, width, true);
            let computed = n.binary(if scale { "mul" } else { "add" }, &input, &constant);
            if scale {
                n.extract(&computed, 8, 64)
            } else {
                computed
            }
        }
        ScOp::Reduce { mode, .. } => {
            let width = if *mode == ReduceMode::Max {
                layout.width.max(64)
            } else {
                64
            };
            let mut values = Vec::new();
            for i in 0..layout.count {
                let lane = n.extract(&source, i * layout.width, layout.width);
                let lane = Value {
                    signed: layout.signed,
                    ..lane
                };
                values.push(n.resize(&lane, width, layout.signed));
            }
            let mut result = values[0].clone();
            for value in &values[1..] {
                result = if *mode == ReduceMode::Sum {
                    n.binary("add", &result, value)
                } else {
                    let greater =
                        n.compare(if layout.signed { "sgt" } else { "ugt" }, &result, value);
                    n.select(&greater, &result, value)
                };
            }
            n.resize(&result, 64, layout.signed)
        }
        _ => unreachable!(),
    };
    n.bind(&format!("%v{}", id.0), &result);
    Ok(true)
}
