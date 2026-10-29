// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Quantized graph, attention and phase kernel support

//! Keep the maintained constant-operand Q8.16 hardware contract explicit.

mod attention;
mod graph;
mod kuramoto;

use super::ops::{Net, Value};
use crate::ir::graph::{ScGraph, ScOp, ValueId};

/// Read the same original numeric constant vectors consumed by the direct RTL backend.
fn values(g: &ScGraph, input: ValueId, name: &str) -> Result<Vec<f64>, String> {
    crate::ir::kernel_contract::const_f64_vec(g, input)
        .ok_or_else(|| format!("{name} requires constant numeric vectors"))
}

/// Quantize a physical Q8.16 lane with the same checked rounding as direct RTL.
fn constant(n: &mut Net<'_>, value: f64) -> Result<Value, String> {
    let word = crate::ir::kernel_contract::q_fixed(value, 16, 24)
        .ok_or("kernel constant is not representable in signed Q8.16")?;
    Ok(n.constant(word, 24, true))
}

/// Accumulate a full signed product at the receiver's declared wrapping width.
fn mac(n: &mut Net<'_>, acc: &Value, a: &Value, b: &Value) -> Value {
    let a = n.resize(a, acc.width, true);
    let b = n.resize(b, acc.width, true);
    let product = n.binary("mul", &a, &b);
    n.binary("add", acc, &product)
}

/// Concatenate row-major results with the first result occupying the low lane.
fn pack(n: &mut Net<'_>, id: ValueId, mut values: Vec<Value>) {
    values.reverse();
    let packed = if values.len() == 1 {
        values[0].clone()
    } else {
        n.concat(&values)
    };
    n.bind(&format!("%v{}", id.0), &packed);
}

/// Lower an accepted compound numeric kernel, leaving unrelated operations untouched.
pub(super) fn emit(text: &mut String, g: &ScGraph, op: &ScOp) -> Result<bool, String> {
    match op {
        ScOp::GraphForward {
            id,
            features,
            adjacency,
            n_nodes,
            n_features,
        } => {
            graph::emit(text, g, *id, [*features, *adjacency], *n_nodes, *n_features)?;
        }
        ScOp::SoftmaxAttention { id, q, k, v, dim_k } => {
            attention::emit(text, g, *id, [*q, *k, *v], *dim_k)?;
        }
        ScOp::KuramotoStep {
            id,
            phases,
            omega,
            coupling,
            dt,
        } => {
            kuramoto::emit(text, g, *id, [*phases, *omega, *coupling], *dt)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
