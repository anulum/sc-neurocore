// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Shared quantized kernel geometry and constants

//! Check constant-operand graph, attention and phase kernels before allocating hardware ports.

use super::graph::*;

/// Quantize a finite real by nearest rounding, refusing any unrepresentable signed word.
pub(crate) fn q_fixed(value: f64, fraction: usize, width: usize) -> Option<i64> {
    if !value.is_finite() || !(1..=64).contains(&width) || fraction >= width {
        return None;
    }
    let raw = (value * 2.0f64.powi(fraction as i32)).round();
    let limit = 2.0f64.powi(width as i32 - 1);
    (raw >= -limit && raw < limit).then_some(raw as i64)
}

/// Read original scalar-vector numbers; integer kernel constants retain their established real-unit meaning.
pub(crate) fn const_f64_vec(graph: &ScGraph, id: ValueId) -> Option<Vec<f64>> {
    graph.ops.iter().find_map(|op| {
        if op.result_id() != id {
            return None;
        }
        match op {
            ScOp::Constant {
                value: ScConst::F64Vec(values),
                ..
            } => Some(values.clone()),
            ScOp::Constant {
                value: ScConst::I64Vec(values),
                ..
            } => Some(values.iter().map(|v| *v as f64).collect()),
            _ => None,
        }
    })
}

/// Infer rectangular attention dimensions without invalid division or incomplete rows.
pub(crate) fn attention_shape(
    g: &ScGraph,
    q: ValueId,
    k: ValueId,
    v: ValueId,
    dim: usize,
) -> Option<(usize, usize, usize)> {
    if dim == 0 {
        return None;
    }
    let q = const_f64_vec(g, q)?;
    let k = const_f64_vec(g, k)?;
    let v = const_f64_vec(g, v)?;
    if q.len() % dim != 0 || k.len() % dim != 0 {
        return None;
    }
    let queries = q.len() / dim;
    let keys = k.len() / dim;
    if queries == 0 || keys == 0 || v.is_empty() || v.len() % keys != 0 {
        return None;
    }
    Some((queries, keys, v.len() / keys))
}

/// Require a nonempty result whose Q8.16 bus fits the supported HDL indexing range.
fn output_words(count: Option<usize>, kind: &str) -> Result<(), String> {
    if count
        .and_then(|n| n.checked_mul(24))
        .is_some_and(|w| w > 0 && w <= i32::MAX as usize)
    {
        Ok(())
    } else {
        Err(format!(
            "{kind} output word geometry exceeds the supported hardware indexing range"
        ))
    }
}

/// Check every original real-unit lane against the maintained Q8.16 kernel representation.
fn values(g: &ScGraph, id: ValueId, kind: &str, phase: bool) -> Result<Vec<f64>, String> {
    let values =
        const_f64_vec(g, id).ok_or_else(|| format!("{kind} requires constant numeric vectors"))?;
    for value in &values {
        let number = if phase {
            value.rem_euclid(std::f64::consts::TAU)
        } else {
            *value
        };
        if q_fixed(number, 16, 24).is_none() {
            return Err(format!(
                "{kind} constant is not representable in signed Q8.16"
            ));
        }
    }
    Ok(values)
}

/// Admit complete existing kernel contracts consistently in public verification and both emitters.
pub(crate) fn validate(g: &ScGraph, op: &ScOp) -> Result<(), String> {
    match op {
        ScOp::GraphForward {
            features,
            adjacency,
            n_nodes,
            n_features,
            ..
        } => {
            let input = values(g, *features, "GraphForward features", false)?;
            let edges = values(g, *adjacency, "GraphForward adjacency", false)?;
            let output_count = n_nodes.checked_mul(*n_features);
            output_words(output_count, "GraphForward")?;
            if *n_nodes == 0
                || *n_features == 0
                || output_count != Some(input.len())
                || n_nodes.checked_mul(*n_nodes) != Some(edges.len())
            {
                return Err(
                    "GraphForward constant lengths must match its positive node/feature geometry"
                        .into(),
                );
            }
        }
        ScOp::SoftmaxAttention { q, k, v, dim_k, .. } => {
            let (queries, _, columns) = attention_shape(g, *q, *k, *v, *dim_k).ok_or("SoftmaxAttention requires complete nonempty query/key/value rows and a positive key dimension")?;
            output_words(queries.checked_mul(columns), "SoftmaxAttention")?;
            for input in [q, k, v] {
                values(g, *input, "SoftmaxAttention", false)?;
            }
        }
        ScOp::KuramotoStep {
            phases,
            omega,
            coupling,
            dt,
            ..
        } => {
            let phases = values(g, *phases, "KuramotoStep phases", true)?;
            let omega = values(g, *omega, "KuramotoStep omega", false)?;
            let coupling = values(g, *coupling, "KuramotoStep coupling", false)?;
            let count = phases.len();
            output_words(Some(count), "KuramotoStep")?;
            if count != omega.len() || count.checked_mul(count) != Some(coupling.len()) {
                return Err("KuramotoStep requires matching phase/omega vectors and a square coupling matrix".into());
            }
            if q_fixed(*dt, 16, 24).is_none() {
                return Err("KuramotoStep dt is not representable in signed Q8.16".into());
            }
        }
        _ => {}
    }
    Ok(())
}
