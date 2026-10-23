// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — state contract

//! Register reset words and atomic driver verification.

use super::{
    constant_words::scalar_layout, graph::*, stream_contract::value_type, verify::VerifyError,
};
use std::collections::HashMap;

pub(crate) fn verify(graph: &ScGraph, errors: &mut Vec<VerifyError>) {
    let mut drivers = HashMap::new();
    for (op_index, op) in graph.ops.iter().enumerate() {
        let result = match op {
            ScOp::State { initial, ty, .. } => scalar_layout(ty).and_then(|_| {
                match graph.ops[..op_index]
                    .iter()
                    .find(|op| op.result_id() == *initial)
                {
                    Some(ScOp::Constant { ty: reset_ty, .. }) if reset_ty == ty => Ok(()),
                    _ => Err(
                        "State reset requires an earlier constant with the exact state type".into(),
                    ),
                }
            }),
            ScOp::Store {
                state,
                next,
                enable,
                ..
            } => {
                if drivers.insert(*state, op_index).is_some() {
                    errors.push(VerifyError {
                        op_index,
                        message: format!("State {state} has multiple Store drivers"),
                    });
                }
                validate_store(graph, *state, *next, *enable)
            }
            _ => Ok(()),
        };
        if let Err(message) = result {
            errors.push(VerifyError { op_index, message });
        }
    }
    for (op_index, op) in graph.ops.iter().enumerate() {
        if let ScOp::State { id, .. } = op {
            if !drivers.contains_key(id) {
                errors.push(VerifyError {
                    op_index,
                    message: format!("State {id} requires exactly one Store driver"),
                });
            }
        }
    }
}

fn validate_store(
    graph: &ScGraph,
    state: ValueId,
    next: ValueId,
    enable: Option<ValueId>,
) -> Result<(), String> {
    let ty = graph
        .ops
        .iter()
        .find_map(|op| match op {
            ScOp::State { id, ty, .. } if *id == state => Some(ty),
            _ => None,
        })
        .ok_or_else(|| format!("Store target {state} must be a State"))?;
    if value_type(graph, next)? != *ty {
        return Err(format!(
            "Store next {next} must have the exact state type {ty}"
        ));
    }
    if let Some(enable) = enable {
        if value_type(graph, enable)? != ScType::Bool {
            return Err("Store enable must have Bool type".into());
        }
    }
    Ok(())
}
