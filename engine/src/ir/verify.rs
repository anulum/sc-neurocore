// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Graph verification passes

//! Graph verification passes.

use std::collections::{HashMap, HashSet};

use crate::ir::graph::*;

/// Verification error with location info.
#[derive(Debug, Clone)]
pub struct VerifyError {
    pub op_index: usize,
    pub message: String,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "op[{}]: {}", self.op_index, self.message)
    }
}

/// Run all verification passes on a graph. Returns `Ok(())` if valid,
/// or a list of all errors found.
pub fn verify(graph: &ScGraph) -> Result<(), Vec<VerifyError>> {
    let mut errors = Vec::new();
    if let Some(source) = &graph.source {
        if let Err(message) = source.verify(graph) {
            errors.push(VerifyError {
                op_index: 0,
                message,
            });
        }
    }
    super::identifiers::verify(graph, &mut errors);
    verify_ssa(graph, &mut errors);
    verify_operand_defs(graph, &mut errors);
    if !super::type_contract::verify(graph, &mut errors) {
        return Err(errors);
    }
    verify_hardware_words(graph, &mut errors);
    super::word_contract::verify(graph, &mut errors);
    super::state_contract::verify(graph, &mut errors);
    super::runtime_check::verify(graph, &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Check that every ValueId is defined exactly once.
fn verify_ssa(graph: &ScGraph, errors: &mut Vec<VerifyError>) {
    let mut defined: HashMap<ValueId, usize> = HashMap::new();
    for (idx, op) in graph.ops.iter().enumerate() {
        let id = op.result_id();
        if let Some(prev_idx) = defined.insert(id, idx) {
            errors.push(VerifyError {
                op_index: idx,
                message: format!("{} is already defined by op[{}]", id, prev_idx),
            });
        }
    }
}

/// Check every operand is defined earlier. With unique IDs this also proves acyclicity.
fn verify_operand_defs(graph: &ScGraph, errors: &mut Vec<VerifyError>) {
    let mut defined: HashSet<ValueId> = HashSet::new();
    for (idx, op) in graph.ops.iter().enumerate() {
        for operand in op.operands() {
            if !defined.contains(&operand) {
                errors.push(VerifyError {
                    op_index: idx,
                    message: format!(
                        "operand {} not defined before use in {}",
                        operand,
                        op.op_name()
                    ),
                });
            }
        }
        if !matches!(
            op,
            ScOp::Output { .. } | ScOp::Store { .. } | ScOp::Check { .. }
        ) {
            defined.insert(op.result_id());
        }
    }
}

/// Reject malformed constant words and dense geometry before any HDL dimensions are formed.
fn verify_hardware_words(graph: &ScGraph, errors: &mut Vec<VerifyError>) {
    for (op_index, op) in graph.ops.iter().enumerate() {
        let result = match op {
            ScOp::Constant { value, ty, .. } => {
                super::constant_words::constant_words(value, ty).map(|_| ())
            }
            ScOp::DenseForward {
                inputs,
                weights,
                leak,
                gain,
                params,
                ..
            } => super::dense_contract::validate_dense(
                graph, *inputs, *weights, *leak, *gain, params,
            ),
            ScOp::LifStep {
                current,
                leak,
                gain,
                noise,
                params,
                ..
            } => super::neuron_contract::lif(graph, [*current, *leak, *gain, *noise], params),
            ScOp::DclsLayer {
                spike,
                weights,
                centre,
                sigma,
                params,
                ..
            } => super::neuron_contract::dcls(graph, [*spike, *weights, *centre, *sigma], params),
            ScOp::Encode { prob, seed, .. } => {
                if *seed == 0 {
                    Err("Encode seed must be nonzero".into())
                } else {
                    super::dense_contract::value_type(graph, *prob)
                        .and_then(|ty| super::dense_contract::probability_layout(&ty))
                        .map(|_| ())
                }
            }
            ScOp::GraphForward { .. }
            | ScOp::SoftmaxAttention { .. }
            | ScOp::KuramotoStep { .. } => super::kernel_contract::validate(graph, op),
            ScOp::Scale { .. } | ScOp::Offset { .. } | ScOp::Reduce { .. } => {
                super::arithmetic_contract::validate(graph, op)
            }
            _ => Ok(()),
        };
        if let Err(message) = super::stream_contract::validate(graph, op) {
            errors.push(VerifyError { op_index, message });
        }
        if let Err(message) = result {
            errors.push(VerifyError { op_index, message });
        }
    }
}
