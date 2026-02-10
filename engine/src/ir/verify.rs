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
    verify_ssa(graph, &mut errors);
    verify_operand_defs(graph, &mut errors);
    verify_no_cycles(graph, &mut errors);
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

/// Check that every operand references a value defined by an earlier op.
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
        defined.insert(op.result_id());
    }
}

/// Check that the operation list is acyclic (topological order).
/// Since we enforce operand-before-use in `verify_operand_defs`,
/// this is automatically satisfied if that check passes.
/// This function is a belt-and-suspenders DFS cycle check.
fn verify_no_cycles(graph: &ScGraph, errors: &mut Vec<VerifyError>) {
    // Build adjacency from result_id → operand ids
    let mut adj: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    for op in &graph.ops {
        adj.insert(op.result_id(), op.operands());
    }

    let mut visited: HashSet<ValueId> = HashSet::new();
    let mut in_stack: HashSet<ValueId> = HashSet::new();

    fn dfs(
        node: ValueId,
        adj: &HashMap<ValueId, Vec<ValueId>>,
        visited: &mut HashSet<ValueId>,
        in_stack: &mut HashSet<ValueId>,
    ) -> bool {
        if in_stack.contains(&node) {
            return true; // cycle
        }
        if visited.contains(&node) {
            return false;
        }
        visited.insert(node);
        in_stack.insert(node);
        if let Some(deps) = adj.get(&node) {
            for dep in deps {
                if dfs(*dep, adj, visited, in_stack) {
                    return true;
                }
            }
        }
        in_stack.remove(&node);
        false
    }

    for op in &graph.ops {
        let id = op.result_id();
        if dfs(id, &adj, &mut visited, &mut in_stack) {
            errors.push(VerifyError {
                op_index: 0,
                message: format!("cycle detected involving {}", id),
            });
            break;
        }
    }
}
