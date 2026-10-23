// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — CIRCT register transactions

//! CIRCT register feedback in the HW graph region.

use super::value_wire;
use crate::ir::graph::*;

pub(super) fn emit(
    mlir: &mut String,
    graph: &ScGraph,
    id: ValueId,
    initial: ValueId,
    ty: &ScType,
) -> Result<(), String> {
    let (next, enable) = graph
        .ops
        .iter()
        .find_map(|op| match op {
            ScOp::Store {
                state,
                next,
                enable,
                ..
            } if *state == id => Some((*next, *enable)),
            _ => None,
        })
        .ok_or("State requires a Store driver")?;
    let bits = ty.bit_width();
    let mut next = value_wire(graph, next);
    if let Some(enable) = enable {
        mlir.push_str(&format!(
            "  %v{}_enabled = comb.mux {}, {}, %v{} : i{bits}\n",
            id.0,
            value_wire(graph, enable),
            next,
            id.0
        ));
        next = format!("%v{}_enabled", id.0);
    }
    mlir.push_str(&format!(
        "  %v{} = seq.firreg {} clock %sc_clock reset async %sc_reset, {} : i{bits}\n",
        id.0,
        next,
        value_wire(graph, initial)
    ));
    Ok(())
}
