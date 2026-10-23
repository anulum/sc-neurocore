// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SystemVerilog register transactions

//! Atomic nonblocking register updates, with reset taking priority over enable.

use super::value_to_wire;
use crate::ir::graph::*;

pub(super) fn emit(
    sv: &mut String,
    graph: &ScGraph,
    state: ValueId,
    next: ValueId,
    enable: Option<ValueId>,
) -> Result<(), String> {
    let initial = graph
        .ops
        .iter()
        .find_map(|op| match op {
            ScOp::State { id, initial, .. } if *id == state => Some(*initial),
            _ => None,
        })
        .ok_or("Store target must be a State")?;
    let current = value_to_wire(graph, state);
    let next = value_to_wire(graph, next);
    let initial = value_to_wire(graph, initial);
    let condition = enable.map_or_else(String::new, |id| {
        format!(" if ({})", value_to_wire(graph, id))
    });
    sv.push_str(&format!("    always @(posedge clk or negedge rst_n) begin\n        if (!rst_n) {current} <= {initial};\n        else{condition} {current} <= {next};\n    end\n"));
    Ok(())
}
