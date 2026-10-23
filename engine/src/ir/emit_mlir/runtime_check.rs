// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Structured CIRCT runtime check emission

//! Preserve conditional fatal effects through structured SV operations and synthesis guards.

use crate::ir::{graph::*, runtime_check::RuntimeCheckKind};

pub(super) fn emit(
    mlir: &mut String,
    graph: &ScGraph,
    id: ValueId,
    condition: ValueId,
    enable: Option<ValueId>,
    kind: RuntimeCheckKind,
) {
    let id = id.0;
    let condition = super::value_wire(graph, condition);
    mlir.push_str(&format!(
        "  %v{id} = comb.xor {condition}, %sc_one : i1\n\
         \x20 sv.ifdef @{}__sc_synthesis {{\n\
         \x20 }} else {{\n\
         \x20   sv.alwayscomb {{\n\
         \x20     sv.if %v{id} {{\n",
        graph.name
    ));
    if let Some(enable) = enable {
        mlir.push_str(&format!(
            "        sv.if {} {{\n          sv.fatal.procedural 1, \"{}\"\n        }}\n",
            super::value_wire(graph, enable),
            kind.message()
        ));
    } else {
        mlir.push_str(&format!(
            "        sv.fatal.procedural 1, \"{}\"\n",
            kind.message()
        ));
    }
    mlir.push_str("      }\n    }\n  }\n");
}
