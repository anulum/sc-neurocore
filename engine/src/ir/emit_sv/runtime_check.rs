// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Runtime check SystemVerilog emission

//! Combinational simulation failures have no synthesized storage or data output.

use crate::ir::{graph::*, runtime_check::RuntimeCheckKind};

pub(super) fn emit(
    sv: &mut String,
    graph: &ScGraph,
    condition: ValueId,
    enable: Option<ValueId>,
    kind: RuntimeCheckKind,
) {
    let condition = super::value_to_wire(graph, condition);
    let enable = enable.map_or_else(|| "1'b1".into(), |id| super::value_to_wire(graph, id));
    sv.push_str(&format!(
        "    `ifndef SYNTHESIS\n\
         \x20   always_comb begin\n\
         \x20       if ({enable} && !{condition}) $fatal(1, \"{}\");\n\
         \x20   end\n\
         \x20   `endif\n",
        kind.message()
    ));
}
