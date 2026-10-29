// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Probability words to encoder thresholds

//! Convert declared signed Q probability words to the encoder's unsigned 16-bit scale.

use super::{value_to_wire, ScGraph, ScType, ValueId};
use crate::ir::dense_contract::{probability_layout, value_type};

/// Emit one packed threshold bus, preserving the original operand's binary point.
pub(super) fn emit(
    sv: &mut String,
    graph: &ScGraph,
    id: ValueId,
    count: usize,
    name: &str,
) -> Result<(), String> {
    let ty = value_type(graph, id)?;
    let element = match &ty {
        ScType::Vec { element, .. } => element.as_ref(),
        _ => &ty,
    };
    let layout = probability_layout(element)?;
    let (width, frac) = (layout.width, layout.fraction);
    let source = value_to_wire(graph, id);
    sv.push_str(&format!("    wire [{}:0] {name};\n", count * 16 - 1));
    for i in 0..count {
        let raw = format!("{name}_raw_{i}");
        let product = format!("{name}_product_with_unused_bits_{i}");
        let product_width = width + 17;
        let operand = if matches!(ty, ScType::Vec { .. }) {
            format!("{source}[{} +: {width}]", i * width as usize)
        } else {
            source.clone()
        };
        sv.push_str(&format!(
            "    wire signed [{}:0] {raw} = $signed({operand});\n",
            width - 1
        ));
        sv.push_str(&format!(
            "    wire signed [{}:0] {product} = {raw} * {product_width}'sd65535;\n",
            product_width - 1
        ));
        let upper = if frac + 1 < width {
            format!("{raw} >= {width}'sd{} ? 16'hffff : ", 1_u64 << frac)
        } else {
            String::new()
        };
        sv.push_str(&format!("    assign {name}[{} +: 16] = {raw} <= {width}'sd0 ? 16'd0 : {upper}{product}[{frac} +: 16];\n", i * 16));
    }
    Ok(())
}
