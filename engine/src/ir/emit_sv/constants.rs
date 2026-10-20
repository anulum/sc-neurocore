// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Declared constant bus emission

//! Verilog constants use their declared scalar width and binary point.

use super::{emit_ram_style_attribute, ScConst, ScType, SvTarget, ValueId};
use crate::ir::constant_words::constant_words;

/// Emit constants at their declared element widths and binary points.
pub(super) fn emit(
    sv: &mut String,
    id: ValueId,
    value: &ScConst,
    ty: &ScType,
    target: &SvTarget,
) -> Result<(), String> {
    let (layout, words) = constant_words(value, ty)?;
    let width = layout.width as usize;
    if matches!(ty, ScType::Vec { .. }) {
        let bits = width * words.len();
        emit_ram_style_attribute(sv, target, bits as u64);
        sv.push_str(&format!("    wire [{}:0] c{};\n", bits - 1, id.0));
        for (index, word) in words.into_iter().enumerate() {
            sv.push_str(&format!(
                "    assign c{}[{} +: {width}] = {};\n",
                id.0,
                index * width,
                layout.literal(word)
            ));
        }
    } else {
        let signed = if layout.signed { "signed " } else { "" };
        sv.push_str(&format!(
            "    localparam {signed}[{}:0] c{} = {};\n",
            width - 1,
            id.0,
            layout.literal(words[0])
        ));
    }
    Ok(())
}
