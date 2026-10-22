// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Packed and temporal population counting

//! Emit exact-width population counts and explicit Q8.8 probability decoding.

use super::{value_to_wire, ScGraph, ValueId};
use crate::ir::stream_contract::shape;

/// Accumulate the bits of a physical bus without indexing a scalar wire.
fn sum_bits(sv: &mut String, name: &str, wire: &str, width: usize, bits: u32) {
    sv.push_str(&format!("    logic [{}:0] {name};\n", bits - 1));
    if width == 1 {
        sv.push_str(&format!("    always_comb {name} = {wire};\n"));
    } else {
        sv.push_str(&format!(
            "    always_comb begin\n        {name} = {bits}'d0;\n\
             \x20       for (integer i = 0; i < {width}; i = i + 1)\n\
             \x20           {name} = {name} + {wire}[i];\n    end\n"
        ));
    }
}

/// Emit either combinational packed counting or reset-anchored stream windows.
pub(super) fn emit(
    sv: &mut String,
    graph: &ScGraph,
    id: ValueId,
    input: ValueId,
    decode: bool,
) -> Result<(), String> {
    let layout = shape(graph, input)?;
    let opportunities = layout.opportunities()?;
    let wire = value_to_wire(graph, input);
    let count = if decode {
        format!("v{}_count", id.0)
    } else {
        format!("v{}", id.0)
    };
    if decode {
        sv.push_str(&format!("    logic [63:0] {count};\n"));
    }
    let prefix = format!("v{}_observation", id.0);
    let count_bits = 64 - opportunities.leading_zeros();
    let delta_bits = 64 - (layout.width as u64).leading_zeros();
    sum_bits(
        sv,
        &format!("{prefix}_delta"),
        &wire,
        layout.width,
        delta_bits,
    );
    if let Some(window) = layout.window {
        let phase_bits = (64 - (window - 1).leading_zeros()).max(1);
        sv.push_str(&format!(
            "    logic [{}:0] {prefix}_accumulator;\n\
             \x20   logic [{}:0] {prefix}_phase;\n\
             \x20   wire [{}:0] {prefix}_sum = {prefix}_accumulator + {prefix}_delta;\n\
             \x20   always @(posedge clk or negedge rst_n) begin\n\
             \x20       if (!rst_n) begin\n\
             \x20           {prefix}_phase <= {phase_bits}'d0;\n\
             \x20           {prefix}_accumulator <= {count_bits}'d0;\n\
             \x20           {count} <= 64'd0;\n\
             \x20       end else if ({prefix}_phase == {phase_bits}'d{}) begin\n\
             \x20           {count} <= {prefix}_sum;\n\
             \x20           {prefix}_accumulator <= {count_bits}'d0;\n\
             \x20           {prefix}_phase <= {phase_bits}'d0;\n\
             \x20       end else begin\n\
             \x20           {prefix}_accumulator <= {prefix}_sum;\n\
             \x20           {prefix}_phase <= {prefix}_phase + {phase_bits}'d1;\n\
             \x20       end\n    end\n",
            count_bits - 1,
            phase_bits - 1,
            count_bits - 1,
            window - 1
        ));
    } else {
        sv.push_str(&format!("    always_comb {count} = {prefix}_delta;\n"));
    }
    if decode {
        let scaled_width = count_bits as usize + 8;
        sv.push_str(&format!(
            "    wire [{}:0] {prefix}_scaled = {{{count}[{}:0], 8'd0}};\n",
            scaled_width - 1,
            count_bits - 1
        ));
        super::divide::emit(
            sv,
            &format!("v{}", id.0),
            &format!("{prefix}_scaled"),
            scaled_width,
            opportunities,
            16,
        );
    }
    Ok(())
}
