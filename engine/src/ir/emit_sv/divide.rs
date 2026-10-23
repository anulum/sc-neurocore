// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Portable full-word unsigned constant division

//! Preserve high dividend bits before explicitly truncating a quotient.

/// Emit division by a positive u64, avoiding simulator wide-division discrepancies.
pub(super) fn emit(
    sv: &mut String,
    output: &str,
    numerator: &str,
    input_width: usize,
    divisor: u64,
    output_width: usize,
) {
    if divisor == 1 {
        sv.push_str(&format!("    assign {output} = $unsigned({numerator});\n"));
        return;
    }
    if input_width <= 64 {
        sv.push_str(&format!(
            "    assign {output} = {numerator} / 64'd{divisor};\n"
        ));
        return;
    }
    // The remainder is always less than divisor before shifting in one bit,
    // so 65 bits suffice independently of the dividend's physical width.
    let function = format!("{output}_unsigned_divide");
    sv.push_str(&format!(
        "    function automatic [{}:0] {function}(input [{}:0] dividend);\n\
         \x20       reg [64:0] remainder;\n\
         \x20       reg [{}:0] quotient;\n\
         \x20       integer bit_index;\n\
         \x20       begin\n\
         \x20           remainder = 65'd0;\n\
         \x20           quotient = {output_width}'d0;\n\
         \x20           for (bit_index = {}; bit_index >= 0; bit_index = bit_index - 1) begin\n\
         \x20               remainder = {{remainder[63:0], dividend[bit_index]}};\n\
         \x20               quotient = quotient << 1;\n\
         \x20               if (remainder >= 65'd{divisor}) begin\n\
         \x20                   remainder = remainder - 65'd{divisor};\n\
         \x20                   quotient[0] = 1'b1;\n\
         \x20               end\n\
         \x20           end\n\
         \x20           {function} = quotient;\n\
         \x20       end\n    endfunction\n\
         \x20   assign {output} = {function}({numerator});\n",
        output_width - 1,
        input_width - 1,
        output_width - 1,
        input_width - 1
    ));
}
