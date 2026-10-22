// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Full-width unsigned division in CIRCT

//! Portable integer circuits, preserving the complete dividend before narrowing.

/// Emit a positive constant division, with a 65-bit restoring remainder for wide words.
pub(super) fn emit(
    m: &mut String,
    output: &str,
    numerator: &str,
    input_width: usize,
    divisor: u64,
    output_width: usize,
) {
    let p = output.trim_start_matches('%');
    if input_width <= 64 {
        let word = if input_width < 64 {
            let pad = 64 - input_width;
            m.push_str(&format!("  %{p}_padding = hw.constant 0 : i{pad}\n  %{p}_dividend = comb.concat %{p}_padding, {numerator} : i{pad}, i{input_width}\n"));
            format!("%{p}_dividend")
        } else {
            numerator.into()
        };
        m.push_str(&format!(
            "  %{p}_denominator = hw.constant {divisor} : i64\n"
        ));
        if output_width == 64 {
            m.push_str(&format!(
                "  {output} = comb.divu {word}, %{p}_denominator : i64\n"
            ));
        } else {
            m.push_str(&format!("  %{p}_quotient = comb.divu {word}, %{p}_denominator : i64\n  {output} = comb.extract %{p}_quotient from 0 : (i64) -> i{output_width}\n"));
        }
        return;
    }
    m.push_str(&format!("  %{p}_remainder_zero = hw.constant 0 : i65\n  %{p}_denominator = hw.constant {divisor} : i65\n"));
    let mut remainder = format!("%{p}_remainder_zero");
    let mut quotient = Vec::new();
    for i in (0..input_width).rev() {
        m.push_str(&format!(
            "  %{p}_bit{i} = comb.extract {numerator} from {i} : (i{input_width}) -> i1\n\
             \x20 %{p}_low{i} = comb.extract {remainder} from 0 : (i65) -> i64\n\
             \x20 %{p}_shift{i} = comb.concat %{p}_low{i}, %{p}_bit{i} : i64, i1\n\
             \x20 %{p}_take{i} = comb.icmp uge %{p}_shift{i}, %{p}_denominator : i65\n\
             \x20 %{p}_subtract{i} = comb.sub %{p}_shift{i}, %{p}_denominator : i65\n\
             \x20 %{p}_remainder{i} = comb.mux %{p}_take{i}, %{p}_subtract{i}, %{p}_shift{i} : i65\n"
        ));
        remainder = format!("%{p}_remainder{i}");
        if i < output_width {
            quotient.push(format!("%{p}_take{i}"));
        }
    }
    m.push_str(&format!(
        "  {output} = comb.concat {} : {}\n",
        quotient.join(", "),
        vec!["i1"; output_width].join(", ")
    ));
}
