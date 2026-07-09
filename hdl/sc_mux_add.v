// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stochastic Computing MUX gate (scaled addition)

// SC scaled addition via 2:1 multiplexer.
//
// P(out) = P(sel) * P(a) + (1 - P(sel)) * P(b)
//
// When sel is driven by a Bernoulli(0.5) bitstream (e.g., from an LFSR),
// this computes the average: P(out) = (P(a) + P(b)) / 2.
//
// For weighted addition with weight w:
//   Drive sel with Bernoulli(w) → P(out) = w * P(a) + (1-w) * P(b)

`timescale 1ns / 1ps

module sc_mux_add (
    input wire a,      // SC bitstream input A
    input wire b,      // SC bitstream input B
    input wire sel,    // Selection bitstream (weight)
    output wire out    // SC bitstream output: sel*a + (1-sel)*b
);

assign out = sel ? a : b;

endmodule
