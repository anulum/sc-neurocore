// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — CORDIV: stochastic computing divider
//
// Chen and Hayes, "Design of division circuits for stochastic computing",
// IEEE ISVLSI 2016, doi:10.1109/ISVLSI.2016.48.
//
// P(z=1) converges to P(x=1) / P(y=1) when P(x) <= P(y) and x and y are
// compared against one shared random sequence whose successive values are
// independent, so that every 1 of x falls on a 1 of y. A 16-bit LFSR read
// every clock is not such a sequence: with it z misses the quotient, and
// equals x whenever 2 P(x) <= P(y). Independent streams give
// P(x) / (P(x) + P(y) - P(x) P(y)). See docs/hardware/sc_cordiv.md.
//
// Truth table:
//   x=1         → z = 1
//   x=0, y=1    → z = 0
//   x=0, y=0    → z = z_prev (hold)

module sc_cordiv (
    input  wire clk,
    input  wire rst,
    input  wire x,   // numerator bitstream
    input  wire y,   // denominator bitstream
    output reg  z    // quotient bitstream
);

    always @(posedge clk or posedge rst) begin
        if (rst)
            z <= 1'b0;
        else if (x)
            z <= 1'b1;
        else if (y)
            z <= 1'b0;
        // else: hold z (implicit in always block — z keeps previous value)
    end

endmodule
