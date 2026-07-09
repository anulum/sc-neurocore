// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — CORDIV: stochastic computing divider
//
// Li, Qian, Riedel & Bazargan, IEEE Trans. Signal Process. 62(9), 2014.
//
// P(z=1) converges to P(x=1) / P(y=1) when P(x) <= P(y).
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
