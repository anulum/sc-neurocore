// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Triple Modular Redundancy (TMR) majority voter
//
// Generic bit-wise majority voter for fault-tolerant neuromorphic
// deployment. Addresses undervolting / fault injection attacks
// (review §4.3) by triplicating critical control paths.
//
// Usage:
//   Apply selectively to: AER routers, threshold registers,
//   spike output logic. Do NOT TMR the entire stochastic datapath
//   (area-prohibitive; stochastic noise already provides inherent
//   single-bit fault tolerance).
//
// Architecture:
//   Pure combinational bit-wise majority: out[i] = majority(a[i], b[i], c[i])
//   Optional error flag when any input disagrees with the majority.

module sc_tmr_voter #(
    parameter DATA_WIDTH = 16
)(
    input  wire [DATA_WIDTH-1:0] a,       // replica 0
    input  wire [DATA_WIDTH-1:0] b,       // replica 1
    input  wire [DATA_WIDTH-1:0] c,       // replica 2
    output wire [DATA_WIDTH-1:0] voted,   // majority output
    output wire                  error    // any disagreement detected
);

    // Bit-wise majority: out = (a & b) | (b & c) | (a & c)
    assign voted = (a & b) | (b & c) | (a & c);

    // Error flag: at least one replica disagrees with majority
    assign error = |(a ^ voted) | |(b ^ voted) | |(c ^ voted);

endmodule
