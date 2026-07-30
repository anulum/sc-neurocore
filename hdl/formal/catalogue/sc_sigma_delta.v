// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — sampled APSDM specialization in signed Q32.32

// One rising edge advances the default dt=0.1, tau=10, delta=1 profile.
// This bounded representative does not claim continuous-time event timing,
// universal binary64 equivalence, timing/PPA, or device evidence.
`timescale 1ns / 1ps
module sc_sigma_delta(input wire clk,input wire rst_n,input wire signed[63:0] current_t,output reg signed[63:0] sigma_out,output reg signed[63:0] reconstruction_out,output reg event_out);
localparam signed[63:0] DT=64'sd429496730;
localparam signed[63:0] DECAY=64'sd4252231657;
localparam signed[63:0] DELTA=64'sd4294967296;
localparam signed[63:0] HALF_DELTA=64'sd2147483648;
localparam signed[63:0] LIMIT=64'sd4294967296000000;
reg signed[63:0] sigma_reg;reg signed[63:0] reconstruction_reg;
function automatic signed[63:0] qmul(input signed[63:0] a,input signed[63:0] b);reg signed[127:0] p;begin p=$signed(a)*$signed(b);qmul=p>>>32;end endfunction
function automatic signed[63:0] bound(input signed[127:0] v);begin if(v < -$signed(LIMIT))bound=-LIMIT;else if(v>$signed(LIMIT))bound=LIMIT;else bound=v[63:0];end endfunction
wire signed[63:0] sigma_candidate=bound($signed(sigma_reg)+qmul(current_t,DT));
wire signed[63:0] reconstruction_decay=bound(qmul(reconstruction_reg,DECAY));
wire candidate_event=$signed(sigma_candidate)-$signed(reconstruction_decay)>=$signed(HALF_DELTA);
wire signed[63:0] reconstruction_candidate=candidate_event?bound($signed(reconstruction_decay)+$signed(DELTA)):reconstruction_decay;
always@(posedge clk)begin if(!rst_n)begin sigma_reg<=0;reconstruction_reg<=0;sigma_out<=0;reconstruction_out<=0;event_out<=0;end else begin sigma_reg<=sigma_candidate;reconstruction_reg<=reconstruction_candidate;sigma_out<=sigma_candidate;reconstruction_out<=reconstruction_candidate;event_out<=candidate_event;end end
endmodule
