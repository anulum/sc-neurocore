// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained bipolar accumulator in signed Q32.32
`timescale 1ns / 1ps
module sc_sigma_delta_accumulator(input wire clk,input wire rst_n,input wire signed[63:0] current_t,output reg signed[63:0] sigma_out,output reg signed[1:0] event_out);
localparam signed[63:0] THRESHOLD=64'sd4294967296;localparam signed[63:0] LIMIT=64'sd4294967296000000;reg signed[63:0] sigma_reg;
function automatic signed[63:0] bound(input signed[127:0] v);begin if(v<-$signed(LIMIT))bound=-LIMIT;else if(v>$signed(LIMIT))bound=LIMIT;else bound=v[63:0];end endfunction
wire signed[63:0] accumulated=bound($signed(sigma_reg)+$signed(current_t));wire positive=$signed(accumulated)>=$signed(THRESHOLD);wire negative=$signed(accumulated)<=-$signed(THRESHOLD);wire signed[63:0] candidate=positive?bound($signed(accumulated)-$signed(THRESHOLD)):negative?bound($signed(accumulated)+$signed(THRESHOLD)):accumulated;wire signed[1:0] event_candidate=positive?2'sd1:negative?-2'sd1:2'sd0;
always@(posedge clk)begin if(!rst_n)begin sigma_reg<=0;sigma_out<=0;event_out<=0;end else begin sigma_reg<=candidate;sigma_out<=candidate;event_out<=event_candidate;end end
endmodule
