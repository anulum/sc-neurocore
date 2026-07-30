// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained normalized EnergyLIF in signed Q32.32
`timescale 1ns/1ps
module sc_normalized_energy_lif(input wire clk,input wire rst_n,input wire signed[63:0]current_t,output reg signed[63:0]v_out,output reg signed[63:0]epsilon_out,output reg event_out);
localparam signed[63:0]ONE=64'sd4294967296,V_REST=-64'sd300647710720,V_RESET=-64'sd300647710720,V_THRESHOLD=-64'sd214748364800,MD=64'sd3886247119,ED=64'sd4286385946,STEADY=64'sd4087201773,TRANSIENT=64'sd4083049255,ALPHA=64'sd429496730,GATE=64'sd429496730;
reg signed[63:0]v_reg,epsilon_reg;wire signed[63:0]energy_delta=epsilon_reg-ONE;
function automatic signed[63:0]qmul(input signed[63:0]a,input signed[63:0]b);reg signed[127:0]p;begin p=$signed(a)*$signed(b);qmul=p>>>32;end endfunction
wire signed[63:0]epsilon_candidate=ONE+qmul(energy_delta,ED);wire signed[63:0]integral=STEADY+qmul(energy_delta,TRANSIENT);wire signed[63:0]v_candidate=V_REST+qmul(v_reg-V_REST,MD)+qmul(current_t,integral)/10;wire spike=v_candidate>=V_THRESHOLD&&epsilon_candidate>GATE;
always@(posedge clk)begin if(!rst_n)begin v_reg<=V_REST;epsilon_reg<=ONE;v_out<=V_REST;epsilon_out<=ONE;event_out<=0;end else if(spike)begin v_reg<=V_RESET;epsilon_reg<=epsilon_candidate>ALPHA?epsilon_candidate-ALPHA:0;v_out<=V_RESET;epsilon_out<=epsilon_candidate>ALPHA?epsilon_candidate-ALPHA:0;event_out<=1;end else begin v_reg<=v_candidate;epsilon_reg<=epsilon_candidate;v_out<=v_candidate;epsilon_out<=epsilon_candidate;event_out<=0;end end
endmodule
