// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — pinned Fardet-Levina eLIF RK4 in signed Q32.32
`timescale 1ns/1ps
module energy_lif(input wire clk,input wire rst_n,input wire signed[63:0] current_t,output reg signed[63:0] v_out,output reg signed[63:0] epsilon_out,output reg event_out);
localparam signed[63:0] ONE=64'sd4294967296,DT=64'sd429496730,V_INIT=-64'sd261993005056,E_INIT=64'sd1374389535,V_RESET=-64'sd266287972352,V_THRESHOLD=-64'sd253403070464,EPSILON_C=64'sd773094113,DELTA=64'sd42949673,LEAK_BASE=-64'sd251255586816,EF=-64'sd266287972352;
reg signed[63:0] v_reg,epsilon_reg;reg signed[63:0] k1v,k1e,k2v,k2e,k3v,k3e,k4v,k4e,v2,e2,v3,e3,v4,e4,v_candidate,e_candidate;
function automatic signed[63:0] qmul(input signed[63:0]a,input signed[63:0]b);reg signed[127:0]p;begin p=$signed(a)*$signed(b);qmul=p>>>32;end endfunction
function automatic signed[63:0] rhs_v(input signed[63:0]v,input signed[63:0]e,input signed[63:0]i);reg signed[63:0]leak;begin leak=LEAK_BASE-($signed(e)<<<3);rhs_v=(9*$signed(leak-v)+$signed(i))/100;end endfunction
function automatic signed[63:0] rhs_e(input signed[63:0]v,input signed[63:0]e);reg signed[63:0]x,cube,cost;begin x=ONE-($signed(e)<<<1);cube=qmul(qmul(x,x),x);cost=($signed(v)-$signed(EF))/22;rhs_e=($signed(cube)-$signed(cost))/200;end endfunction
always@* begin
 k1v=rhs_v(v_reg,epsilon_reg,current_t);k1e=rhs_e(v_reg,epsilon_reg);
 v2=v_reg+(qmul(DT,k1v)>>>1);e2=epsilon_reg+(qmul(DT,k1e)>>>1);k2v=rhs_v(v2,e2,current_t);k2e=rhs_e(v2,e2);
 v3=v_reg+(qmul(DT,k2v)>>>1);e3=epsilon_reg+(qmul(DT,k2e)>>>1);k3v=rhs_v(v3,e3,current_t);k3e=rhs_e(v3,e3);
 v4=v_reg+qmul(DT,k3v);e4=epsilon_reg+qmul(DT,k3e);k4v=rhs_v(v4,e4,current_t);k4e=rhs_e(v4,e4);
 v_candidate=v_reg+qmul(DT,($signed(k1v)+($signed(k2v)<<<1)+($signed(k3v)<<<1)+$signed(k4v))/6);
 e_candidate=epsilon_reg+qmul(DT,($signed(k1e)+($signed(k2e)<<<1)+($signed(k3e)<<<1)+$signed(k4e))/6);
end
always@(posedge clk)begin if(!rst_n)begin v_reg<=V_INIT;epsilon_reg<=E_INIT;v_out<=V_INIT;epsilon_out<=E_INIT;event_out<=0;end else if(v_candidate>V_THRESHOLD&&e_candidate>EPSILON_C)begin v_reg<=V_RESET;epsilon_reg<=e_candidate-DELTA;v_out<=V_RESET;epsilon_out<=e_candidate-DELTA;event_out<=1;end else begin v_reg<=v_candidate;epsilon_reg<=e_candidate;v_out<=v_candidate;epsilon_out<=e_candidate;event_out<=0;end end
endmodule
