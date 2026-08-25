// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SC-NeuroCore — retained SC triangular recurrence in signed Q32.32
`timescale 1ns/1ps
module sc_triangular_mckean(
    input wire clk,
    input wire rst_n,
    input wire signed [63:0] current_t,
    output reg signed [63:0] v_out,
    output reg signed [63:0] w_out,
    output reg event_out
);
localparam signed [63:0] ONE=64'sd4294967296;
localparam signed [63:0] A=64'sd1073741824;
localparam signed [63:0] EPSILON=64'sd42949673;
localparam signed [63:0] GAMMA=64'sd2147483648;
localparam signed [63:0] DT=64'sd429496730;
localparam signed [63:0] V_PEAK=64'sd3435973837;
localparam signed [63:0] HALF_A=64'sd536870912;
localparam signed [63:0] MID=64'sd2684354560;
reg signed [63:0] v_reg,w_reg;
reg signed [63:0] k1v,k1w,k2v,k2w,k3v,k3w,k4v,k4w,v2,w2,v3,w3,v4,w4,v_candidate,w_candidate;
function automatic signed [63:0] qmul(input signed [63:0] x,input signed [63:0] y);
    reg signed [127:0] product;
    begin product=$signed(x)*$signed(y);qmul=product>>>32;end
endfunction
function automatic signed [63:0] f_v(input signed [63:0] v);
    begin
        if($signed(v)<$signed(HALF_A)) f_v=-$signed(v);
        else if($signed(v)<$signed(MID)) f_v=$signed(v)-$signed(A);
        else f_v=$signed(ONE)-$signed(v);
    end
endfunction
function automatic signed [63:0] rhs_v(input signed [63:0] v,input signed [63:0] w,input signed [63:0] current);
    begin rhs_v=f_v(v)-$signed(w)+$signed(current);end
endfunction
function automatic signed [63:0] rhs_w(input signed [63:0] v,input signed [63:0] w);
    begin rhs_w=qmul(EPSILON,$signed(v)-qmul(GAMMA,w));end
endfunction
always @* begin
    k1v=rhs_v(v_reg,w_reg,current_t);k1w=rhs_w(v_reg,w_reg);
    v2=v_reg+(qmul(DT,k1v)>>>1);w2=w_reg+(qmul(DT,k1w)>>>1);
    k2v=rhs_v(v2,w2,current_t);k2w=rhs_w(v2,w2);
    v3=v_reg+(qmul(DT,k2v)>>>1);w3=w_reg+(qmul(DT,k2w)>>>1);
    k3v=rhs_v(v3,w3,current_t);k3w=rhs_w(v3,w3);
    v4=v_reg+qmul(DT,k3v);w4=w_reg+qmul(DT,k3w);
    k4v=rhs_v(v4,w4,current_t);k4w=rhs_w(v4,w4);
    v_candidate=v_reg+qmul(DT,($signed(k1v)+($signed(k2v)<<<1)+($signed(k3v)<<<1)+$signed(k4v))/6);
    w_candidate=w_reg+qmul(DT,($signed(k1w)+($signed(k2w)<<<1)+($signed(k3w)<<<1)+$signed(k4w))/6);
end
always @(posedge clk) begin
    if(!rst_n) begin v_reg<=0;w_reg<=0;v_out<=0;w_out<=0;event_out<=0;end
    else begin
        event_out<=($signed(v_reg)<$signed(V_PEAK))&&($signed(v_candidate)>=$signed(V_PEAK));
        v_reg<=v_candidate;w_reg<=w_candidate;v_out<=v_candidate;w_out<=w_candidate;
    end
end
endmodule
