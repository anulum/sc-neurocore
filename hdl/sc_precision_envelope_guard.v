
`timescale 1ns / 1ps

module sc_precision_envelope_guard #(
    parameter integer N_OUTPUTS = 32,
    parameter integer OUTPUT_WIDTH = 32,
    parameter integer BOUND_WIDTH = 48
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [N_OUTPUTS*BOUND_WIDTH-1:0] abs_bounds_q,
    output reg valid_out,
    output reg [N_OUTPUTS-1:0] violation_vector,
    output wire envelope_violation
);

localparam [BOUND_WIDTH-1:0] MAX_SAFE_BOUND =
    {{(BOUND_WIDTH-OUTPUT_WIDTH){1'b0}}, 1'b0, {OUTPUT_WIDTH-1{1'b1}}};

assign envelope_violation = |violation_vector;

integer output_idx;
integer bound_offset;
reg [BOUND_WIDTH-1:0] bound_lane;
reg [N_OUTPUTS-1:0] violation_next;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_out <= 1'b0;
        violation_vector <= {N_OUTPUTS{1'b0}};
    end else begin
        violation_next = {N_OUTPUTS{1'b0}};
        if (valid_in) begin
            for (output_idx = 0; output_idx < N_OUTPUTS; output_idx = output_idx + 1) begin
                bound_offset = output_idx * BOUND_WIDTH;
                bound_lane = abs_bounds_q[bound_offset +: BOUND_WIDTH];
                violation_next[output_idx] = (bound_lane > MAX_SAFE_BOUND);
            end
        end
        valid_out <= valid_in;
        violation_vector <= violation_next;
    end
end

endmodule
