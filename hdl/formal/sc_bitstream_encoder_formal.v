`default_nettype none

module sc_bitstream_encoder_formal(
    input wire clk,
    input wire rst_n,
    input wire [15:0] x_value,
    input wire [31:0] t_index
);

    wire bit_out;

    // Instantiate the module under test
    sc_bitstream_encoder #(
        .DATA_WIDTH(16),
        .LFSR_WIDTH(16),
        .SEED_INIT(16'hACE1)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .x_value(x_value),
        .t_index(t_index),
        .bit_out(bit_out)
    );

`ifdef FORMAL
    // Initial state tracking
    reg init = 1;
    always @(posedge clk) begin
        if (init) init <= 0;
        
        // 1. Invariant: LFSR must never lock into the all-zero state
        // The LFSR uses XOR feedback which gets stuck at 0 if it ever reaches 0.
        if (!init && rst_n) begin
            assert(uut.lfsr_reg != 16'h0000);
        end
        
        // 2. Invariant: If x_value is exactly 0, the output must be 0
        // (probability of spiking is 0)
        if (!init && rst_n && x_value == 16'h0000) begin
            assert(bit_out == 1'b0);
        end

        // 3. Cover: Ensure it's possible to generate a spike
        if (!init && rst_n) begin
            cover(bit_out == 1'b1);
        end
    end
`endif

endmodule
