// Auto-generated testbench for sc_equation_neuron
// SC-NeuroCore equation compiler
`timescale 1ns / 1ps

module tb_sc_equation_neuron;

reg clk;
reg rst_n;
wire spike_out;
wire signed [15:0] v_out;

sc_equation_neuron uut (
    .clk(clk),
    .rst_n(rst_n),
    .I_t(16'sd256),
    .spike_out(spike_out),
    .v_out(v_out)
);

// Clock: 10ns period (100 MHz)
initial clk = 0;
always #5 clk = ~clk;

integer spike_count;

initial begin
    $dumpfile("tb_sc_equation_neuron.vcd");
    $dumpvars(0, tb_sc_equation_neuron);
    spike_count = 0;

    // Reset
    rst_n = 0;
    #20;
    rst_n = 1;

    // Run 200 cycles
    repeat (200) begin
        @(posedge clk);
        if (spike_out) spike_count = spike_count + 1;
    end

    $display("Simulation complete: %0d spikes in 200 cycles", spike_count);
    $display("Final v = %0d (Q8.8)", v_out);
    $finish;
end

endmodule
