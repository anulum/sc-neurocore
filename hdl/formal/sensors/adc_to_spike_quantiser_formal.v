// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - formal contract for ADC-to-spike quantiser.

`default_nettype none

module adc_to_spike_quantiser_formal (
    input wire clk
);
    reg rst_n = 1'b0;
    reg past_valid = 1'b0;
    reg [4:0] cycle = 5'd0;

    reg signed [3:0] adc_sample;
    reg adc_valid;
    wire adc_ready;
    wire [3:0] aer_address;
    wire aer_valid;
    reg aer_ready;
    wire aer_polarity;
    wire dropped_sample;
    wire threshold_error;
    wire [31:0] sample_count;
    wire [31:0] spike_count;
    wire signed [3:0] last_window_q;
    wire window_complete;
    wire signed [3:0] window_average_q;
    wire [31:0] window_spike_budget;
    wire [31:0] pending_spikes;

    adc_to_spike_quantiser #(
        .ADC_WIDTH(4),
        .Q_INT(2),
        .Q_FRAC(2),
        .AER_ADDR_WIDTH(4),
        .DECIMATION(2),
        .SIGNED_INPUT(1),
        .BASE_ADDRESS(4'd4),
        .NEGATIVE_OFFSET(4'd1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .adc_sample(adc_sample),
        .adc_valid(adc_valid),
        .adc_ready(adc_ready),
        .threshold_q(4'd2),
        .aer_address(aer_address),
        .aer_valid(aer_valid),
        .aer_ready(aer_ready),
        .aer_polarity(aer_polarity),
        .dropped_sample(dropped_sample),
        .threshold_error(threshold_error),
        .sample_count(sample_count),
        .spike_count(spike_count),
        .last_window_q(last_window_q),
        .window_complete(window_complete),
        .window_average_q(window_average_q),
        .window_spike_budget(window_spike_budget),
        .pending_spikes(pending_spikes)
    );

    always @* begin
        adc_sample = 4'sd0;
        adc_valid = 1'b0;
        aer_ready = 1'b1;
        if (rst_n) begin
            case (cycle)
                5'd1: begin
                    adc_sample = 4'sd2;
                    adc_valid = adc_ready;
                end
                5'd2: begin
                    adc_sample = 4'sd2;
                    adc_valid = adc_ready;
                end
                5'd5: begin
                    adc_sample = -4'sd4;
                    adc_valid = adc_ready;
                end
                5'd6: begin
                    adc_sample = -4'sd4;
                    adc_valid = adc_ready;
                end
                5'd7,
                5'd8: begin
                    aer_ready = 1'b0;
                end
                default: begin
                    adc_valid = 1'b0;
                end
            endcase
        end
    end

    always @(posedge clk) begin
        past_valid <= 1'b1;
        rst_n <= past_valid;
        if (!rst_n) begin
            cycle <= 5'd0;
        end else if (cycle != 5'd31) begin
            cycle <= cycle + 5'd1;
        end
    end

    always @(posedge clk) begin
        if (rst_n && past_valid) begin
            assert(!threshold_error);
            assert(!dropped_sample);
            assert(pending_spikes <= 32'd4);
            if (aer_valid) begin
                assert(aer_address == (aer_polarity ? 4'd5 : 4'd4));
            end
            if ($past(rst_n) && $past(window_complete)) begin
                assert(last_window_q == $past(window_average_q));
                assert(pending_spikes == $past(window_spike_budget) || spike_count > $past(spike_count));
            end
            if (aer_valid && (last_window_q < 4'sd0)) begin
                assert(aer_polarity);
            end
            if (aer_valid && (last_window_q >= 4'sd0)) begin
                assert(!aer_polarity);
            end
            cover(last_window_q == 4'sd2);
            cover(last_window_q == -4'sd4);
            cover(aer_valid && !aer_ready);
            cover(spike_count >= 32'd3);
        end
    end
endmodule

`default_nettype wire
