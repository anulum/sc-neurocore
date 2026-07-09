// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore - ADC-to-spike quantiser for AER sensor ingress.

`default_nettype none

module adc_to_spike_quantiser #(
    parameter integer ADC_WIDTH      = 16,
    parameter integer SAMPLE_RATE_HZ = 1000000000,
    parameter integer Q_INT          = 8,
    parameter integer Q_FRAC         = 8,
    parameter integer AER_ADDR_WIDTH = 16,
    parameter integer DECIMATION     = 8,
    parameter integer SIGNED_INPUT   = 1,
    parameter [AER_ADDR_WIDTH-1:0] BASE_ADDRESS = {AER_ADDR_WIDTH{1'b0}},
    parameter [AER_ADDR_WIDTH-1:0] NEGATIVE_OFFSET = {{(AER_ADDR_WIDTH-1){1'b0}}, 1'b1}
)(
    input  wire clk,
    input  wire rst_n,

    input  wire signed [ADC_WIDTH-1:0] adc_sample,
    input  wire                        adc_valid,
    output wire                        adc_ready,

    input  wire [Q_INT+Q_FRAC-1:0]     threshold_q,

    output wire [AER_ADDR_WIDTH-1:0]   aer_address,
    output wire                        aer_valid,
    input  wire                        aer_ready,
    output wire                        aer_polarity,

    output reg                         dropped_sample,
    output reg                         threshold_error,
    output reg [31:0]                  sample_count,
    output reg [31:0]                  spike_count,
    output reg signed [Q_INT+Q_FRAC-1:0] last_window_q,
    output wire                        window_complete,
    output wire signed [Q_INT+Q_FRAC-1:0] window_average_q,
    output wire [31:0]                 window_spike_budget,
    output wire [31:0]                 pending_spikes
);
    localparam integer Q_TOTAL = Q_INT + Q_FRAC;
    localparam integer DECIM_COUNTER_WIDTH = (DECIMATION <= 1) ? 1 : $clog2(DECIMATION);
    localparam integer SUM_WIDTH = Q_TOTAL + DECIM_COUNTER_WIDTH + 3;
    localparam integer WIDE_WIDTH = (ADC_WIDTH > Q_TOTAL) ? (ADC_WIDTH + 3) : (Q_TOTAL + 3);

    localparam signed [Q_TOTAL-1:0] Q_MAX = {1'b0, {(Q_TOTAL-1){1'b1}}};
    localparam signed [Q_TOTAL-1:0] Q_MIN = {1'b1, {(Q_TOTAL-1){1'b0}}};
    localparam [DECIM_COUNTER_WIDTH-1:0] DECIMATION_LAST = DECIM_COUNTER_WIDTH'(DECIMATION - 1);
    localparam signed [SUM_WIDTH-1:0] DECIMATION_HALF_SIGNED = SUM_WIDTH'(DECIMATION / 2);
    localparam signed [SUM_WIDTH-1:0] DECIMATION_SIGNED = SUM_WIDTH'(DECIMATION);
    wire sample_rate_configured;

    reg [DECIM_COUNTER_WIDTH-1:0] decim_count;
    reg signed [SUM_WIDTH-1:0] window_sum_q;
    reg [31:0] pending_spike_count;
    reg pending_polarity;

    wire accepted_sample;
    wire accepted_spike;
    wire signed [Q_TOTAL-1:0] quantised_sample_q;
    wire signed [SUM_WIDTH-1:0] next_window_sum_q;
    wire [Q_TOTAL-1:0] abs_window_q;

    assign accepted_sample = adc_valid && adc_ready;
    assign accepted_spike = aer_valid && aer_ready;
    assign sample_rate_configured = (SAMPLE_RATE_HZ > 0);
    assign adc_ready = (pending_spike_count == 32'd0) && (threshold_q != {Q_TOTAL{1'b0}}) && sample_rate_configured;
    assign aer_valid = (pending_spike_count != 32'd0);
    assign aer_polarity = pending_polarity;
    assign aer_address = pending_polarity ? (BASE_ADDRESS + NEGATIVE_OFFSET) : BASE_ADDRESS;
    assign pending_spikes = pending_spike_count;
    assign quantised_sample_q = quantise_adc(adc_sample);
    assign next_window_sum_q = window_sum_q + {{(SUM_WIDTH-Q_TOTAL){quantised_sample_q[Q_TOTAL-1]}}, quantised_sample_q};
    assign window_complete = accepted_sample && (decim_count == DECIMATION_LAST);
    assign window_average_q = average_window(next_window_sum_q);
    assign abs_window_q = abs_q(window_average_q);
    assign window_spike_budget = (threshold_q == {Q_TOTAL{1'b0}})
        ? 32'd0
        : ({{(32-Q_TOTAL){1'b0}}, abs_window_q} / {{(32-Q_TOTAL){1'b0}}, threshold_q});

    function signed [Q_TOTAL-1:0] clamp_q;
        input signed [WIDE_WIDTH-1:0] value;
        begin
            if (value > {{(WIDE_WIDTH-Q_TOTAL){Q_MAX[Q_TOTAL-1]}}, Q_MAX}) begin
                clamp_q = Q_MAX;
            end else if (value < {{(WIDE_WIDTH-Q_TOTAL){Q_MIN[Q_TOTAL-1]}}, Q_MIN}) begin
                clamp_q = Q_MIN;
            end else begin
                clamp_q = value[Q_TOTAL-1:0];
            end
        end
    endfunction

    function signed [Q_TOTAL-1:0] quantise_adc;
        input signed [ADC_WIDTH-1:0] sample;
        reg signed [WIDE_WIDTH-1:0] centred;
        reg signed [WIDE_WIDTH-1:0] rounded;
        integer shift;
        begin
            if (SIGNED_INPUT != 0) begin
                centred = {{(WIDE_WIDTH-ADC_WIDTH){sample[ADC_WIDTH-1]}}, sample};
            end else begin
                centred = {{(WIDE_WIDTH-ADC_WIDTH){1'b0}}, sample}
                    - ({{(WIDE_WIDTH-1){1'b0}}, 1'b1} <<< (ADC_WIDTH - 1));
            end

            if (Q_TOTAL > ADC_WIDTH) begin
                rounded = centred <<< (Q_TOTAL - ADC_WIDTH);
            end else if (ADC_WIDTH > Q_TOTAL) begin
                shift = ADC_WIDTH - Q_TOTAL;
                if (centred >= $signed({WIDE_WIDTH{1'b0}})) begin
                    rounded = (centred + ({{(WIDE_WIDTH-1){1'b0}}, 1'b1} <<< (shift - 1))) >>> shift;
                end else begin
                    rounded = (centred - ({{(WIDE_WIDTH-1){1'b0}}, 1'b1} <<< (shift - 1))) >>> shift;
                end
            end else begin
                rounded = centred;
            end
            quantise_adc = clamp_q(rounded);
        end
    endfunction

    function signed [Q_TOTAL-1:0] clamp_sum_to_q;
        input signed [SUM_WIDTH-1:0] value;
        begin
            if (value > {{(SUM_WIDTH-Q_TOTAL){Q_MAX[Q_TOTAL-1]}}, Q_MAX}) begin
                clamp_sum_to_q = Q_MAX;
            end else if (value < {{(SUM_WIDTH-Q_TOTAL){Q_MIN[Q_TOTAL-1]}}, Q_MIN}) begin
                clamp_sum_to_q = Q_MIN;
            end else begin
                clamp_sum_to_q = value[Q_TOTAL-1:0];
            end
        end
    endfunction

    function signed [Q_TOTAL-1:0] average_window;
        input signed [SUM_WIDTH-1:0] sum_q;
        reg signed [SUM_WIDTH-1:0] adjusted;
        begin
            if (sum_q >= $signed({SUM_WIDTH{1'b0}})) begin
                adjusted = sum_q + DECIMATION_HALF_SIGNED;
            end else begin
                adjusted = sum_q - DECIMATION_HALF_SIGNED;
            end
            average_window = clamp_sum_to_q(adjusted / DECIMATION_SIGNED);
        end
    endfunction

    function [Q_TOTAL-1:0] abs_q;
        input signed [Q_TOTAL-1:0] value;
        begin
            if (value == Q_MIN) begin
                abs_q = Q_MAX[Q_TOTAL-1:0];
            end else if (value < $signed({Q_TOTAL{1'b0}})) begin
                abs_q = -value;
            end else begin
                abs_q = value[Q_TOTAL-1:0];
            end
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            decim_count <= {DECIM_COUNTER_WIDTH{1'b0}};
            window_sum_q <= {SUM_WIDTH{1'b0}};
            pending_spike_count <= 32'd0;
            pending_polarity <= 1'b0;
            dropped_sample <= 1'b0;
            threshold_error <= 1'b0;
            sample_count <= 32'd0;
            spike_count <= 32'd0;
            last_window_q <= {Q_TOTAL{1'b0}};
        end else begin
            if (threshold_q == {Q_TOTAL{1'b0}}) begin
                threshold_error <= 1'b1;
            end

            if (adc_valid && !adc_ready) begin
                dropped_sample <= 1'b1;
            end

            if (accepted_spike) begin
                pending_spike_count <= pending_spike_count - 32'd1;
                spike_count <= spike_count + 32'd1;
            end

            if (accepted_sample) begin
                sample_count <= sample_count + 32'd1;
                if (window_complete) begin
                    last_window_q <= window_average_q;
                    pending_spike_count <= window_spike_budget;
                    pending_polarity <= window_average_q[Q_TOTAL-1];
                    window_sum_q <= {SUM_WIDTH{1'b0}};
                    decim_count <= {DECIM_COUNTER_WIDTH{1'b0}};
                end else begin
                    window_sum_q <= next_window_sum_q;
                    decim_count <= decim_count + {{(DECIM_COUNTER_WIDTH-1){1'b0}}, 1'b1};
                end
            end
        end
    end

endmodule

`default_nettype wire
