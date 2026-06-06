`ifndef SC_TIMING_ASSERTIONS_SVH
`define SC_TIMING_ASSERTIONS_SVH

`define SC_ASSERT_LATENCY_LE(NAME, CLK, RST_N, START_EVENT, END_EVENT, BOUND_CYCLES) \
    wire NAME``_violation; \
    wire NAME``_active; \
    wire [31:0] NAME``_age; \
    sc_latency_monitor #( \
        .MAX_CYCLES(BOUND_CYCLES), \
        .COUNTER_WIDTH(32) \
    ) NAME``_latency_monitor ( \
        .clk(CLK), \
        .rst_n(RST_N), \
        .start_event(START_EVENT), \
        .end_event(END_EVENT), \
        .violation(NAME``_violation), \
        .active(NAME``_active), \
        .age(NAME``_age) \
    ); \
    always @(posedge CLK) begin \
        if (RST_N) begin \
            assert (!NAME``_violation); \
            cover (NAME``_active || (|NAME``_age)); \
        end \
    end

`define SC_ASSERT_DEADLINE_LE(NAME, CLK, RST_N, START_EVENT, END_EVENT, BOUND_CYCLES) \
    wire NAME``_violation; \
    wire NAME``_active; \
    wire [31:0] NAME``_age; \
    sc_deadline_monitor #( \
        .DEADLINE_CYCLES(BOUND_CYCLES), \
        .COUNTER_WIDTH(32) \
    ) NAME``_deadline_monitor ( \
        .clk(CLK), \
        .rst_n(RST_N), \
        .deadline_start(START_EVENT), \
        .completion_event(END_EVENT), \
        .violation(NAME``_violation), \
        .active(NAME``_active), \
        .age(NAME``_age) \
    ); \
    always @(posedge CLK) begin \
        if (RST_N) begin \
            assert (!NAME``_violation); \
            cover (NAME``_active || (|NAME``_age)); \
        end \
    end

`define SC_ASSERT_BOUNDED_LIVENESS(NAME, CLK, RST_N, REQUEST_EVENT, WITNESS_EVENT, BOUND_CYCLES) \
    wire NAME``_violation; \
    wire NAME``_active; \
    wire [31:0] NAME``_age; \
    sc_bounded_liveness_monitor #( \
        .WINDOW_CYCLES(BOUND_CYCLES), \
        .COUNTER_WIDTH(32) \
    ) NAME``_liveness_monitor ( \
        .clk(CLK), \
        .rst_n(RST_N), \
        .request_event(REQUEST_EVENT), \
        .witness_event(WITNESS_EVENT), \
        .violation(NAME``_violation), \
        .active(NAME``_active), \
        .age(NAME``_age) \
    ); \
    always @(posedge CLK) begin \
        if (RST_N) begin \
            assert (!NAME``_violation); \
            cover (NAME``_active || (|NAME``_age)); \
        end \
    end

`endif
