# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DVS → AER bridge

"""Dynamic Vision Sensor (DVS) to Address-Event Representation (AER) bridge.

Generates Verilog modules to interface event cameras with spike networks.
"""

from __future__ import annotations


def generate_dvs_aer_bridge(
    module_name: str = "sc_dvs_aer_bridge",
    *,
    addr_width: int = 16,
    polarity_bit: bool = True,
    timestamp_width: int = 32,
    fifo_depth: int = 64,
) -> str:
    """Generate a DVS (Dynamic Vision Sensor) to AER bridge in Verilog.

    Converts Prophesee / Metavision / Sony IMX636 event packets into
    the SC-NeuroCore AER address-event protocol for zero-copy sensor-
    to-spike-network interfacing on FPGA.

    Parameters
    ----------
    module_name : str
        Output module name.
    addr_width : int
        Pixel address width (covers X*Y event space).
    polarity_bit : bool
        Include ON/OFF polarity in the event word.
    timestamp_width : int
        Timestamp field width in bits.
    fifo_depth : int
        Input event FIFO depth (power of 2).

    Returns
    -------
    str
        Synthesisable Verilog module.
    """
    total_w = addr_width + (1 if polarity_bit else 0) + timestamp_width
    fifo_addr_w = max(1, (fifo_depth - 1).bit_length())

    return f"""// Auto-generated DVS → AER bridge: {module_name}
// SC-NeuroCore event-camera integration
// Addr: {addr_width}b, Polarity: {polarity_bit}, Timestamp: {timestamp_width}b

module {module_name} (
    input  wire                     clk,
    input  wire                     rst,

    // DVS event input (streaming)
    input  wire                     dvs_valid,
    output wire                     dvs_ready,
    input  wire [{addr_width - 1}:0]         dvs_addr,
    input  wire                     dvs_polarity,
    input  wire [{timestamp_width - 1}:0]    dvs_timestamp,

    // AER output (to spike network)
    output wire                     aer_req,
    input  wire                     aer_ack,
    output wire [{addr_width - 1}:0]         aer_addr,
    output wire                     aer_polarity,
    output wire [{timestamp_width - 1}:0]    aer_timestamp,

    // Status
    output wire [{fifo_addr_w}:0]            fifo_count,
    output wire                     fifo_overflow
);

    // ── FIFO storage ────────────────────────────────────────
    reg [{total_w - 1}:0] fifo_mem [0:{fifo_depth - 1}];
    reg [{fifo_addr_w - 1}:0] wr_ptr, rd_ptr;
    reg [{fifo_addr_w}:0] count;
    reg overflow_r;

    wire fifo_full  = (count == {fifo_depth});
    wire fifo_empty = (count == 0);

    assign dvs_ready     = ~fifo_full;
    assign fifo_count    = count;
    assign fifo_overflow = overflow_r;

    // ── Write side (DVS input) ──────────────────────────────
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            wr_ptr     <= 0;
            overflow_r <= 1'b0;
        end else if (dvs_valid && dvs_ready) begin
            fifo_mem[wr_ptr] <= {{dvs_polarity, dvs_timestamp, dvs_addr}};
            wr_ptr <= wr_ptr + 1'b1;
        end else if (dvs_valid && fifo_full) begin
            overflow_r <= 1'b1;
        end
    end

    // ── Read side (AER output) ──────────────────────────────
    reg aer_req_r;
    reg [{total_w - 1}:0] aer_data_r;

    assign aer_req       = aer_req_r;
    assign aer_addr      = aer_data_r[{addr_width - 1}:0];
    assign aer_timestamp = aer_data_r[{addr_width + timestamp_width - 1}:{addr_width}];
    assign aer_polarity  = aer_data_r[{total_w - 1}];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rd_ptr    <= 0;
            aer_req_r <= 1'b0;
        end else begin
            if (aer_req_r && aer_ack) begin
                aer_req_r <= 1'b0;
                rd_ptr    <= rd_ptr + 1'b1;
            end else if (!aer_req_r && !fifo_empty) begin
                aer_data_r <= fifo_mem[rd_ptr];
                aer_req_r  <= 1'b1;
            end
        end
    end

    // ── Count tracker ───────────────────────────────────────
    always @(posedge clk or posedge rst) begin
        if (rst)
            count <= 0;
        else begin
            case ({{(dvs_valid && dvs_ready), (aer_req_r && aer_ack)}})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: ;
            endcase
        end
    end

endmodule
"""
