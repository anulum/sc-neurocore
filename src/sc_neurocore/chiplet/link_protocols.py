# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet CDC and protected-link RTL

"""Clock-domain, frame-protection, and credit-flow contracts for package links."""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass

from sc_neurocore.chiplet._sv import SPDX_HEADER, _require_sv_identifier
from sc_neurocore.chiplet.topology import ChipletTopology


@dataclass
class CDCConfig:
    """Describe asynchronous crossing parameters for one directed link."""

    src_clk_mhz: float
    dst_clk_mhz: float
    fifo_depth_log2: int = 4
    sync_stages: int = 2

    def __post_init__(self) -> None:
        """Validate clock, FIFO-depth, and synchronizer-stage boundaries."""
        if not math.isfinite(self.src_clk_mhz) or self.src_clk_mhz < 0:
            raise ValueError("src_clk_mhz must be finite and >= 0")
        if not math.isfinite(self.dst_clk_mhz) or self.dst_clk_mhz < 0:
            raise ValueError("dst_clk_mhz must be finite and >= 0")
        if self.fifo_depth_log2 <= 0 or self.sync_stages < 2:
            raise ValueError("fifo_depth_log2 must be > 0 and sync_stages must be >= 2")

    @property
    def ratio(self) -> float:
        """Return source-to-destination clock ratio, or one for a zero destination clock."""
        return 1.0 if self.dst_clk_mhz == 0 else self.src_clk_mhz / self.dst_clk_mhz

    @property
    def is_mesochronous(self) -> bool:
        """Return whether source and destination clocks differ by less than one percent."""
        return abs(self.ratio - 1.0) < 0.01


def compute_cdc_configs(topology: ChipletTopology) -> dict[tuple[int, int], CDCConfig]:
    """Derive per-link CDC settings from topology die clocks."""
    configs: dict[tuple[int, int], CDCConfig] = {}
    for link in topology.links:
        source = topology.get_die(link.src_die)
        destination = topology.get_die(link.dst_die)
        if source is None or destination is None:
            continue
        configs[(link.src_die, link.dst_die)] = CDCConfig(
            src_clk_mhz=source.clock_mhz,
            dst_clk_mhz=destination.clock_mhz,
            fifo_depth_log2=link.fifo_depth_log2,
            sync_stages=3 if source.clock_mhz != destination.clock_mhz else 2,
        )
    return configs


@dataclass
class LinkProtection:
    """Describe frame-integrity overhead for one die-to-die link."""

    mode: str = "crc32"
    overhead_bits: int = 0

    def __post_init__(self) -> None:
        """Validate the protection mode and derive its frame overhead."""
        overhead = {"none": 0, "parity": 1, "crc8": 8, "crc32": 32, "secded": 8}
        try:
            self.overhead_bits = overhead[self.mode]
        except KeyError as error:
            raise ValueError(f"unsupported link-protection mode: {self.mode}") from error

    @property
    def effective_bandwidth_ratio(self) -> float:
        """Return payload bits divided by payload plus protection bits."""
        return 1.0 if self.overhead_bits == 0 else 64.0 / (64.0 + self.overhead_bits)


def emit_crc32_sv(data_width: int = 64) -> str:
    """Emit an IEEE 802.3 CRC-32 frame checker for ``data_width`` bits."""
    if data_width <= 0:
        raise ValueError("data_width must be a positive integer")
    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — CRC-32 link checker

module sc_chiplet_crc32 #(
    parameter DATA_W             = {data_width},
    parameter CRC32_POLY_NORMAL  = 32'h04C11DB7,
    parameter CRC32_POLY_REFLECT = 32'hEDB88320,
    parameter REFLECT_INPUT      = 1'b1
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               crc_init,
    input  wire [DATA_W-1:0]  data_in,
    input  wire               data_valid,
    input  wire [31:0]        expected_crc,
    input  wire               crc_check,
    output reg  [31:0]        crc_out,
    output reg                crc_valid,
    output reg                crc_error
);

    reg [31:0] crc_reg;
    wire [31:0] crc_next;
    wire [31:0] crc_candidate;
    wire [31:0] crc_compare_value;

    function automatic [31:0] crc32_update;
        input [31:0] crc;
        input [DATA_W-1:0] data;
        reg [31:0] next_crc;
        integer bit_idx;
        begin
            next_crc = crc;
            for (bit_idx = 0; bit_idx < DATA_W; bit_idx = bit_idx + 1) begin
                if (REFLECT_INPUT) begin
                    if (next_crc[0] ^ data[bit_idx])
                        next_crc = {{1'b0, next_crc[31:1]}} ^ CRC32_POLY_REFLECT;
                    else
                        next_crc = {{1'b0, next_crc[31:1]}};
                end else begin
                    if (next_crc[31] ^ data[DATA_W-1-bit_idx])
                        next_crc = {{next_crc[30:0], 1'b0}} ^ CRC32_POLY_NORMAL;
                    else
                        next_crc = {{next_crc[30:0], 1'b0}};
                end
            end
            crc32_update = next_crc;
        end
    endfunction

    assign crc_next = crc32_update(crc_reg, data_in);
    assign crc_candidate = data_valid ? crc_next : crc_reg;
    assign crc_compare_value = crc_candidate ^ 32'hFFFFFFFF;

    always @(posedge clk) begin
        if (!rst_n) begin
            crc_reg   <= 32'hFFFFFFFF;
            crc_out   <= 32'h00000000;
            crc_valid <= 1'b0;
            crc_error <= 1'b0;
        end else if (crc_init) begin
            crc_reg   <= 32'hFFFFFFFF;
            crc_out   <= 32'h00000000;
            crc_valid <= 1'b0;
            crc_error <= 1'b0;
        end else begin
            if (data_valid) begin
                crc_reg <= crc_next;
                crc_out <= crc_next ^ 32'hFFFFFFFF;
            end
            crc_valid <= data_valid || crc_check;
            if (crc_check) begin
                crc_error <= (crc_compare_value != expected_crc);
                crc_out   <= crc_compare_value;
            end
        end
    end

endmodule
""")


@dataclass
class CreditConfig:
    """Configure receiver-buffer credits for a package link."""

    initial_credits: int = 16
    credit_granularity: int = 1

    def __post_init__(self) -> None:
        """Validate positive credit-count and credit-granularity boundaries."""
        if self.initial_credits <= 0:
            raise ValueError("initial_credits must be > 0")
        if self.credit_granularity <= 0:
            raise ValueError("credit_granularity must be > 0")

    @property
    def buffer_flits(self) -> int:
        """Return total receiver capacity represented by the credit counter."""
        return self.initial_credits * self.credit_granularity

    @property
    def credit_width(self) -> int:
        """Return counter width sufficient to represent the full buffer."""
        return max(1, self.buffer_flits.bit_length())


def emit_credit_controller_sv(config: CreditConfig, link_name: str = "link") -> str:
    """Emit a saturating credit controller for ``link_name``."""
    _require_sv_identifier(link_name, "link_name")
    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Credit controller for {link_name}

module sc_chiplet_credit_{link_name} #(
    parameter INIT_CREDITS = {config.initial_credits},
    parameter MAX_CREDITS = {config.initial_credits},
    parameter CREDIT_GRANULARITY = {config.credit_granularity},
    parameter MAX_FLITS = {config.buffer_flits},
    parameter CREDIT_W = {config.credit_width},
    parameter DATA_W = 64
)(
    input  wire               clk,
    input  wire               rst_n,
    // TX side
    input  wire [DATA_W-1:0]  tx_data,
    input  wire               tx_valid,
    output wire               tx_ready,
    // RX credit return
    input  wire               credit_return,
    output reg  [CREDIT_W-1:0] credits_available
);

    wire consume_credit = tx_valid && tx_ready;
    wire return_credit  = credit_return;
    reg [CREDIT_W:0] next_credits;

    always @* begin
        next_credits = {{1'b0, credits_available}};
        if (consume_credit && next_credits != 0)
            next_credits = next_credits - 1'b1;
        if (return_credit)
            next_credits = next_credits + CREDIT_GRANULARITY;
        if (next_credits > MAX_FLITS)
            next_credits = MAX_FLITS;
    end

    always @(posedge clk) begin
        if (!rst_n)
            credits_available <= MAX_FLITS;
        else
            credits_available <= next_credits[CREDIT_W-1:0];
    end

    assign tx_ready = (credits_available != 0);

endmodule
""")


__all__ = [
    "CDCConfig",
    "CreditConfig",
    "LinkProtection",
    "compute_cdc_configs",
    "emit_crc32_sv",
    "emit_credit_controller_sv",
]
