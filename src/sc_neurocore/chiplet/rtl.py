# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-die AER and AXI-Stream RTL generator

"""Emit die wrappers, CDC bridges, routing tables, package RTL, and XDC."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

from sc_neurocore.chiplet._sv import SPDX_HEADER
from sc_neurocore.chiplet.routing import RoutingTable, compute_decorrelation_seeds
from sc_neurocore.chiplet.topology import ChipletDie, ChipletTopology, InterposerLink


@dataclass
class ChipletOutput:
    """Contain generated source and constraint artefacts for one topology."""

    top_sv: str
    die_modules: dict[int, str]
    link_bridges: dict[tuple[int, int], str]
    routing_tables: dict[int, str]
    constraints_xdc: str
    filelist: list[str]

    def to_dict(self) -> dict[str, str]:
        """Return generated artefacts keyed by their output filename."""
        artefacts = {"sc_chiplet_top.sv": self.top_sv}
        artefacts.update(
            {f"sc_chiplet_die_{die_id}.sv": source for die_id, source in self.die_modules.items()}
        )
        artefacts.update(
            {
                f"sc_chiplet_bridge_{source}_to_{destination}.sv": text
                for (source, destination), text in self.link_bridges.items()
            }
        )
        artefacts.update(
            {
                f"sc_chiplet_rtable_{die_id}.sv": source
                for die_id, source in self.routing_tables.items()
            }
        )
        artefacts["chiplet_constraints.xdc"] = self.constraints_xdc
        return artefacts


class ChipletGenerator:
    """Generate connected multi-die routing RTL from a chiplet topology."""

    def generate(
        self,
        topology: ChipletTopology,
        routing: dict[int, RoutingTable] | None = None,
    ) -> ChipletOutput:
        """Generate package RTL and timing constraints.

        Parameters
        ----------
        topology
            Directed die and link graph.
        routing
            Optional per-die AER routing tables.

        Returns
        -------
        ChipletOutput
            Generated top, die, bridge, route, and constraint artefacts.
        """
        seeds = compute_decorrelation_seeds(topology)
        die_modules: dict[int, str] = {}
        link_bridges: dict[tuple[int, int], str] = {}
        routing_tables: dict[int, str] = {}
        for die in topology.dies:
            die_modules[die.die_id] = self._emit_die_wrapper(die, topology)
            if routing and die.die_id in routing:
                routing_tables[die.die_id] = self._emit_routing_table(routing[die.die_id], die)
        for link in topology.links:
            seed = seeds.get((link.src_die, link.dst_die), 0xACE1)
            link_bridges[(link.src_die, link.dst_die)] = self._emit_bridge(link, seed)
        output = ChipletOutput(
            self._emit_top(topology),
            die_modules,
            link_bridges,
            routing_tables,
            self._emit_constraints(topology),
            [],
        )
        output.filelist = list(output.to_dict())
        return output

    def _emit_die_wrapper(self, die: ChipletDie, topology: ChipletTopology) -> str:
        outgoing = topology.get_links_from(die.die_id)
        incoming = topology.get_links_to(die.die_id)
        out_ports = "\n".join(
            f"    output wire [{link.data_width - 1}:0] link_out_{link.dst_die}_tdata,\n"
            f"    output wire                  link_out_{link.dst_die}_tvalid,\n"
            f"    input  wire                  link_out_{link.dst_die}_tready,"
            for link in outgoing
        )
        in_ports = "\n".join(
            f"    input  wire [{link.data_width - 1}:0] link_in_{link.src_die}_tdata,\n"
            f"    input  wire                  link_in_{link.src_die}_tvalid,\n"
            f"    output wire                  link_in_{link.src_die}_tready,"
            for link in incoming
        )
        out_assigns = "\n".join(
            f"    assign link_out_{link.dst_die}_tdata = "
            f"{{{{({link.data_width}-AER_ID_W){{1'b0}}}}, local_out_id}};\n"
            f"    assign link_out_{link.dst_die}_tvalid = local_out_valid;"
            for link in outgoing
        )
        in_assigns = "\n".join(
            f"    assign link_in_{link.src_die}_tready = 1'b1;" for link in incoming
        )
        return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Die {die.die_id} wrapper

module sc_chiplet_die_{die.die_id} #(
    parameter N_NEURONS = {die.n_neurons},
    parameter AER_ID_W  = {die.aer_id_width},
    parameter DATA_W    = {die.data_width},
    parameter AER_PRIO_W = 2,
    parameter LFSR_SEED = 16'h{die.lfsr_seed:04X}
)(
    input  wire clk,
    input  wire rst_n,
{out_ports}
{in_ports}
    // Local AER interface
    input  wire                 local_spike_valid,
    input  wire [AER_ID_W-1:0] local_spike_id,
    output wire                 local_out_valid,
    output wire [AER_ID_W-1:0] local_out_id,
    output wire [DATA_W-1:0]   local_out_weight
);

    // LFSR-16 for bitstream decorrelation (unique per die)
    reg [15:0] lfsr = LFSR_SEED;
    always @(posedge clk)
        lfsr <= {{lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]}};

    // Local AER router instance
    sc_aer_router #(
        .N_SRC(N_NEURONS), .N_TGT(N_NEURONS),
        .DATA_WIDTH(DATA_W),
        .PRIO_WIDTH(AER_PRIO_W)
    ) local_router (
        .clk(clk), .rst_n(rst_n),
        .in_event_valid(local_spike_valid),
        .in_event_ready(),
        .in_neuron_id(local_spike_id),
        .in_timestamp(16'd0),
        .in_priority({{AER_PRIO_W{{1'b0}}}}),
        .out_event_valid(local_out_valid),
        .out_event_ready(1'b1),
        .out_target_id(local_out_id),
        .out_weight(local_out_weight),
        .out_timestamp(),
        .out_priority(),
        .busy(),
        .queue_full(),
        .dropped_event(),
        .critical_deadline_violation()
    );

{in_assigns}
{out_assigns}

endmodule
""")

    def _emit_bridge(self, link: InterposerLink, decorrelation_seed: int) -> str:
        fifo_depth = link.fifo_depth_log2
        latency = link.latency_cycles
        return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Bridge die {link.src_die} → die {link.dst_die}
// Technology: {link.technology.value}
// Latency: {link.latency_ns} ns ({latency} cycles @ 200 MHz)
// Bandwidth: {link.bandwidth_gbps} Gb/s
// BER: {link.bit_error_rate}

module sc_chiplet_bridge_{link.src_die}_to_{link.dst_die} #(
    parameter DATA_W       = {link.data_width},
    parameter FIFO_DEPTH   = {fifo_depth},
    parameter LATENCY_CYC  = {latency},
    parameter DECOR_SEED   = 16'h{decorrelation_seed:04X}
)(
    input  wire               src_clk,
    input  wire               src_rst,
    input  wire               dst_clk,
    input  wire               dst_rst,

    // AXI-Stream slave (from source die)
    input  wire [DATA_W-1:0]  s_tdata,
    input  wire               s_tvalid,
    output wire               s_tready,

    // AXI-Stream master (to destination die)
    output wire [DATA_W-1:0]  m_tdata,
    output wire               m_tvalid,
    input  wire               m_tready
);

    // CDC via async FIFO (from sc_cdc_primitives.v)
    wire [DATA_W-1:0] fifo_rd_data;
    wire fifo_rd_empty, fifo_wr_full;

    sc_async_fifo #(
        .DATA_WIDTH(DATA_W),
        .DEPTH_LOG2(FIFO_DEPTH)
    ) cdc_fifo (
        .wr_clk(src_clk), .wr_rst(src_rst),
        .wr_data(s_tdata), .wr_en(s_tvalid && !fifo_wr_full),
        .wr_full(fifo_wr_full),
        .rd_clk(dst_clk), .rd_rst(dst_rst),
        .rd_data(fifo_rd_data), .rd_en(!fifo_rd_empty && m_tready),
        .rd_empty(fifo_rd_empty)
    );

    assign s_tready = !fifo_wr_full;

    // Interposer latency model (shift register)
    reg [DATA_W-1:0] delay_pipe [0:LATENCY_CYC-1];
    reg [LATENCY_CYC-1:0] valid_pipe;
    integer k;

    always @(posedge dst_clk) begin
        if (dst_rst) begin
            valid_pipe <= 0;
            for (k = 0; k < LATENCY_CYC; k = k + 1)
                delay_pipe[k] <= 0;
        end else begin
            delay_pipe[0] <= fifo_rd_data;
            valid_pipe[0] <= !fifo_rd_empty;
            for (k = 1; k < LATENCY_CYC; k = k + 1) begin
                delay_pipe[k] <= delay_pipe[k-1];
                valid_pipe[k] <= valid_pipe[k-1];
            end
        end
    end

    // Bitstream decorrelation: XOR with LFSR output
    reg [15:0] lfsr = DECOR_SEED;
    always @(posedge dst_clk)
        lfsr <= {{lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]}};

    wire [DATA_W-1:0] decorrelated;
    genvar g;
    generate
        for (g = 0; g < DATA_W; g = g + 1) begin : decor
            assign decorrelated[g] = delay_pipe[LATENCY_CYC-1][g] ^ lfsr[g % 16];
        end
    endgenerate

    assign m_tdata  = decorrelated;
    assign m_tvalid = valid_pipe[LATENCY_CYC-1];

endmodule
""")

    def _emit_routing_table(self, table: RoutingTable, die: ChipletDie) -> str:
        entries = [
            f"        rt_target_die[{entry.src_neuron}]    = {entry.dst_die};\n"
            f"        rt_target_neuron[{entry.src_neuron}] = {entry.dst_neuron};\n"
            f"        rt_weight[{entry.src_neuron}]        = 16'sd{entry.weight_q88};"
            for entry in table.entries
        ]
        initialisation = "\n".join(entries) if entries else "        // No inter-die routes"
        n_entries = max(len(table.entries), 1)
        return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Routing table for die {die.die_id}

module sc_chiplet_rtable_{die.die_id} #(
    parameter N_ENTRIES   = {n_entries},
    parameter AER_ID_W    = {die.aer_id_width},
    parameter DIE_ID_W    = 4,
    parameter DATA_W      = {die.data_width}
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire [AER_ID_W-1:0] query_neuron,
    output reg  [DIE_ID_W-1:0] target_die,
    output reg  [AER_ID_W-1:0] target_neuron,
    output reg  signed [DATA_W-1:0] weight
);

    reg [DIE_ID_W-1:0]         rt_target_die    [0:N_ENTRIES-1];
    reg [AER_ID_W-1:0]         rt_target_neuron [0:N_ENTRIES-1];
    reg signed [DATA_W-1:0]    rt_weight        [0:N_ENTRIES-1];

    initial begin
{initialisation}
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            target_die    <= 0;
            target_neuron <= 0;
            weight        <= 0;
        end else begin
            target_die    <= rt_target_die[query_neuron];
            target_neuron <= rt_target_neuron[query_neuron];
            weight        <= rt_weight[query_neuron];
        end
    end

endmodule
""")

    def _emit_top(self, topology: ChipletTopology) -> str:
        link_wires = [
            f"    wire [{link.data_width - 1}:0] link_{link.src_die}_{link.dst_die}_tdata;\n"
            f"    wire                  link_{link.src_die}_{link.dst_die}_tvalid;\n"
            f"    wire                  link_{link.src_die}_{link.dst_die}_tready;\n"
            f"    wire [{link.data_width - 1}:0] link_{link.src_die}_{link.dst_die}_rx_tdata;\n"
            f"    wire                  link_{link.src_die}_{link.dst_die}_rx_tvalid;\n"
            f"    wire                  link_{link.src_die}_{link.dst_die}_rx_tready;"
            for link in topology.links
        ]
        die_instances: list[str] = []
        for die in topology.dies:
            ports: list[str] = []
            for link in topology.get_links_from(die.die_id):
                ports.extend(
                    [
                        f"        .link_out_{link.dst_die}_tdata(link_{link.src_die}_{link.dst_die}_tdata)",
                        f"        .link_out_{link.dst_die}_tvalid(link_{link.src_die}_{link.dst_die}_tvalid)",
                        f"        .link_out_{link.dst_die}_tready(link_{link.src_die}_{link.dst_die}_tready)",
                    ]
                )
            for link in topology.get_links_to(die.die_id):
                ports.extend(
                    [
                        f"        .link_in_{link.src_die}_tdata(link_{link.src_die}_{link.dst_die}_rx_tdata)",
                        f"        .link_in_{link.src_die}_tvalid(link_{link.src_die}_{link.dst_die}_rx_tvalid)",
                        f"        .link_in_{link.src_die}_tready(link_{link.src_die}_{link.dst_die}_rx_tready)",
                    ]
                )
            link_port_block = ",\n" + ",\n".join(ports) if ports else ""
            die_instances.append(
                f"    // Die {die.die_id}\n"
                f"    sc_chiplet_die_{die.die_id} die_{die.die_id}_inst (\n"
                f"        .clk(clk),\n"
                f"        .rst_n(rst_n)"
                f"{link_port_block},\n"
                f"        .local_spike_valid(1'b0),\n"
                f"        .local_spike_id({die.aer_id_width}'d0),\n"
                f"        .local_out_valid(),\n"
                f"        .local_out_id(),\n"
                f"        .local_out_weight()\n"
                f"    );"
            )
        bridge_instances = [
            f"    // Bridge {link.src_die} → {link.dst_die} ({link.technology.value})\n"
            f"    sc_chiplet_bridge_{link.src_die}_to_{link.dst_die} bridge_{link.src_die}_{link.dst_die}_inst (\n"
            f"        .src_clk(clk), .src_rst(!rst_n),\n"
            f"        .dst_clk(clk), .dst_rst(!rst_n),\n"
            f"        .s_tdata(link_{link.src_die}_{link.dst_die}_tdata),\n"
            f"        .s_tvalid(link_{link.src_die}_{link.dst_die}_tvalid),\n"
            f"        .s_tready(link_{link.src_die}_{link.dst_die}_tready),\n"
            f"        .m_tdata(link_{link.src_die}_{link.dst_die}_rx_tdata),\n"
            f"        .m_tvalid(link_{link.src_die}_{link.dst_die}_rx_tvalid),\n"
            f"        .m_tready(link_{link.src_die}_{link.dst_die}_rx_tready)\n"
            f"    );"
            for link in topology.links
        ]
        link_wire_block = "\n".join(link_wires)
        die_block = "\n\n".join(die_instances)
        bridge_block = "\n\n".join(bridge_instances)
        return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Top-level multi-die package
// Dies: {topology.num_dies}
// Links: {len(topology.links)}

module sc_chiplet_top (
    input wire clk,
    input wire rst_n
);

{link_wire_block}

{die_block}

{bridge_block}

endmodule
""")

    def _emit_constraints(self, topology: ChipletTopology) -> str:
        lines = [
            "# SC-NeuroCore Chiplet — Timing constraints",
            "# Auto-generated for multi-die package",
            "",
        ]
        for die in topology.dies:
            lines.extend(
                [
                    f"# Die {die.die_id}: {die.clock_mhz} MHz",
                    f"create_clock -name clk_die_{die.die_id} "
                    f"-period {die.clock_period_ns:.3f} "
                    f"[get_pins die_{die.die_id}_inst/clk]",
                ]
            )
        lines.append("")
        for link in topology.links:
            lines.extend(
                [
                    f"# Link {link.src_die} → {link.dst_die}: {link.latency_ns} ns",
                    f"set_max_delay -from [get_clocks clk_die_{link.src_die}] "
                    f"-to [get_clocks clk_die_{link.dst_die}] {link.latency_ns}",
                ]
            )
        return "\n".join(lines) + "\n"


__all__ = ["ChipletGenerator", "ChipletOutput"]
