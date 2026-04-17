# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet/Interposer SC Routing Generator

"""Auto-generates multi-die AXI-Stream + AER routing for chiplet packages.

Given a chiplet topology (dies, links, partitions from the exascale
partitioner), emits:

- **InterposerLink**: Timing model for die-to-die links (latency, jitter,
  bandwidth, BER) with per-technology presets (UCIe, BoW, EMIB, CoWoS).
- **ChipletDie**: Die abstraction with clock domain, LFSR seed, and
  neuron partition assignment.
- **ChipletTopology**: Directed graph of dies + interposer links.
- **RoutingTable**: Per-die AER routing tables mapping source neuron IDs
  to (target_die, target_neuron) pairs.
- **SystemVerilog Generator**: Emits CDC bridges (async FIFO), AXI-Stream
  routers, AER crossbar switches, and LFSR-based bitstream decorrelators
  for each die-to-die link.

Hooks into:
- ``hdl/sc_aer_router.v`` — AER spike router
- ``hdl/sc_axis_interface.v`` — AXI-Stream bridge
- ``hdl/sc_cdc_primitives.v`` — async FIFO + gray counter + 2FF sync
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ── Interposer Technology ────────────────────────────────────────────


class InterposerTech(Enum):
    UCIE = "UCIe"
    BOW = "BoW"
    EMIB = "EMIB"
    COWOS = "CoWoS"
    ORGANIC = "Organic"
    CUSTOM = "Custom"


@dataclass
class InterposerLink:
    """Timing model for a die-to-die link."""

    src_die: int
    dst_die: int
    technology: InterposerTech = InterposerTech.UCIE
    latency_ns: float = 2.0
    jitter_ns: float = 0.1
    bandwidth_gbps: float = 32.0
    bit_error_rate: float = 1e-15
    data_width: int = 64
    is_bidirectional: bool = True

    @classmethod
    def from_tech(cls, src: int, dst: int, tech: InterposerTech) -> InterposerLink:
        """Create link with technology-specific defaults."""
        # Heterogeneous values (float for timing/BER, int implicit for none yet);
        # explicit `dict[str, Any]` so dataclass-field type inference does not narrow.
        presets: Dict[InterposerTech, Dict[str, Any]] = {
            InterposerTech.UCIE: dict(
                latency_ns=2.0, jitter_ns=0.05, bandwidth_gbps=32.0, bit_error_rate=1e-15
            ),
            InterposerTech.BOW: dict(
                latency_ns=1.5, jitter_ns=0.03, bandwidth_gbps=16.0, bit_error_rate=1e-12
            ),
            InterposerTech.EMIB: dict(
                latency_ns=1.0, jitter_ns=0.02, bandwidth_gbps=64.0, bit_error_rate=1e-15
            ),
            InterposerTech.COWOS: dict(
                latency_ns=0.5, jitter_ns=0.01, bandwidth_gbps=128.0, bit_error_rate=1e-16
            ),
            InterposerTech.ORGANIC: dict(
                latency_ns=5.0, jitter_ns=0.5, bandwidth_gbps=8.0, bit_error_rate=1e-12
            ),
            InterposerTech.CUSTOM: dict(
                latency_ns=2.0, jitter_ns=0.1, bandwidth_gbps=32.0, bit_error_rate=1e-15
            ),
        }
        return cls(src_die=src, dst_die=dst, technology=tech, **presets[tech])

    @property
    def latency_cycles(self) -> int:
        """Latency in clock cycles at 200 MHz (5 ns period)."""
        return max(1, int(self.latency_ns / 5.0 + 0.5))

    @property
    def fifo_depth_log2(self) -> int:
        """Minimum async FIFO depth (log2) to absorb jitter."""
        jitter_cycles = max(1, int(self.jitter_ns / 5.0 + 0.5))
        depth = 1
        while (1 << depth) < jitter_cycles * 4:
            depth += 1
        return max(depth, 3)


# ── Chiplet Die ──────────────────────────────────────────────────────


@dataclass
class ChipletDie:
    """One die in a chiplet package."""

    die_id: int
    clock_mhz: float = 200.0
    lfsr_seed: int = 0xACE1
    neuron_ids: List[int] = field(default_factory=list)
    n_neurons: int = 128
    aer_id_width: int = 10
    data_width: int = 16

    @property
    def clock_period_ns(self) -> float:
        return 1000.0 / self.clock_mhz


# ── Chiplet Topology ─────────────────────────────────────────────────


@dataclass
class ChipletTopology:
    """Directed graph of dies + interposer links."""

    dies: List[ChipletDie] = field(default_factory=list)
    links: List[InterposerLink] = field(default_factory=list)

    def add_die(self, die: ChipletDie) -> None:
        self.dies.append(die)

    def add_link(self, link: InterposerLink) -> None:
        self.links.append(link)

    @classmethod
    def mesh_2d(
        cls, rows: int, cols: int, tech: InterposerTech = InterposerTech.UCIE
    ) -> ChipletTopology:
        """Create a 2D mesh topology."""
        topo = cls()
        for r in range(rows):
            for c in range(cols):
                die_id = r * cols + c
                seed = (0xACE1 + die_id * 7919) & 0xFFFF
                if seed == 0:
                    seed = 1
                topo.add_die(ChipletDie(die_id=die_id, lfsr_seed=seed))

        for r in range(rows):
            for c in range(cols):
                src = r * cols + c
                if c + 1 < cols:
                    topo.add_link(InterposerLink.from_tech(src, src + 1, tech))
                if r + 1 < rows:
                    topo.add_link(InterposerLink.from_tech(src, src + cols, tech))
        return topo

    @classmethod
    def ring(cls, n_dies: int, tech: InterposerTech = InterposerTech.UCIE) -> ChipletTopology:
        """Create a ring topology."""
        topo = cls()
        for i in range(n_dies):
            seed = (0xACE1 + i * 7919) & 0xFFFF
            if seed == 0:
                seed = 1
            topo.add_die(ChipletDie(die_id=i, lfsr_seed=seed))
        for i in range(n_dies):
            topo.add_link(InterposerLink.from_tech(i, (i + 1) % n_dies, tech))
        return topo

    @classmethod
    def star(cls, n_dies: int, tech: InterposerTech = InterposerTech.UCIE) -> ChipletTopology:
        """Create a star topology with die 0 as the hub."""
        topo = cls()
        for i in range(n_dies):
            seed = (0xACE1 + i * 7919) & 0xFFFF
            if seed == 0:
                seed = 1
            topo.add_die(ChipletDie(die_id=i, lfsr_seed=seed))
        for i in range(1, n_dies):
            topo.add_link(InterposerLink.from_tech(0, i, tech))
            topo.add_link(InterposerLink.from_tech(i, 0, tech))
        return topo

    def get_links_from(self, die_id: int) -> List[InterposerLink]:
        return [l for l in self.links if l.src_die == die_id]

    def get_links_to(self, die_id: int) -> List[InterposerLink]:
        return [l for l in self.links if l.dst_die == die_id]

    def get_die(self, die_id: int) -> Optional[ChipletDie]:
        return next((d for d in self.dies if d.die_id == die_id), None)

    @property
    def num_dies(self) -> int:
        return len(self.dies)


# ── Routing Table ────────────────────────────────────────────────────


@dataclass
class RoutingEntry:
    """One routing table entry: source neuron → target die + neuron."""

    src_neuron: int
    dst_die: int
    dst_neuron: int
    weight_q88: int = 256  # Q8.8 = 1.0


@dataclass
class RoutingTable:
    """Per-die AER routing table for inter-die communication."""

    die_id: int
    entries: List[RoutingEntry] = field(default_factory=list)

    def add_route(self, src: int, dst_die: int, dst_neuron: int, weight: int = 256) -> None:
        self.entries.append(RoutingEntry(src, dst_die, dst_neuron, weight))

    def routes_to_die(self, target_die: int) -> List[RoutingEntry]:
        return [e for e in self.entries if e.dst_die == target_die]

    @property
    def num_entries(self) -> int:
        return len(self.entries)

    @property
    def target_dies(self) -> List[int]:
        return sorted(set(e.dst_die for e in self.entries))


# ── LFSR Decorrelation Schedule ──────────────────────────────────────


def compute_decorrelation_seeds(topology: ChipletTopology) -> Dict[Tuple[int, int], int]:
    """Assign independent LFSR seeds per link for bitstream decorrelation.

    Uses golden-ratio hashing to ensure maximal separation between
    LFSR sequences across dies (avoids correlated bitstreams).
    """
    phi_inv = 0.6180339887498949
    seeds = {}
    for i, link in enumerate(topology.links):
        key = (link.src_die, link.dst_die)
        raw = int((i * phi_inv * 65535) % 65535) + 1
        seeds[key] = raw
    return seeds


# ── Energy per Bit Model ─────────────────────────────────────────────

_ENERGY_PJ_PER_BIT: Dict[InterposerTech, float] = {
    InterposerTech.UCIE: 0.5,
    InterposerTech.BOW: 0.3,
    InterposerTech.EMIB: 0.2,
    InterposerTech.COWOS: 0.1,
    InterposerTech.ORGANIC: 2.0,
    InterposerTech.CUSTOM: 0.5,
}


def link_energy_pj(link: InterposerLink, bits: int) -> float:
    """Estimate total energy (pJ) for transmitting 'bits' over a link."""
    epb = _ENERGY_PJ_PER_BIT.get(link.technology, 0.5)
    return epb * bits


@dataclass
class PackageEnergyReport:
    """Energy report for the full chiplet package."""

    per_link_pj: Dict[Tuple[int, int], float] = field(default_factory=dict)
    total_pj: float = 0.0

    @property
    def total_nj(self) -> float:
        return self.total_pj / 1000.0


def estimate_package_energy(
    topology: ChipletTopology,
    bits_per_link: int = 256,
) -> PackageEnergyReport:
    """Estimate total communication energy for one inference cycle."""
    report = PackageEnergyReport()
    for link in topology.links:
        key = (link.src_die, link.dst_die)
        epj = link_energy_pj(link, bits_per_link)
        report.per_link_pj[key] = epj
        report.total_pj += epj
    return report


# ── Congestion Estimator ─────────────────────────────────────────────


@dataclass
class CongestionReport:
    """Link utilisation analysis."""

    utilisation: Dict[Tuple[int, int], float] = field(default_factory=dict)
    bottleneck: Optional[Tuple[int, int]] = None
    max_utilisation: float = 0.0


def estimate_congestion(
    topology: ChipletTopology,
    routing_tables: Dict[int, RoutingTable],
    events_per_cycle: int = 100,
) -> CongestionReport:
    """Estimate per-link utilisation given routing tables and traffic.

    Utilisation = (events × data_width) / (bandwidth × 1e9 × period_ns × 1e-9).
    A link with utilisation > 1.0 is over-saturated.
    """
    report = CongestionReport()
    link_traffic: Dict[Tuple[int, int], int] = {}

    for die_id, rt in routing_tables.items():
        for entry in rt.entries:
            key = (die_id, entry.dst_die)
            link_traffic[key] = link_traffic.get(key, 0) + events_per_cycle

    for link in topology.links:
        key = (link.src_die, link.dst_die)
        traffic = link_traffic.get(key, 0)
        bits_per_sec = traffic * link.data_width * 200e6  # at 200 MHz
        capacity_bps = link.bandwidth_gbps * 1e9
        util = bits_per_sec / capacity_bps if capacity_bps > 0 else 0.0
        report.utilisation[key] = util
        if util > report.max_utilisation:
            report.max_utilisation = util
            report.bottleneck = key

    return report


# ── Fault-Tolerant Routing ───────────────────────────────────────────


def find_disjoint_paths(
    topology: ChipletTopology,
    src_die: int,
    dst_die: int,
    max_paths: int = 2,
) -> List[List[int]]:
    """Find up to max_paths link-disjoint paths between two dies.

    Uses iterative BFS with link exclusion to find alternative routes
    for fault-tolerant routing.
    """
    if src_die == dst_die:
        return [[src_die]]

    paths = []
    excluded_links: set = set()

    for _ in range(max_paths):
        path = _bfs_path(topology, src_die, dst_die, excluded_links)
        if path is None:
            break
        paths.append(path)
        for i in range(len(path) - 1):
            excluded_links.add((path[i], path[i + 1]))

    return paths


def _bfs_path(
    topology: ChipletTopology,
    src: int,
    dst: int,
    excluded: set,
) -> Optional[List[int]]:
    """BFS shortest path avoiding excluded links."""
    visited = {src: [src]}
    queue = [src]
    while queue:
        current = queue.pop(0)
        for link in topology.get_links_from(current):
            nxt = link.dst_die
            if (current, nxt) in excluded:
                continue
            if nxt not in visited:
                visited[nxt] = visited[current] + [nxt]
                if nxt == dst:
                    return visited[nxt]
                queue.append(nxt)
    return None


# ── SystemVerilog Generator ──────────────────────────────────────────

_SPDX = """\
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li"""


@dataclass
class ChipletOutput:
    """Generated output files for one chiplet topology."""

    top_sv: str
    die_modules: Dict[int, str]
    link_bridges: Dict[Tuple[int, int], str]
    routing_tables: Dict[int, str]
    constraints_xdc: str
    filelist: List[str]

    def to_dict(self) -> Dict[str, str]:
        d = {"sc_chiplet_top.sv": self.top_sv}
        for die_id, sv in self.die_modules.items():
            d[f"sc_chiplet_die_{die_id}.sv"] = sv
        for (src, dst), sv in self.link_bridges.items():
            d[f"sc_chiplet_bridge_{src}_to_{dst}.sv"] = sv
        for die_id, sv in self.routing_tables.items():
            d[f"sc_chiplet_rtable_{die_id}.sv"] = sv
        d["chiplet_constraints.xdc"] = self.constraints_xdc
        return d


class ChipletGenerator:
    """Generates multi-die routing RTL from a chiplet topology."""

    def generate(
        self,
        topology: ChipletTopology,
        routing: Optional[Dict[int, RoutingTable]] = None,
    ) -> ChipletOutput:
        seeds = compute_decorrelation_seeds(topology)
        die_modules = {}
        link_bridges = {}
        routing_tables = {}

        for die in topology.dies:
            die_modules[die.die_id] = self._emit_die_wrapper(die, topology)
            if routing and die.die_id in routing:
                routing_tables[die.die_id] = self._emit_routing_table(routing[die.die_id], die)

        for link in topology.links:
            seed = seeds.get((link.src_die, link.dst_die), 0xACE1)
            link_bridges[(link.src_die, link.dst_die)] = self._emit_bridge(link, seed)

        top = self._emit_top(topology)
        xdc = self._emit_constraints(topology)
        filelist = list(
            ChipletOutput(top, die_modules, link_bridges, routing_tables, xdc, []).to_dict().keys()
        )

        return ChipletOutput(top, die_modules, link_bridges, routing_tables, xdc, filelist)

    def _emit_die_wrapper(self, die: ChipletDie, topo: ChipletTopology) -> str:
        outgoing = topo.get_links_from(die.die_id)
        incoming = topo.get_links_to(die.die_id)
        out_ports = "\n".join(
            f"    output wire [{l.data_width - 1}:0] link_out_{l.dst_die}_tdata,\n"
            f"    output wire                  link_out_{l.dst_die}_tvalid,\n"
            f"    input  wire                  link_out_{l.dst_die}_tready,"
            for l in outgoing
        )
        in_ports = "\n".join(
            f"    input  wire [{l.data_width - 1}:0] link_in_{l.src_die}_tdata,\n"
            f"    input  wire                  link_in_{l.src_die}_tvalid,\n"
            f"    output wire                  link_in_{l.src_die}_tready,"
            for l in incoming
        )
        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore Chiplet — Die {die.die_id} wrapper

module sc_chiplet_die_{die.die_id} #(
    parameter N_NEURONS = {die.n_neurons},
    parameter AER_ID_W  = {die.aer_id_width},
    parameter DATA_W    = {die.data_width},
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
        .DATA_WIDTH(DATA_W)
    ) local_router (
        .clk(clk), .rst_n(rst_n),
        .in_event_valid(local_spike_valid),
        .in_neuron_id(local_spike_id),
        .in_timestamp(16'd0),
        .out_event_valid(local_out_valid),
        .out_target_id(local_out_id),
        .out_weight(local_out_weight)
    );

endmodule
""")

    def _emit_bridge(self, link: InterposerLink, decor_seed: int) -> str:
        fifo_depth = link.fifo_depth_log2
        latency = link.latency_cycles
        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore Chiplet — Bridge die {link.src_die} → die {link.dst_die}
// Technology: {link.technology.value}
// Latency: {link.latency_ns} ns ({latency} cycles @ 200 MHz)
// Bandwidth: {link.bandwidth_gbps} Gb/s
// BER: {link.bit_error_rate}

module sc_chiplet_bridge_{link.src_die}_to_{link.dst_die} #(
    parameter DATA_W       = {link.data_width},
    parameter FIFO_DEPTH   = {fifo_depth},
    parameter LATENCY_CYC  = {latency},
    parameter DECOR_SEED   = 16'h{decor_seed:04X}
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
        entries_sv = []
        for e in table.entries:
            entries_sv.append(
                f"        rt_target_die[{e.src_neuron}]    = {e.dst_die};\n"
                f"        rt_target_neuron[{e.src_neuron}] = {e.dst_neuron};\n"
                f"        rt_weight[{e.src_neuron}]        = 16'sd{e.weight_q88};"
            )
        init_block = "\n".join(entries_sv) if entries_sv else "        // No inter-die routes"
        n_entries = max(len(table.entries), 1)

        return textwrap.dedent(f"""\
{_SPDX}
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
{init_block}
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

    def _emit_top(self, topo: ChipletTopology) -> str:
        die_insts = []
        for die in topo.dies:
            die_insts.append(
                f"    // Die {die.die_id}\n"
                f"    sc_chiplet_die_{die.die_id} die_{die.die_id}_inst (\n"
                f"        .clk(clk), .rst_n(rst_n)\n"
                f"        // TODO: wire link ports\n"
                f"    );"
            )
        bridge_insts = []
        for link in topo.links:
            bridge_insts.append(
                f"    // Bridge {link.src_die} → {link.dst_die} ({link.technology.value})\n"
                f"    sc_chiplet_bridge_{link.src_die}_to_{link.dst_die} bridge_{link.src_die}_{link.dst_die}_inst (\n"
                f"        .src_clk(clk), .src_rst(!rst_n),\n"
                f"        .dst_clk(clk), .dst_rst(!rst_n)\n"
                f"        // TODO: wire AXI-Stream ports\n"
                f"    );"
            )

        die_block = "\n\n".join(die_insts)
        bridge_block = "\n\n".join(bridge_insts)

        return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore Chiplet — Top-level multi-die package
// Dies: {topo.num_dies}
// Links: {len(topo.links)}

module sc_chiplet_top (
    input wire clk,
    input wire rst_n
);

{die_block}

{bridge_block}

endmodule
""")

    def _emit_constraints(self, topo: ChipletTopology) -> str:
        lines = [
            "# SC-NeuroCore Chiplet — Timing constraints",
            "# Auto-generated for multi-die package",
            "",
        ]
        for die in topo.dies:
            lines.append(f"# Die {die.die_id}: {die.clock_mhz} MHz")
            lines.append(
                f"create_clock -name clk_die_{die.die_id} "
                f"-period {die.clock_period_ns:.3f} "
                f"[get_pins die_{die.die_id}_inst/clk]"
            )
        lines.append("")
        for link in topo.links:
            lines.append(f"# Link {link.src_die} → {link.dst_die}: {link.latency_ns} ns")
            lines.append(
                f"set_max_delay -from [get_clocks clk_die_{link.src_die}] "
                f"-to [get_clocks clk_die_{link.dst_die}] {link.latency_ns}"
            )
        return "\n".join(lines) + "\n"


# ── Timing Simulator ─────────────────────────────────────────────────


@dataclass
class TimingSimResult:
    """Result of interposer timing simulation."""

    total_latency_ns: float
    max_jitter_ns: float
    min_bandwidth_gbps: float
    worst_ber: float
    path: List[int]


def simulate_timing(
    topology: ChipletTopology, src_die: int, dst_die: int
) -> Optional[TimingSimResult]:
    """BFS shortest-path timing simulation between two dies."""
    if src_die == dst_die:
        return TimingSimResult(0.0, 0.0, float("inf"), 0.0, [src_die])

    # BFS
    visited = {src_die: (0.0, 0.0, float("inf"), 0.0, [src_die])}
    queue = [src_die]
    while queue:
        current = queue.pop(0)
        lat, jit, bw, ber, path = visited[current]
        for link in topology.get_links_from(current):
            nxt = link.dst_die
            new_lat = lat + link.latency_ns
            new_jit = max(jit, link.jitter_ns)
            new_bw = min(bw, link.bandwidth_gbps)
            new_ber = max(ber, link.bit_error_rate)
            new_path = path + [nxt]
            if nxt not in visited or visited[nxt][0] > new_lat:
                visited[nxt] = (new_lat, new_jit, new_bw, new_ber, new_path)
                queue.append(nxt)

    if dst_die not in visited:
        return None
    lat, jit, bw, ber, path = visited[dst_die]
    return TimingSimResult(lat, jit, bw, ber, path)


# ── Torus Topology (Gap 1) ───────────────────────────────────────────


def make_torus(
    rows: int,
    cols: int,
    tech: InterposerTech = InterposerTech.UCIE,
) -> ChipletTopology:
    """Create a 2D torus topology (mesh with wrap-around edges)."""
    topo = ChipletTopology()
    for r in range(rows):
        for c in range(cols):
            die_id = r * cols + c
            seed = (0xACE1 + die_id * 7919) & 0xFFFF
            if seed == 0:
                seed = 1
            topo.add_die(ChipletDie(die_id=die_id, lfsr_seed=seed))

    for r in range(rows):
        for c in range(cols):
            src = r * cols + c
            # Right neighbour (wraps)
            right = r * cols + (c + 1) % cols
            topo.add_link(InterposerLink.from_tech(src, right, tech))
            # Down neighbour (wraps)
            down = ((r + 1) % rows) * cols + c
            topo.add_link(InterposerLink.from_tech(src, down, tech))
    return topo


# ── Heterogeneous Clock Domain CDC (Gap 2) ───────────────────────────


@dataclass
class CDCConfig:
    """Per-link CDC configuration for heterogeneous clock domains."""

    src_clk_mhz: float
    dst_clk_mhz: float
    fifo_depth_log2: int = 4
    sync_stages: int = 2

    @property
    def ratio(self) -> float:
        if self.dst_clk_mhz == 0:
            return 1.0
        return self.src_clk_mhz / self.dst_clk_mhz

    @property
    def is_mesochronous(self) -> bool:
        return abs(self.ratio - 1.0) < 0.01


def compute_cdc_configs(topology: ChipletTopology) -> Dict[Tuple[int, int], CDCConfig]:
    """Auto-compute per-link CDC configs from die clock frequencies."""
    configs = {}
    for link in topology.links:
        src_die = topology.get_die(link.src_die)
        dst_die = topology.get_die(link.dst_die)
        if src_die is None or dst_die is None:
            continue
        fifo = link.fifo_depth_log2
        sync = 3 if src_die.clock_mhz != dst_die.clock_mhz else 2
        configs[(link.src_die, link.dst_die)] = CDCConfig(
            src_clk_mhz=src_die.clock_mhz,
            dst_clk_mhz=dst_die.clock_mhz,
            fifo_depth_log2=fifo,
            sync_stages=sync,
        )
    return configs


# ── Thermal Model (Gap 3) ────────────────────────────────────────────


@dataclass
class DieThermal:
    """Thermal state for one die."""

    die_id: int
    temperature_c: float = 25.0
    power_mw: float = 100.0
    thermal_resistance_k_per_w: float = 5.0
    max_temperature_c: float = 105.0

    @property
    def is_throttled(self) -> bool:
        return self.temperature_c >= self.max_temperature_c

    def step(self, ambient_c: float = 25.0) -> float:
        """One thermal step: T = T_ambient + P * R_th."""
        self.temperature_c = ambient_c + (self.power_mw / 1000.0) * self.thermal_resistance_k_per_w
        return self.temperature_c


@dataclass
class PackageThermalReport:
    """Thermal report for the full package."""

    die_temps: Dict[int, float] = field(default_factory=dict)
    max_temp: float = 0.0
    throttled_dies: List[int] = field(default_factory=list)


def simulate_thermal(
    topology: ChipletTopology,
    power_per_die_mw: Optional[Dict[int, float]] = None,
    ambient_c: float = 25.0,
) -> PackageThermalReport:
    """Estimate die temperatures for a chiplet package."""
    report = PackageThermalReport()
    for die in topology.dies:
        p = power_per_die_mw.get(die.die_id, 100.0) if power_per_die_mw else 100.0
        dt = DieThermal(die_id=die.die_id, power_mw=p)
        temp = dt.step(ambient_c)
        report.die_temps[die.die_id] = temp
        if temp > report.max_temp:
            report.max_temp = temp
        if dt.is_throttled:
            report.throttled_dies.append(die.die_id)
    return report


# ── Adaptive Routing (Gap 4) ─────────────────────────────────────────


def adaptive_route(
    topology: ChipletTopology,
    src_die: int,
    dst_die: int,
    congestion: CongestionReport,
    congestion_threshold: float = 0.8,
) -> Optional[List[int]]:
    """Congestion-reactive routing: avoid saturated links.

    Uses BFS but excludes links with utilisation above the threshold.
    Falls back to shortest path if no uncongested route exists.
    """
    excluded = set()
    for (s, d), util in congestion.utilisation.items():
        if util > congestion_threshold:
            excluded.add((s, d))

    path = _bfs_path(topology, src_die, dst_die, excluded)
    if path is not None:
        return path
    # Fallback: ignore congestion
    return _bfs_path(topology, src_die, dst_die, set())


# ── ECC / CRC Link Protection (Gap 5) ────────────────────────────────


@dataclass
class LinkProtection:
    """ECC/CRC configuration for a die-to-die link."""

    mode: str = "crc32"  # "none", "parity", "crc8", "crc32", "secded"
    overhead_bits: int = 0

    def __post_init__(self) -> None:
        overhead_map = {
            "none": 0,
            "parity": 1,
            "crc8": 8,
            "crc32": 32,
            "secded": 8,  # SEC-DED for 64-bit data
        }
        self.overhead_bits = overhead_map.get(self.mode, 0)

    @property
    def effective_bandwidth_ratio(self) -> float:
        if self.overhead_bits == 0:
            return 1.0
        return 64.0 / (64.0 + self.overhead_bits)


def emit_crc32_sv(data_width: int = 64) -> str:
    """CRC-32 checker stub for link error detection."""
    return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore Chiplet — CRC-32 link checker

module sc_chiplet_crc32 #(
    parameter DATA_W = {data_width}
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire [DATA_W-1:0]  data_in,
    input  wire               data_valid,
    output reg  [31:0]        crc_out,
    output reg                crc_valid,
    output reg                crc_error
);

    reg [31:0] crc_reg;
    integer i;

    always @(posedge clk) begin
        if (!rst_n) begin
            crc_reg   <= 32'hFFFFFFFF;
            crc_out   <= 0;
            crc_valid <= 0;
            crc_error <= 0;
        end else if (data_valid) begin
            crc_reg <= crc_reg;  // Placeholder: real CRC-32 polynomial
            for (i = 0; i < DATA_W; i = i + 1)
                crc_reg <= {{crc_reg[30:0], crc_reg[31] ^ data_in[i]}};
            crc_out   <= crc_reg;
            crc_valid <= 1;
        end else begin
            crc_valid <= 0;
        end
    end

endmodule
""")


# ── Bandwidth-Aware Routing (Gap 6) ──────────────────────────────────


def bandwidth_aware_route(
    topology: ChipletTopology,
    src_die: int,
    dst_die: int,
    required_gbps: float,
) -> Optional[List[int]]:
    """Find a path where all links have bandwidth >= required_gbps."""
    if src_die == dst_die:
        return [src_die]

    visited = {src_die: [src_die]}
    queue = [src_die]
    while queue:
        current = queue.pop(0)
        for link in topology.get_links_from(current):
            nxt = link.dst_die
            if nxt in visited:
                continue
            if link.bandwidth_gbps < required_gbps:
                continue
            visited[nxt] = visited[current] + [nxt]
            if nxt == dst_die:
                return visited[nxt]
            queue.append(nxt)
    return None


# ── Credit-Based Flow Control (Gap 7) ────────────────────────────────


@dataclass
class CreditConfig:
    """Per-link credit-based flow control parameters."""

    initial_credits: int = 16
    credit_granularity: int = 1  # flits per credit

    @property
    def buffer_flits(self) -> int:
        return self.initial_credits * self.credit_granularity


def emit_credit_controller_sv(config: CreditConfig, link_name: str = "link") -> str:
    """Credit-based flow controller stub."""
    return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore Chiplet — Credit controller for {link_name}

module sc_chiplet_credit_{link_name} #(
    parameter INIT_CREDITS = {config.initial_credits},
    parameter DATA_W       = 64
)(
    input  wire               clk,
    input  wire               rst_n,
    // TX side
    input  wire [DATA_W-1:0]  tx_data,
    input  wire               tx_valid,
    output wire               tx_ready,
    // RX credit return
    input  wire               credit_return,
    output reg  [7:0]         credits_available
);

    always @(posedge clk) begin
        if (!rst_n)
            credits_available <= INIT_CREDITS;
        else begin
            if (tx_valid && tx_ready && !credit_return)
                credits_available <= credits_available - 1;
            else if (!tx_valid && credit_return)
                credits_available <= credits_available + 1;
        end
    end

    assign tx_ready = (credits_available > 0);

endmodule
""")


# ── 3D Stacking / TSV Model (Gap 8) ──────────────────────────────────


class StackingType(Enum):
    COPLANAR = "coplanar"
    TSV_3D = "tsv_3d"
    HYBRID_BONDING = "hybrid_bonding"


@dataclass
class TSVLink:
    """Through-silicon via (TSV) link for 3D stacking."""

    src_die: int
    dst_die: int
    stacking: StackingType = StackingType.TSV_3D
    tsv_pitch_um: float = 10.0
    tsv_count: int = 1024
    latency_ps: float = 50.0  # TSV is very fast

    @property
    def latency_ns(self) -> float:
        return self.latency_ps / 1000.0

    @property
    def bandwidth_gbps(self) -> float:
        return self.tsv_count * 200e6 / 1e9  # 1 bit/TSV at 200 MHz


def add_3d_stack(
    topology: ChipletTopology,
    bottom_die: int,
    top_die: int,
    stacking: StackingType = StackingType.TSV_3D,
) -> InterposerLink:
    """Add a vertical (3D) link between stacked dies."""
    presets: Dict[StackingType, Dict[str, Any]] = {
        StackingType.TSV_3D: dict(latency_ns=0.05, bandwidth_gbps=256.0, bit_error_rate=1e-18),
        StackingType.HYBRID_BONDING: dict(
            latency_ns=0.01, bandwidth_gbps=512.0, bit_error_rate=1e-20
        ),
        StackingType.COPLANAR: dict(latency_ns=2.0, bandwidth_gbps=32.0, bit_error_rate=1e-15),
    }
    params = presets.get(stacking, presets[StackingType.COPLANAR])
    link = InterposerLink(
        src_die=bottom_die,
        dst_die=top_die,
        technology=InterposerTech.CUSTOM,
        is_bidirectional=True,
        **params,
    )
    topology.add_link(link)
    # Reverse direction
    rev = InterposerLink(
        src_die=top_die,
        dst_die=bottom_die,
        technology=InterposerTech.CUSTOM,
        is_bidirectional=True,
        **params,
    )
    topology.add_link(rev)
    return link


# ── Power Domain Isolation (Gap 9) ───────────────────────────────────


@dataclass
class PowerDomain:
    """Voltage island / power domain for a subset of dies."""

    domain_id: int
    die_ids: List[int] = field(default_factory=list)
    voltage_mv: int = 800
    is_active: bool = True

    @property
    def is_gated(self) -> bool:
        return not self.is_active


@dataclass
class PowerDomainMap:
    """Maps dies to power domains for isolation/gating."""

    domains: List[PowerDomain] = field(default_factory=list)

    def add_domain(self, domain: PowerDomain) -> None:
        self.domains.append(domain)

    def domain_for_die(self, die_id: int) -> Optional[PowerDomain]:
        for d in self.domains:
            if die_id in d.die_ids:
                return d
        return None

    def active_dies(self) -> List[int]:
        result = []
        for d in self.domains:
            if d.is_active:
                result.extend(d.die_ids)
        return sorted(result)

    def gated_dies(self) -> List[int]:
        result = []
        for d in self.domains:
            if not d.is_active:
                result.extend(d.die_ids)
        return sorted(result)


def emit_power_gating_sv(domain: PowerDomain) -> str:
    """Power-gating controller stub for a voltage island."""
    die_list = ", ".join(str(d) for d in domain.die_ids)
    return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore Chiplet — Power domain {domain.domain_id} controller
// Dies: [{die_list}]
// Voltage: {domain.voltage_mv} mV

module sc_chiplet_pwr_domain_{domain.domain_id} (
    input  wire clk,
    input  wire rst_n,
    input  wire enable,
    output reg  domain_active,
    output reg  isolation_en
);

    always @(posedge clk) begin
        if (!rst_n) begin
            domain_active <= 1'b0;
            isolation_en  <= 1'b1;
        end else begin
            domain_active <= enable;
            isolation_en  <= !enable;
        end
    end

endmodule
""")


# ── Auto-Partitioning Hook (Gap 10) ──────────────────────────────────


@dataclass
class PartitionAssignment:
    """Maps neurons/layers to dies from the hierarchical partitioner."""

    die_assignments: Dict[int, List[int]] = field(default_factory=dict)

    def assign(self, neuron_id: int, die_id: int) -> None:
        self.die_assignments.setdefault(die_id, []).append(neuron_id)

    def neurons_on_die(self, die_id: int) -> List[int]:
        return self.die_assignments.get(die_id, [])

    def to_routing_tables(
        self, connectivity: List[Tuple[int, int, int]]
    ) -> Dict[int, RoutingTable]:
        """Convert partition + connectivity to per-die routing tables.

        ``connectivity`` is a list of (src_neuron, dst_neuron, weight_q88).
        Cross-die connections become routing table entries.
        """
        neuron_to_die: Dict[int, int] = {}
        for die_id, neurons in self.die_assignments.items():
            for n in neurons:
                neuron_to_die[n] = die_id

        tables: Dict[int, RoutingTable] = {}
        for src, dst, w in connectivity:
            src_die = neuron_to_die.get(src)
            dst_die = neuron_to_die.get(dst)
            if src_die is None or dst_die is None:
                continue
            if src_die == dst_die:
                continue  # local — no inter-die routing needed
            if src_die not in tables:
                tables[src_die] = RoutingTable(die_id=src_die)
            tables[src_die].add_route(src, dst_die, dst, w)

        return tables
