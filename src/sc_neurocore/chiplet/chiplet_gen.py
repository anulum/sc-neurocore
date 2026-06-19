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

import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    thermal_resistance_k_per_w: Optional[float] = None

    def __post_init__(self) -> None:
        if self.thermal_resistance_k_per_w is not None and self.thermal_resistance_k_per_w <= 0:
            raise ValueError("thermal_resistance_k_per_w must be > 0 when provided")

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

_SV_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_sv_identifier(value: str, field_name: str) -> str:
    """Validate a generated SystemVerilog identifier fragment."""
    if not _SV_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a valid SystemVerilog identifier")
    return value


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
    excluded_links: set[tuple[int, int]] = set()

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
    excluded: set[tuple[int, int]],
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
        out_assigns = "\n".join(
            f"    assign link_out_{l.dst_die}_tdata = "
            f"{{{{({l.data_width}-AER_ID_W){{1'b0}}}}, local_out_id}};\n"
            f"    assign link_out_{l.dst_die}_tvalid = local_out_valid;"
            for l in outgoing
        )
        in_assigns = "\n".join(f"    assign link_in_{l.src_die}_tready = 1'b1;" for l in incoming)
        return textwrap.dedent(f"""\
{_SPDX}
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
        link_wires = []
        for link in topo.links:
            link_wires.append(
                f"    wire [{link.data_width - 1}:0] link_{link.src_die}_{link.dst_die}_tdata;\n"
                f"    wire                  link_{link.src_die}_{link.dst_die}_tvalid;\n"
                f"    wire                  link_{link.src_die}_{link.dst_die}_tready;\n"
                f"    wire [{link.data_width - 1}:0] link_{link.src_die}_{link.dst_die}_rx_tdata;\n"
                f"    wire                  link_{link.src_die}_{link.dst_die}_rx_tvalid;\n"
                f"    wire                  link_{link.src_die}_{link.dst_die}_rx_tready;"
            )
        link_wire_block = "\n".join(link_wires)

        die_insts = []
        for die in topo.dies:
            outgoing = topo.get_links_from(die.die_id)
            incoming = topo.get_links_to(die.die_id)
            link_ports = []
            for link in outgoing:
                link_ports.extend(
                    [
                        f"        .link_out_{link.dst_die}_tdata(link_{link.src_die}_{link.dst_die}_tdata)",
                        f"        .link_out_{link.dst_die}_tvalid(link_{link.src_die}_{link.dst_die}_tvalid)",
                        f"        .link_out_{link.dst_die}_tready(link_{link.src_die}_{link.dst_die}_tready)",
                    ]
                )
            for link in incoming:
                link_ports.extend(
                    [
                        f"        .link_in_{link.src_die}_tdata(link_{link.src_die}_{link.dst_die}_rx_tdata)",
                        f"        .link_in_{link.src_die}_tvalid(link_{link.src_die}_{link.dst_die}_rx_tvalid)",
                        f"        .link_in_{link.src_die}_tready(link_{link.src_die}_{link.dst_die}_rx_tready)",
                    ]
                )
            link_port_block = ""
            if link_ports:
                link_port_block = ",\n" + ",\n".join(link_ports)
            die_insts.append(
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
        bridge_insts = []
        for link in topo.links:
            bridge_insts.append(
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

{link_wire_block}

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


# ── Thermal Model — HotSpot-style conductance-matrix solver ──────────
#
# Implementation reference:
#   Skadron, K. et al. "HotSpot: A Compact Thermal Modeling
#   Methodology for Early-Stage VLSI Design." IEEE Trans. on
#   VLSI Systems, 2006.
#   https://lava.cs.virginia.edu/HotSpot/
#
# Plus the inter-chiplet thermal coupling model from:
#   Coskun, A. et al. "Cross-Layer Thermal Modeling and
#   Management for 3D Stacked Multi-Chip Modules." DATE 2013.
#
# The earlier single-die lumped-element equation (T = T_amb + P·R)
# was replaced 2026-04-17 per `feedback_sophisticated_from_start.md`.


# Per-interposer-technology thermal resistance (K/W) for the
# die-to-die bond. Values are representative orders of magnitude
# from published vendor data; refine with PDK-specific values
# when designing for a real package.
_R_THERMAL_K_PER_W: Dict[InterposerTech, float] = {
    InterposerTech.UCIE: 0.8,  # silicon interposer, fine-pitch microbumps
    InterposerTech.BOW: 3.0,  # organic substrate
    InterposerTech.EMIB: 0.5,  # silicon bridge, high conductivity
    InterposerTech.COWOS: 0.3,  # bulk silicon interposer, very low R
    InterposerTech.ORGANIC: 8.0,  # organic only, high R
    InterposerTech.CUSTOM: 1.0,
}


@dataclass
class DieThermal:
    """Per-die thermal properties: power source, capacity, ambient path.

    `power_mw` and `temperature_c` are runtime state; the rest are
    geometry / packaging properties. The dataclass holds the data
    consumed by the package-level conductance-matrix solver below.
    """

    die_id: int
    temperature_c: float = 25.0
    power_mw: float = 100.0

    # Per-die thermal capacity (J/K). For a 10×10 mm² × 500 µm
    # silicon die with c_p,Si = 1.65 J/(cm³·K), C ≈ 0.083 J/K.
    heat_capacity_j_per_k: float = 0.083

    # Junction-to-ambient resistance (K/W). Models the path from
    # the die through the heat spreader / heat sink to the
    # ambient air. Default 1.5 K/W is representative of an
    # actively cooled BGA package.
    r_to_ambient_k_per_w: float = 1.5

    # Spreading resistance from the bond pad to the die centre
    # (K/W). Adds in series to the inter-die bond resistance to
    # form the effective coupling resistance.
    r_spread_k_per_w: float = 0.2

    max_temperature_c: float = 105.0

    @property
    def is_throttled(self) -> bool:
        return self.temperature_c >= self.max_temperature_c


@dataclass
class PackageThermalReport:
    """Steady-state + transient thermal report for the full package."""

    # Steady-state per-die temperatures (°C)
    die_temps: Dict[int, float] = field(default_factory=dict)
    # Max per-die temperature across the package
    max_temp: float = 0.0
    # Dies whose steady-state temperature exceeds the throttle limit
    throttled_dies: List[int] = field(default_factory=list)
    # Transient time-series — populated only if `transient_steps>0`
    # was passed to `simulate_thermal`. Shape: (n_steps, n_dies).
    transient_temps: Optional[np.ndarray[Any, Any]] = None
    transient_times_s: Optional[np.ndarray[Any, Any]] = None
    # Conductance matrix actually used (for inspection / debugging).
    # Shape: (n_dies, n_dies).
    conductance_matrix: Optional[np.ndarray[Any, Any]] = None


def _build_conductance_matrix(
    topology: ChipletTopology,
    die_state: Dict[int, DieThermal],
) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], List[int]]:
    """Build (G, g_amb, die_id_order) for the thermal network.

    G is the off-diagonal conductance matrix (W/K) for inter-die
    couplings. g_amb is the per-die conductance to ambient.
    die_id_order is the row → die_id mapping used by the solver.

    For each link (i, j) in `topology.links` with technology
    `tech`, the bond resistance is
        R_bond(i,j) = R_THERMAL[tech] + R_spread(i) + R_spread(j)
    so the off-diagonal conductance is
        G[i,j] = G[j,i] = 1 / R_bond(i,j)

    Per-die ambient conductance:
        g_amb[i] = 1 / R_to_ambient(i)
    """
    die_id_order = [die.die_id for die in topology.dies]
    n = len(die_id_order)
    idx_of = {did: k for k, did in enumerate(die_id_order)}

    G = np.zeros((n, n), dtype=np.float64)
    g_amb = np.zeros(n, dtype=np.float64)

    for did in die_id_order:
        g_amb[idx_of[did]] = 1.0 / die_state[did].r_to_ambient_k_per_w

    for link in topology.links:
        i = idx_of.get(link.src_die)
        j = idx_of.get(link.dst_die)
        if i is None or j is None or i == j:
            continue
        r_link = (
            link.thermal_resistance_k_per_w
            if link.thermal_resistance_k_per_w is not None
            else _R_THERMAL_K_PER_W.get(link.technology, 1.0)
        )
        r_bond = (
            r_link
            + die_state[link.src_die].r_spread_k_per_w
            + die_state[link.dst_die].r_spread_k_per_w
        )
        g_link = 1.0 / r_bond
        # Symmetric: heat flows both ways across the bond
        G[i, j] += g_link
        G[j, i] += g_link

    return G, g_amb, die_id_order


def _solve_steady_state(
    G_off: np.ndarray[Any, Any],
    g_amb: np.ndarray[Any, Any],
    p_w: np.ndarray[Any, Any],
    t_amb_c: float,
) -> np.ndarray[Any, Any]:
    """Solve `(D - G_off) · T = P + g_amb · T_amb` for T.

    where D is the diagonal of row-sums (Kirchhoff's current law
    for the thermal network).

    Returns the per-die steady-state temperature in °C.
    """
    n = G_off.shape[0]
    # Diagonal of effective conductance matrix:
    #   each die's diagonal entry = sum of off-diag conductances + g_amb
    diag = G_off.sum(axis=1) + g_amb
    A = np.diag(diag) - G_off
    b = p_w + g_amb * t_amb_c
    return np.linalg.solve(A, b)


def _solve_transient(
    G_off: np.ndarray[Any, Any],
    g_amb: np.ndarray[Any, Any],
    capacities: np.ndarray[Any, Any],
    p_w: np.ndarray[Any, Any],
    t_amb_c: float,
    initial_t_c: np.ndarray[Any, Any],
    dt_s: float,
    n_steps: int,
) -> np.ndarray[Any, Any]:
    """Implicit-Euler integration of `C · dT/dt = -A · T + b`.

    A and b are the same matrices as in `_solve_steady_state`.
    Implicit Euler is unconditionally stable for the stiff
    thermal system and converges to the steady-state solution
    as t → ∞.

    Per step: `(C/dt + A) · T_new = C/dt · T_old + b`.

    Returns array of shape (n_steps, n_dies) with the temperature
    trajectory.
    """
    n = G_off.shape[0]
    diag = G_off.sum(axis=1) + g_amb
    A = np.diag(diag) - G_off
    b = p_w + g_amb * t_amb_c

    C_over_dt = np.diag(capacities / dt_s)
    M = C_over_dt + A

    T = initial_t_c.copy()
    out = np.empty((n_steps, n), dtype=np.float64)
    for k in range(n_steps):
        rhs = (capacities / dt_s) * T + b
        T = np.linalg.solve(M, rhs)
        out[k] = T
    return out


def simulate_thermal(
    topology: ChipletTopology,
    power_per_die_mw: Optional[Dict[int, float]] = None,
    ambient_c: float = 25.0,
    *,
    die_state: Optional[Dict[int, DieThermal]] = None,
    transient_steps: int = 0,
    transient_dt_s: float = 1e-3,
) -> PackageThermalReport:
    """Solve the full chiplet thermal network.

    Builds a per-die conductance matrix from `topology.links` (using
    per-technology thermal resistance) plus a per-die ambient path.
    The steady-state temperature is the solution to a linear system;
    a transient response can also be requested with implicit-Euler
    time-stepping.

    Parameters
    ----------
    topology : ChipletTopology
        Dies + interposer links.
    power_per_die_mw : dict, optional
        Per-die power dissipation in mW. Defaults to 100 mW per
        die when missing.
    ambient_c : float
        Ambient air temperature in °C (default 25).
    die_state : dict, optional
        Per-die thermal property overrides. When omitted, defaults
        from `DieThermal` are used (10×10 mm² silicon die, 1.5 K/W
        junction-to-ambient).
    transient_steps : int
        If > 0, also compute a transient response of this many
        steps. The trajectory and time array are placed in the
        returned report.
    transient_dt_s : float
        Time step (s) for transient integration. Default 1 ms is
        appropriate for ~0.08 J/K dies under ~0.1 W loading
        (thermal time constant ~0.1 s).
    """
    # Materialise per-die state (apply defaults / overrides).
    state: Dict[int, DieThermal] = {}
    for die in topology.dies:
        if die_state is not None and die.die_id in die_state:
            ds = die_state[die.die_id]
        else:
            ds = DieThermal(die_id=die.die_id)
        # Apply per-die power override
        p_mw = power_per_die_mw.get(die.die_id, 100.0) if power_per_die_mw else 100.0
        ds.power_mw = p_mw
        state[die.die_id] = ds

    G_off, g_amb, die_id_order = _build_conductance_matrix(topology, state)
    n = len(die_id_order)
    p_w = np.array([state[d].power_mw / 1000.0 for d in die_id_order], dtype=np.float64)

    t_steady = _solve_steady_state(G_off, g_amb, p_w, ambient_c)

    # Update state + report
    for k, did in enumerate(die_id_order):
        state[did].temperature_c = float(t_steady[k])

    report = PackageThermalReport()
    report.conductance_matrix = G_off
    for k, did in enumerate(die_id_order):
        t = float(t_steady[k])
        report.die_temps[did] = t
        if t > report.max_temp:
            report.max_temp = t
        if state[did].is_throttled:
            report.throttled_dies.append(did)

    if transient_steps > 0:
        capacities = np.array(
            [state[d].heat_capacity_j_per_k for d in die_id_order],
            dtype=np.float64,
        )
        # Start from ambient (cold-boot transient).
        initial_t = np.full(n, ambient_c, dtype=np.float64)
        traj = _solve_transient(
            G_off,
            g_amb,
            capacities,
            p_w,
            ambient_c,
            initial_t,
            transient_dt_s,
            transient_steps,
        )
        report.transient_temps = traj
        report.transient_times_s = np.arange(1, transient_steps + 1) * transient_dt_s

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
    """Emit IEEE 802.3 CRC-32 feedback logic for link error detection."""
    if data_width <= 0:
        raise ValueError("data_width must be a positive integer")

    return textwrap.dedent(f"""\
{_SPDX}
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

    def __post_init__(self) -> None:
        if self.initial_credits <= 0:
            raise ValueError("initial_credits must be > 0")
        if self.credit_granularity <= 0:
            raise ValueError("credit_granularity must be > 0")

    @property
    def buffer_flits(self) -> int:
        return self.initial_credits * self.credit_granularity

    @property
    def credit_width(self) -> int:
        return max(1, self.buffer_flits.bit_length())


def emit_credit_controller_sv(config: CreditConfig, link_name: str = "link") -> str:
    """Emit saturating credit-based flow control for a die-to-die link."""
    _require_sv_identifier(link_name, "link_name")
    return textwrap.dedent(f"""\
{_SPDX}
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

    def __post_init__(self) -> None:
        if self.domain_id < 0:
            raise ValueError("domain_id must be >= 0")
        if not self.die_ids:
            raise ValueError("die_ids must contain at least one die")
        if any(die_id < 0 or die_id >= 64 for die_id in self.die_ids):
            raise ValueError("die_ids must be in the range [0, 63]")
        if len(set(self.die_ids)) != len(self.die_ids):
            raise ValueError("die_ids must not contain duplicates")
        if self.voltage_mv <= 0:
            raise ValueError("voltage_mv must be > 0")

    @property
    def is_gated(self) -> bool:
        return not self.is_active

    @property
    def die_mask(self) -> int:
        mask = 0
        for die_id in self.die_ids:
            mask |= 1 << die_id
        return mask


@dataclass
class PowerDomainMap:
    """Maps dies to power domains for isolation/gating."""

    domains: List[PowerDomain] = field(default_factory=list)

    def add_domain(self, domain: PowerDomain) -> None:
        assigned = {die_id for existing in self.domains for die_id in existing.die_ids}
        overlap = assigned.intersection(domain.die_ids)
        if overlap:
            die_list = ", ".join(str(die_id) for die_id in sorted(overlap))
            raise ValueError(f"die_ids already assigned to a power domain: {die_list}")
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
    """Emit sequenced isolation and switch control for a voltage island."""
    die_list = ", ".join(str(d) for d in domain.die_ids)
    return textwrap.dedent(f"""\
{_SPDX}
// SC-NeuroCore Chiplet — Power domain {domain.domain_id} controller
// Dies: [{die_list}]
// Voltage: {domain.voltage_mv} mV

module sc_chiplet_pwr_domain_{domain.domain_id} #(
    parameter DOMAIN_ID = {domain.domain_id},
    parameter DIE_COUNT = {len(domain.die_ids)},
    parameter [63:0] DIE_MASK = 64'h{domain.die_mask:016X},
    parameter VOLTAGE_MV = {domain.voltage_mv}
)(
    input  wire clk,
    input  wire rst_n,
    input  wire enable,
    output reg  domain_active,
    output reg  isolation_en,
    output reg  power_switch_en
);

    localparam PWR_OFF     = 2'd0;
    localparam PWR_ON      = 2'd1;
    localparam PWR_ISOLATE = 2'd2;
    localparam PWR_RESTORE = 2'd3;
    localparam ISO_CYCLES  = 4;
    localparam RESTORE_CYCLES = 4;

    reg [1:0] state;
    reg [2:0] iso_count;
    reg [2:0] restore_count;

    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= PWR_OFF;
            iso_count     <= 0;
            restore_count <= 0;
            domain_active <= 1'b0;
            isolation_en  <= 1'b1;
            power_switch_en <= 1'b0;
        end else begin
            case (state)
                PWR_OFF: begin
                    domain_active <= 1'b0;
                    isolation_en <= 1'b1;
                    power_switch_en <= 1'b0;
                    iso_count <= 0;
                    restore_count <= 0;
                    if (enable)
                        state <= PWR_RESTORE;
                end
                PWR_RESTORE: begin
                    power_switch_en <= 1'b1;
                    isolation_en <= 1'b1;
                    domain_active <= 1'b0;
                    iso_count <= 0;
                    if (restore_count == RESTORE_CYCLES[2:0] - 1'b1) begin
                        restore_count <= 0;
                        state <= PWR_ON;
                    end else begin
                        restore_count <= restore_count + 1'b1;
                    end
                end
                PWR_ON: begin
                    domain_active <= 1'b1;
                    isolation_en <= 1'b0;
                    power_switch_en <= 1'b1;
                    iso_count <= 0;
                    restore_count <= 0;
                    if (!enable)
                        state <= PWR_ISOLATE;
                end
                PWR_ISOLATE: begin
                    domain_active <= 1'b1;
                    isolation_en <= 1'b1;
                    power_switch_en <= 1'b1;
                    restore_count <= 0;
                    if (iso_count == ISO_CYCLES[2:0]) begin
                        domain_active <= 1'b0;
                        power_switch_en <= 1'b0;
                        state <= PWR_OFF;
                    end else begin
                        iso_count <= iso_count + 1'b1;
                    end
                end
                default: state <= PWR_OFF;
            endcase
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
