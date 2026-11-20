# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet package clocks: one per die, in the RTL and in the XDC

"""Hold the package top and its constraints to one clock per die.

Each die runs on its own ``clk_die_<d>`` port at its ``clock_mhz``; each bridge writes on its
source die's clock and reads on its destination die's. The XDC must create exactly those
clocks, on those ports, and join only clocks it created. A three-die ring whose dies run at
200, 125 and 80 MHz delivers one-hop, two-hop and return packets bit-exactly.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import pytest

from sc_neurocore.chiplet import (
    ChipletDie,
    ChipletGenerator,
    ChipletTopology,
    InterposerLink,
    InterposerTech,
    RoutingTable,
)
from sc_neurocore.chiplet._rtl_links import bridge_timing

REPO_ROOT = Path(__file__).resolve().parents[2]
HDL_PRIMITIVES = (
    REPO_ROOT / "hdl/sc_aer_router.v",
    REPO_ROOT / "hdl/sc_aer_priority_queue.v",
    REPO_ROOT / "hdl/sc_cdc_primitives.v",
)


def _three_clock_ring() -> ChipletTopology:
    return ChipletTopology(
        dies=[
            ChipletDie(die_id=0, clock_mhz=200.0),
            ChipletDie(die_id=1, clock_mhz=125.0),
            ChipletDie(die_id=2, clock_mhz=80.0),
        ],
        links=[InterposerLink(0, 1), InterposerLink(1, 2), InterposerLink(2, 0)],
    )


_TOPOLOGIES = {
    "ring3": ChipletTopology.ring(3, InterposerTech.UCIE),
    "star3": ChipletTopology.star(3, InterposerTech.ORGANIC),
    "three_clocks": _three_clock_ring(),
}


@pytest.mark.parametrize("name", sorted(_TOPOLOGIES))
def test_the_xdc_creates_one_clock_per_die_on_a_top_port_and_joins_only_those(name: str) -> None:
    """Every created clock sits on a top input; every link constraint names created clocks."""
    topology = _TOPOLOGIES[name]
    output = ChipletGenerator().generate(topology)
    top_inputs = set(re.findall(r"input wire (?:\[[^\]]*\] )?(\w+)", output.top_sv))
    created = re.findall(
        r"^create_clock -name (\w+) -period ([0-9.]+) \[get_ports (\w+)\]$",
        output.constraints_xdc,
        re.MULTILINE,
    )
    assert "get_pins" not in output.constraints_xdc
    assert [(n, p) for n, _, p in created] == [
        (f"clk_die_{d.die_id}", f"clk_die_{d.die_id}") for d in topology.dies
    ]
    assert {port for _, _, port in created} <= top_inputs
    for (_, period, _), die in zip(created, topology.dies, strict=True):
        assert period == f"{1000.0 / die.clock_mhz:.3f}"
    names = {n for n, _, _ in created}
    joined = re.findall(r"\[get_clocks (\w+)\]", output.constraints_xdc)
    assert len(joined) == 2 * len(topology.links)
    assert set(joined) <= names
    assert "input wire clk," not in output.top_sv


def test_each_die_and_each_bridge_side_takes_its_own_die_clock_and_reset() -> None:
    """A die runs on its clock and reset; a bridge side takes its own die's clock and reset."""
    topology = _three_clock_ring()
    top = ChipletGenerator().generate(topology).top_sv
    for die in topology.dies:
        d = die.die_id
        assert (
            f"sc_chiplet_die_{d} die_{d}_inst (\n        .clk(clk_die_{d}), .rst_n(rst_n_die_{d}),"
            in top
        )
    for link in topology.links:
        u, v = link.src_die, link.dst_die
        assert (
            f"bridge_{u}_{v}_inst (\n        .src_clk(clk_die_{u}), .src_rst(!rst_n_die_{u}),\n"
            f"        .dst_clk(clk_die_{v}), .dst_rst(!rst_n_die_{v})," in top
        )


def test_rst_n_is_released_in_each_die_domain_by_its_own_synchronizer() -> None:
    """One two-stage synchroniser per die clock; nothing but the synchronisers reads raw rst_n."""
    topology = _three_clock_ring()
    top = ChipletGenerator().generate(topology).top_sv
    for die in topology.dies:
        d = die.die_id
        assert (
            f"sc_sync_2ff rst_sync_die_{d} (\n"
            f"        .clk_dst(clk_die_{d}), .rst_dst(!rst_n), .data_in(1'b1),\n"
            f"        .data_out(rst_n_die_{d})\n    );" in top
        )
    code = "\n".join(line for line in top.splitlines() if not line.lstrip().startswith("//"))
    raw = re.findall(r"(?<![.\w])rst_n\b(?!_die)", code)  # not a port name, not a die reset
    assert len(raw) == 1 + len(topology.dies)  # the port and one synchroniser per die


def test_each_crossing_is_bounded_by_the_faster_clock_and_the_reset_is_a_false_path() -> None:
    """ORGANIC's 5 ns between two 400 MHz dies is bounded at 2.5 ns, not at the link latency."""
    fast = ChipletTopology(
        dies=[ChipletDie(0, clock_mhz=400.0), ChipletDie(1, clock_mhz=400.0)],
        links=[InterposerLink.from_tech(0, 1, InterposerTech.ORGANIC)],
    )
    xdc = ChipletGenerator().generate(fast).constraints_xdc
    assert "set_false_path -from [get_ports rst_n]" in xdc
    bound = "-from [get_clocks clk_die_0] -to [get_clocks clk_die_1] 2.500"
    assert f"set_max_delay -datapath_only {bound}" in xdc
    mixed = ChipletGenerator().generate(_three_clock_ring()).constraints_xdc
    bounds = re.findall(r"-to \[get_clocks (\w+)\] ([0-9.]+)$", mixed, re.MULTILINE)
    assert bounds == [("clk_die_1", "5.000"), ("clk_die_2", "8.000"), ("clk_die_0", "5.000")]


@pytest.mark.parametrize("tech", list(InterposerTech))
def test_at_200_mhz_a_bridge_keeps_the_link_s_own_latency_and_fifo_depth(
    tech: InterposerTech,
) -> None:
    """Two 200 MHz dies reproduce the link's historical latency_cycles and fifo_depth_log2."""
    link = InterposerLink.from_tech(0, 1, tech)
    assert bridge_timing(link, ChipletDie(0), ChipletDie(1)) == (
        link.latency_cycles,
        link.fifo_depth_log2,
    )


def test_bridge_latency_follows_the_destination_clock_and_fifo_the_faster_one() -> None:
    """10 ns is 1 cycle at 100 MHz and 4 at 400 MHz; 10 ns of jitter needs 2**4 at 400 MHz."""
    link = InterposerLink(0, 1, latency_ns=10.0, jitter_ns=10.0)
    slow, fast = ChipletDie(0, clock_mhz=100.0), ChipletDie(1, clock_mhz=400.0)
    assert bridge_timing(link, fast, slow) == (1, 4)
    assert bridge_timing(link, slow, fast) == (4, 4)
    bridge = (
        ChipletGenerator()
        .generate(ChipletTopology(dies=[slow, fast], links=[link]))
        .link_bridges[(0, 1)]
    )
    assert "parameter LATENCY_CYC = 4" in bridge
    assert "4 cycles of die 1's 400.0 MHz clock" in bridge


_TB_THREE_CLOCKS = """\
`timescale 1ns/1ps
module tb_three_clocks;
    reg clk0 = 0, clk1 = 0, clk2 = 0, rst_n = 0;
    always #2.5 clk0 = ~clk0;
    always #4 clk1 = ~clk1;
    always #6.25 clk2 = ~clk2;
    reg s0_v = 0, s1_v = 0, s2_v = 0;
    wire s0_r, s1_r, s2_r, o0_v, o1_v, o2_v;
    reg [9:0] s0_id = 0, s1_id = 0, s2_id = 0;
    wire [9:0] o0_id, o1_id, o2_id;
    wire signed [15:0] o0_w, o1_w, o2_w;
    sc_chiplet_top dut (
        .clk_die_0(clk0), .clk_die_1(clk1), .clk_die_2(clk2), .rst_n(rst_n),
        .die_0_spike_valid(s0_v), .die_0_spike_ready(s0_r), .die_0_spike_id(s0_id),
        .die_0_out_valid(o0_v), .die_0_out_ready(1'b1), .die_0_out_id(o0_id),
        .die_0_out_weight(o0_w),
        .die_1_spike_valid(s1_v), .die_1_spike_ready(s1_r), .die_1_spike_id(s1_id),
        .die_1_out_valid(o1_v), .die_1_out_ready(1'b1), .die_1_out_id(o1_id),
        .die_1_out_weight(o1_w),
        .die_2_spike_valid(s2_v), .die_2_spike_ready(s2_r), .die_2_spike_id(s2_id),
        .die_2_out_valid(o2_v), .die_2_out_ready(1'b1), .die_2_out_id(o2_id),
        .die_2_out_weight(o2_w)
    );
    integer local0 = 0, back0 = 0, got1 = 0, got2 = 0;
    always @(posedge clk0) if (rst_n && o0_v) begin
        if (o0_id === 10'd3 && o0_w === 16'sd64) local0 <= local0 + 1;
        else if (o0_id === 10'd42 && o0_w === -16'sd500) back0 <= back0 + 1;
        else $fatal(1, "die0 got id=%0d w=%0d", o0_id, o0_w);
    end
    always @(posedge clk1) if (rst_n && o1_v) begin
        if (o1_id !== 10'd10 || o1_w !== 16'sd256) $fatal(1, "die1 got id=%0d", o1_id);
        got1 <= got1 + 1;
    end
    always @(posedge clk2) if (rst_n && o2_v) begin
        if (o2_id !== 10'd19 || o2_w !== -16'sd128) $fatal(1, "die2 got id=%0d", o2_id);
        got2 <= got2 + 1;
    end
    task automatic spike0(input [9:0] id);
        begin
            @(posedge clk0); s0_v <= 1; s0_id <= id;
            @(posedge clk0); while (!s0_r) @(posedge clk0);
            s0_v <= 0;
        end
    endtask
    initial begin
        #40 rst_n = 1;
        repeat (4) @(posedge clk2);
        spike0(10'd5);
        repeat (6) @(posedge clk0);
        spike0(10'd7);
        @(posedge clk2); s2_v <= 1; s2_id <= 10'd4;
        @(posedge clk2); while (!s2_r) @(posedge clk2);
        s2_v <= 0;
        repeat (120) @(posedge clk2);
        if (local0 !== 1 || back0 !== 1 || got1 !== 1 || got2 !== 1)
            $fatal(1, "deliveries local0=%0d back0=%0d die1=%0d die2=%0d",
                   local0, back0, got1, got2);
        $display("THREE_CLOCKS_PASS");
        $finish;
    end
endmodule
"""


@pytest.mark.parametrize("decorrelate_links", [False, True])
def test_a_ring_of_three_clock_domains_delivers_every_packet_once(
    tmp_path: Path, decorrelate_links: bool
) -> None:
    """Packets cross 200→125, 125→80 and 80→200 MHz bridges, one and two hops, bit-exactly."""
    assert shutil.which("iverilog") is not None
    rt0, rt2 = RoutingTable(die_id=0), RoutingTable(die_id=2)
    rt0.add_route(5, 1, 10, 256)
    rt0.add_route(5, 2, 19, -128)
    rt0.add_route(7, 0, 3, 64)
    rt2.add_route(4, 0, 42, -500)
    output = ChipletGenerator().generate(
        _three_clock_ring(), routing={0: rt0, 2: rt2}, decorrelate_links=decorrelate_links
    )
    paths = [str(p) for p in HDL_PRIMITIVES]
    for filename, text in output.to_dict().items():
        if filename.endswith(".sv"):
            (tmp_path / filename).write_text(text, encoding="utf-8")
            paths.append(str(tmp_path / filename))
    (tmp_path / "tb.sv").write_text(_TB_THREE_CLOCKS, encoding="utf-8")
    vvp = tmp_path / "sim.vvp"
    comp = subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp), *paths, str(tmp_path / "tb.sv")],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert comp.returncode == 0, comp.stderr
    run = subprocess.run(["vvp", str(vvp)], capture_output=True, text=True, check=False, timeout=60)
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    assert "THREE_CLOCKS_PASS" in run.stdout
