# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet RTL generation and multi-die AER co-simulation contracts

"""End-to-end tests and Icarus Verilog co-simulations for multi-die AER RTL."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from sc_neurocore.chiplet import (
    ChipletDie,
    ChipletGenerator,
    ChipletOutput,
    ChipletTopology,
    InterposerLink,
    InterposerTech,
    RoutingTable,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HDL_PRIMITIVES = (
    REPO_ROOT / "hdl/sc_aer_router.v",
    REPO_ROOT / "hdl/sc_aer_priority_queue.v",
    REPO_ROOT / "hdl/sc_cdc_primitives.v",
)


def _generated_ring(technology: InterposerTech = InterposerTech.UCIE) -> ChipletOutput:
    return ChipletGenerator().generate(ChipletTopology.ring(2, technology))


def _compile_and_run(tmp_path: Path, sv_files: dict[str, str], tb_source: str) -> str:
    assert shutil.which("iverilog") is not None
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: list[str] = [str(p) for p in HDL_PRIMITIVES]
    for filename, text in sv_files.items():
        if filename.endswith(".sv"):
            dest = tmp_path / filename
            dest.write_text(text, encoding="utf-8")
            paths.append(str(dest))
    tb_path = tmp_path / "tb_chiplet_sim.sv"
    tb_path.write_text(tb_source, encoding="utf-8")
    paths.append(str(tb_path))
    vvp_path = tmp_path / "sim.vvp"
    comp = subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp_path), *paths],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert comp.returncode == 0, comp.stderr
    run = subprocess.run(
        ["vvp", str(vvp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    return run.stdout


def test_generation_returns_complete_named_artefact_set() -> None:
    """Emit top, die wrappers, link bridges, constraints, and matching filelist."""
    output = _generated_ring()
    assert isinstance(output, ChipletOutput)
    assert output.filelist == list(output.to_dict())
    assert "sc_chiplet_top.sv" in output.filelist
    assert "chiplet_constraints.xdc" in output.filelist


def test_top_connects_die_and_bridge_stream_ports() -> None:
    """Expose per-die AER ports on sc_chiplet_top and wire AXI-Stream bridges."""
    top = _generated_ring().top_sv
    assert "module sc_chiplet_top" in top
    assert "Dies: 2" in top
    assert "die_0_spike_valid" in top
    assert "die_1_out_weight" in top
    assert ".link_out_1_tdata(link_0_1_tdata)" in top
    assert ".s_tdata(link_0_1_tdata)" in top
    assert ".m_tdata(link_0_1_rx_tdata)" in top
    assert "TODO" not in top


def test_die_wrapper_wires_router_tables_and_arbitrated_backpressure() -> None:
    """Bind sc_aer_router fanout arrays and honour outgoing/local ready signals."""
    die = _generated_ring().die_modules[0]
    assert "sc_aer_router" in die
    assert ".fanout_count(rt_fanout_count)" in die
    assert ".target_id(rt_target_id)" in die
    assert ".target_weight(rt_target_weight)" in die
    assert "link_out_1_tready" in die
    assert "local_out_ready" in die


def test_bridge_stalls_pipeline_on_backpressure_and_preserves_binary_aer_by_default() -> None:
    """Gate delay_pipe advancement by pipe_advance and disable XOR unless requested."""
    bridge = _generated_ring(InterposerTech.EMIB).link_bridges[(0, 1)]
    assert "sc_async_fifo" in bridge
    assert "pipe_advance" in bridge
    assert "delay_pipe" in bridge
    assert "LATENCY_CYC" in bridge
    assert "tx_scrambled" in bridge
    assert "rx_descrambled" in bridge
    assert "parameter DECORRELATE_PAYLOAD = 0" in bridge
    assert "EMIB" in bridge


def test_routing_table_and_datapath_only_constraints_are_emitted(tmp_path: Path) -> None:
    """Size sc_chiplet_rtable by die.n_neurons, format negative weights, and simulate lookup."""
    topology = ChipletTopology.ring(2)
    table = RoutingTable(die_id=0)
    table.add_route(5, 1, 10, -128)
    table.add_route(5, 1, 11, 64)
    empty_table = RoutingTable(die_id=1)
    output = ChipletGenerator().generate(topology, routing={0: table, 1: empty_table})
    rtable0 = output.routing_tables[0]
    assert "parameter N_ENTRIES = 128" in rtable0
    assert "-16'sd128" in rtable0
    assert "No inter-die routes" in output.routing_tables[1]
    assert "create_clock" in output.constraints_xdc
    assert "set_max_delay -datapath_only" in output.constraints_xdc

    tb = """\
module tb_rtable;
    reg clk = 0;
    always #2 clk = ~clk;
    reg rst_n = 0;
    reg [9:0] query_neuron = 0;
    wire route_valid;
    wire [3:0] target_die;
    wire [9:0] target_neuron;
    wire signed [15:0] weight;

    sc_chiplet_rtable_0 dut (
        .clk(clk), .rst_n(rst_n), .query_neuron(query_neuron),
        .route_valid(route_valid), .target_die(target_die),
        .target_neuron(target_neuron), .weight(weight)
    );

    initial begin
        #6 rst_n = 1;
        query_neuron = 10'd5;
        @(posedge clk); #1;
        if (!route_valid || target_die !== 4'd1 || target_neuron !== 10'd10 || weight !== -16'sd128)
            $fatal(1, "mapped lookup mismatch");
        query_neuron = 10'd99;
        @(posedge clk); #1;
        if (route_valid !== 1'b0 || weight !== 16'sd0)
            $fatal(1, "unmapped lookup mismatch");
        $display("RTABLE_PASS");
        $finish;
    end
endmodule
"""
    stdout = _compile_and_run(tmp_path, {"sc_chiplet_rtable_0.sv": rtable0}, tb)
    assert "RTABLE_PASS" in stdout


def test_every_generated_systemverilog_file_has_spdx_header() -> None:
    """Include the canonical SPDX identifier in every generated SystemVerilog file."""
    for name, content in _generated_ring().to_dict().items():
        if name.endswith(".sv"):
            assert ": ".join(("SPDX-License-Identifier", "AGPL-3.0-or-later")) in content, name


@pytest.mark.parametrize("decorrelate_links", [False, True])
def test_cosim_two_die_fanout_and_three_die_multihop_delivery(
    tmp_path: Path, decorrelate_links: bool
) -> None:
    """Deliver multi-fanout signed AER packets across 1-hop and 2-hop ring bridges."""
    topo = ChipletTopology.ring(3, InterposerTech.UCIE)
    rt0 = RoutingTable(die_id=0)
    rt0.add_route(5, 1, 10, 256)
    rt0.add_route(5, 2, 19, -128)
    rt0.add_route(7, 0, 3, 64)
    output = ChipletGenerator().generate(
        topo, routing={0: rt0}, decorrelate_links=decorrelate_links
    )

    tb = """\
module tb_ring3;
    reg clk = 0;
    always #2 clk = ~clk;
    reg rst_n = 0;

    reg die_0_spike_valid = 0, die_1_spike_valid = 0, die_2_spike_valid = 0;
    wire die_0_spike_ready, die_1_spike_ready, die_2_spike_ready;
    reg [9:0] die_0_spike_id = 0, die_1_spike_id = 0, die_2_spike_id = 0;
    wire die_0_out_valid, die_1_out_valid, die_2_out_valid;
    reg die_0_out_ready = 1, die_1_out_ready = 1, die_2_out_ready = 1;
    wire [9:0] die_0_out_id, die_1_out_id, die_2_out_id;
    wire signed [15:0] die_0_out_weight, die_1_out_weight, die_2_out_weight;

    sc_chiplet_top dut (
        .clk_die_0(clk), .clk_die_1(clk), .clk_die_2(clk), .rst_n(rst_n),
        .die_0_spike_valid(die_0_spike_valid), .die_0_spike_ready(die_0_spike_ready),
        .die_0_spike_id(die_0_spike_id), .die_0_out_valid(die_0_out_valid),
        .die_0_out_ready(die_0_out_ready), .die_0_out_id(die_0_out_id),
        .die_0_out_weight(die_0_out_weight),
        .die_1_spike_valid(die_1_spike_valid), .die_1_spike_ready(die_1_spike_ready),
        .die_1_spike_id(die_1_spike_id), .die_1_out_valid(die_1_out_valid),
        .die_1_out_ready(die_1_out_ready), .die_1_out_id(die_1_out_id),
        .die_1_out_weight(die_1_out_weight),
        .die_2_spike_valid(die_2_spike_valid), .die_2_spike_ready(die_2_spike_ready),
        .die_2_spike_id(die_2_spike_id), .die_2_out_valid(die_2_out_valid),
        .die_2_out_ready(die_2_out_ready), .die_2_out_id(die_2_out_id),
        .die_2_out_weight(die_2_out_weight)
    );

    integer got0 = 0, got1 = 0, got2 = 0, cyc;
    always @(posedge clk) if (rst_n) begin
        if (die_0_out_valid && die_0_out_ready) begin
            if (die_0_out_id !== 10'd3 || die_0_out_weight !== 16'sd64)
                $fatal(1, "die0 local route mismatch");
            got0 <= got0 + 1;
        end
        if (die_1_out_valid && die_1_out_ready) begin
            if (die_1_out_id !== 10'd10 || die_1_out_weight !== 16'sd256)
                $fatal(1, "die1 hop1 mismatch");
            got1 <= got1 + 1;
        end
        if (die_2_out_valid && die_2_out_ready) begin
            if (die_2_out_id !== 10'd19 || die_2_out_weight !== -16'sd128)
                $fatal(1, "die2 hop2 mismatch");
            got2 <= got2 + 1;
        end
    end

    initial begin
        #10 rst_n = 1;
        repeat (3) @(posedge clk); // per-die reset synchronisers
        die_0_spike_valid <= 1;
        die_0_spike_id <= 10'd5;
        @(posedge clk);
        die_0_spike_valid <= 0;
        wait (die_0_spike_ready == 1'b1);
        @(posedge clk);
        wait (die_0_spike_ready == 1'b1);
        die_0_spike_valid <= 1;
        die_0_spike_id <= 10'd7;
        @(posedge clk);
        die_0_spike_valid <= 0;

        for (cyc = 0; cyc < 80; cyc = cyc + 1) @(posedge clk);
        if (got0 !== 1 || got1 !== 1 || got2 !== 1)
            $fatal(1, "expected 1 delivery per die, got %0d %0d %0d", got0, got1, got2);
        $display("RING3_PASS");
        $finish;
    end
endmodule
"""
    stdout = _compile_and_run(tmp_path, output.to_dict(), tb)
    assert "RING3_PASS" in stdout


@pytest.mark.parametrize("decorrelate_links", [False, True])
def test_cosim_star_contention_and_extended_backpressure_without_loss(
    tmp_path: Path, decorrelate_links: bool
) -> None:
    """Hold packets across simultaneous spoke arrivals and 18-cycle hub backpressure."""
    topo = ChipletTopology.star(3, InterposerTech.ORGANIC)
    rt1 = RoutingTable(die_id=1)
    rt1.add_route(2, 0, 21, 300)
    rt2 = RoutingTable(die_id=2)
    rt2.add_route(4, 0, 42, -500)
    output = ChipletGenerator().generate(
        topo, routing={1: rt1, 2: rt2}, decorrelate_links=decorrelate_links
    )

    tb = """\
module tb_star_bp;
    reg clk = 0;
    always #2 clk = ~clk;
    reg rst_n = 0;

    reg s0_v = 0, s1_v = 0, s2_v = 0;
    wire s0_r, s1_r, s2_r;
    reg [9:0] s0_id = 0, s1_id = 0, s2_id = 0;
    wire o0_v, o1_v, o2_v;
    reg o0_r = 0, o1_r = 1, o2_r = 1;
    wire [9:0] o0_id, o1_id, o2_id;
    wire signed [15:0] o0_w, o1_w, o2_w;

    sc_chiplet_top dut (
        .clk_die_0(clk), .clk_die_1(clk), .clk_die_2(clk), .rst_n(rst_n),
        .die_0_spike_valid(s0_v), .die_0_spike_ready(s0_r), .die_0_spike_id(s0_id),
        .die_0_out_valid(o0_v), .die_0_out_ready(o0_r), .die_0_out_id(o0_id), .die_0_out_weight(o0_w),
        .die_1_spike_valid(s1_v), .die_1_spike_ready(s1_r), .die_1_spike_id(s1_id),
        .die_1_out_valid(o1_v), .die_1_out_ready(o1_r), .die_1_out_id(o1_id), .die_1_out_weight(o1_w),
        .die_2_spike_valid(s2_v), .die_2_spike_ready(s2_r), .die_2_spike_id(s2_id),
        .die_2_out_valid(o2_v), .die_2_out_ready(o2_r), .die_2_out_id(o2_id), .die_2_out_weight(o2_w)
    );

    integer seen21 = 0, seen42 = 0, c;
    always @(posedge clk) if (rst_n && o0_v && o0_r) begin
        if (o0_id === 10'd21 && o0_w === 16'sd300) seen21 <= seen21 + 1;
        else if (o0_id === 10'd42 && o0_w === -16'sd500) seen42 <= seen42 + 1;
        else $fatal(1, "unexpected hub event id=%0d w=%0d", o0_id, o0_w);
    end

    initial begin
        #10 rst_n = 1;
        repeat (3) @(posedge clk); // per-die reset synchronisers
        s1_v <= 1; s1_id <= 10'd2;
        s2_v <= 1; s2_id <= 10'd4;
        @(posedge clk);
        s1_v <= 0; s2_v <= 0;
        for (c = 0; c < 18; c = c + 1) @(posedge clk);
        o0_r <= 1;
        for (c = 0; c < 20; c = c + 1) @(posedge clk);
        if (seen21 !== 1 || seen42 !== 1)
            $fatal(1, "star contention loss/dup: seen21=%0d seen42=%0d", seen21, seen42);
        $display("STAR_BP_PASS");
        $finish;
    end
endmodule
"""
    stdout = _compile_and_run(tmp_path, output.to_dict(), tb)
    assert "STAR_BP_PASS" in stdout


def test_exact_packet_width_isolated_die_and_decorrelation_flag(tmp_path: Path) -> None:
    """Scramble in-flight link words and transparently descramble back-to-back stalled payloads."""
    exact_topo = ChipletTopology(
        dies=[ChipletDie(die_id=0), ChipletDie(die_id=1), ChipletDie(die_id=2)],
        links=[InterposerLink(src_die=0, dst_die=1, data_width=30)],
    )
    out = ChipletGenerator().generate(exact_topo, decorrelate_links=True)
    assert "{0{1'b0}}" not in out.die_modules[0]
    assert "router_ev_ready = 1'b1;" in out.die_modules[2]
    assert "parameter DECORRELATE_PAYLOAD = 1" in out.link_bridges[(0, 1)]

    tb = """\
module tb_decor;
    reg src_clk = 0, dst_clk = 0, rst = 1;
    always #2 src_clk = ~src_clk;
    always #3 dst_clk = ~dst_clk;
    reg [29:0] s_tdata = 30'd0;
    reg s_tvalid = 0, m_tready = 0;
    wire s_tready, m_tvalid;
    wire [29:0] m_tdata;
    sc_chiplet_bridge_0_to_1 dut (
        .src_clk(src_clk), .src_rst(rst), .dst_clk(dst_clk), .dst_rst(rst),
        .s_tdata(s_tdata), .s_tvalid(s_tvalid), .s_tready(s_tready),
        .m_tdata(m_tdata), .m_tvalid(m_tvalid), .m_tready(m_tready)
    );
    integer k, got = 0;
    always @(posedge dst_clk) if (!rst && m_tvalid && m_tready) begin
        if (got == 0 && (dut.delay_pipe[0] === 30'h123456 || m_tdata !== 30'h123456))
            $fatal(1, "pkt0 scramble/descramble failed");
        if (got == 1 && (dut.delay_pipe[0] === 30'h2ABCDE || m_tdata !== 30'h2ABCDE))
            $fatal(1, "pkt1 scramble/descramble failed");
        got <= got + 1;
    end
    initial begin
        #12 rst = 0;
        @(posedge src_clk); s_tdata <= 30'h123456; s_tvalid <= 1;
        @(posedge src_clk); s_tdata <= 30'h2ABCDE;
        @(posedge src_clk); s_tvalid <= 0;
        for (k = 0; k < 6; k = k + 1) @(posedge dst_clk);
        m_tready <= 1;
        for (k = 0; k < 20; k = k + 1) @(posedge dst_clk);
        if (got !== 2) $fatal(1, "expected 2 descrambled packets, got %0d", got);
        $display("DECOR_PASS");
        $finish;
    end
endmodule
"""
    stdout = _compile_and_run(
        tmp_path / "decor", {"sc_chiplet_bridge_0_to_1.sv": out.link_bridges[(0, 1)]}, tb
    )
    assert "DECOR_PASS" in stdout


def test_heterogeneous_die_widths_in_routing_table_and_package_cosim(tmp_path: Path) -> None:
    """Size sc_chiplet_rtable target_neuron by PKG_AER_W and co-simulate heterogeneous dies."""
    topo = ChipletTopology(
        dies=[
            ChipletDie(die_id=0, n_neurons=16, aer_id_width=4, data_width=12),
            ChipletDie(die_id=1, n_neurons=1024, aer_id_width=10, data_width=16),
        ],
        links=[InterposerLink(0, 1, data_width=32), InterposerLink(1, 0, data_width=32)],
    )
    rt0, rt1 = RoutingTable(die_id=0), RoutingTable(die_id=1)
    rt0.add_route(3, 1, 100, -200)
    rt1.add_route(100, 0, 11, 150)
    out = ChipletGenerator().generate(topo, routing={0: rt0, 1: rt1}, decorrelate_links=True)
    assert "parameter PKG_AER_W = 10" in out.routing_tables[0]
    assert "rt_target_neuron[3] = 10'd100;" in out.routing_tables[0]

    tb = """\
module tb_hetero;
    reg clk = 0, rst_n = 0;
    always #2 clk = ~clk;
    wire route_valid; wire [3:0] target_die; wire [9:0] target_neuron; wire signed [11:0] weight;
    sc_chiplet_rtable_0 rt0 (
        .clk(clk), .rst_n(rst_n), .query_neuron(4'd3),
        .route_valid(route_valid), .target_die(target_die),
        .target_neuron(target_neuron), .weight(weight)
    );
    reg s0_v = 0, s1_v = 0; wire s0_r, s1_r, o0_v, o1_v;
    wire [3:0] o0_id; wire [9:0] o1_id;
    wire signed [11:0] o0_w; wire signed [15:0] o1_w;
    sc_chiplet_top top (
        .clk_die_0(clk), .clk_die_1(clk), .rst_n(rst_n),
        .die_0_spike_valid(s0_v), .die_0_spike_ready(s0_r), .die_0_spike_id(4'd3),
        .die_0_out_valid(o0_v), .die_0_out_ready(1'b1), .die_0_out_id(o0_id), .die_0_out_weight(o0_w),
        .die_1_spike_valid(s1_v), .die_1_spike_ready(s1_r), .die_1_spike_id(10'd100),
        .die_1_out_valid(o1_v), .die_1_out_ready(1'b1), .die_1_out_id(o1_id), .die_1_out_weight(o1_w)
    );
    integer g0 = 0, g1 = 0, c;
    always @(posedge clk) if (rst_n) begin
        if (o1_v && (o1_id !== 10'd100 || o1_w !== -16'sd200)) $fatal(1, "0->1 mismatch");
        if (o1_v) g1 <= g1 + 1;
        if (o0_v && (o0_id !== 4'd11 || o0_w !== 12'sd150)) $fatal(1, "1->0 mismatch");
        if (o0_v) g0 <= g0 + 1;
    end
    initial begin
        #8 rst_n = 1;
        repeat (3) @(posedge clk); // per-die reset synchronisers
        s0_v <= 1; s1_v <= 1;
        @(posedge clk); s0_v <= 0; s1_v <= 0;
        if (!route_valid || target_die !== 4'd1 || target_neuron !== 10'd100 || weight !== -12'sd200)
            $fatal(1, "rtable0 truncated target_neuron=%0d", target_neuron);
        for (c = 0; c < 40; c = c + 1) @(posedge clk);
        if (g0 !== 1 || g1 !== 1) $fatal(1, "hetero counts %0d %0d", g0, g1);
        $display("HETERO_PASS");
        $finish;
    end
endmodule
"""
    stdout = _compile_and_run(tmp_path / "hetero", out.to_dict(), tb)
    assert "HETERO_PASS" in stdout


def test_invalid_topologies_and_routes_fail_closed() -> None:
    """Reject duplicate dies/links, narrow links, out-of-range neurons/weights, and broken paths."""
    gen = ChipletGenerator()
    with pytest.raises(ValueError, match="Duplicate die_id"):
        gen.generate(ChipletTopology(dies=[ChipletDie(0), ChipletDie(0)]))
    with pytest.raises(ValueError, match="absent die_id"):
        gen.generate(ChipletTopology(dies=[ChipletDie(0)], links=[InterposerLink(0, 9)]))
    with pytest.raises(ValueError, match="Duplicate directed link"):
        gen.generate(
            ChipletTopology(
                dies=[ChipletDie(0), ChipletDie(1)],
                links=[InterposerLink(0, 1), InterposerLink(0, 1)],
            )
        )
    with pytest.raises(ValueError, match="data_width=16 < 30"):
        gen.generate(
            ChipletTopology(
                dies=[ChipletDie(0), ChipletDie(1)],
                links=[InterposerLink(0, 1, data_width=16)],
            )
        )

    ring2 = ChipletTopology.ring(2)
    with pytest.raises(ValueError, match="Invalid routing table"):
        gen.generate(ring2, routing={9: RoutingTable(9)})
    with pytest.raises(ValueError, match="Invalid routing table"):
        gen.generate(ring2, routing={0: RoutingTable(1)})

    for src, dst_d, dst_n, weight, pat in [
        (200, 1, 0, 64, "src_neuron"),
        (0, 9, 0, 64, "dst_die 9 is absent"),
        (0, 1, 200, 64, "dst_neuron"),
        (0, 1, 0, 99999, "weight_q88"),
    ]:
        bad = RoutingTable(0)
        bad.add_route(src, dst_d, dst_n, weight)
        with pytest.raises(ValueError, match=pat):
            gen.generate(ring2, routing={0: bad})

    disconnected = ChipletTopology(dies=[ChipletDie(0), ChipletDie(1)])
    unreachable = RoutingTable(0)
    unreachable.add_route(0, 1, 0, 64)
    with pytest.raises(ValueError, match="No directed link path"):
        gen.generate(disconnected, routing={0: unreachable})
