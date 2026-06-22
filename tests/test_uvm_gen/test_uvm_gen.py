# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UVM Generator Tests

import pytest

from sc_neurocore.uvm_gen.uvm_gen import (
    CoverageSpec,
    ModulePort,
    PortDirection,
    PortType,
    RTLModule,
    ScoreboardConfig,
    StimulusConfig,
    SIM_TARGETS,
    UVMBenchmark,
    UVMGenerator,
)

# ── Fixtures ─────────────────────────────────────────────────────────

LIF_VERILOG = """\
module sc_lif_neuron #(
    parameter DATA_WIDTH = 16,
    parameter FRACTION = 8,
    parameter V_REST = 0,
    parameter V_THRESHOLD = 16'sd256,
    parameter REFRACTORY_PERIOD = 2
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire signed [15:0] leak_k,
    input  wire signed [15:0] gain_k,
    input  wire signed [15:0] I_t,
    input  wire signed [15:0] noise_in,
    output wire        spike_out,
    output wire signed [15:0] v_out
);
endmodule
"""

DENSE_VERILOG = """\
module sc_dense_layer_core #(
    parameter NUM_NEURONS = 10
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [7:0] input_bus,
    output logic [7:0] output_bus
);
endmodule
"""


def lif_module() -> RTLModule:
    return RTLModule.from_verilog_source(LIF_VERILOG)


def dense_module() -> RTLModule:
    return RTLModule.from_verilog_source(DENSE_VERILOG)


# ── RTLModule Parsing Tests ──────────────────────────────────────────


class TestRTLModuleParsing:
    def test_parse_module_name(self):
        rtl = lif_module()
        assert rtl.name == "sc_lif_neuron"

    def test_parse_ports(self):
        rtl = lif_module()
        assert len(rtl.ports) >= 6

    def test_parse_params(self):
        rtl = lif_module()
        names = [p.name for p in rtl.params]
        assert "DATA_WIDTH" in names
        assert "V_THRESHOLD" in names

    def test_clock_detection(self):
        rtl = lif_module()
        assert rtl.clock_port is not None
        assert rtl.clock_port.name == "clk"

    def test_reset_detection(self):
        rtl = lif_module()
        assert rtl.reset_port is not None
        assert rtl.reset_port.name == "rst_n"

    def test_input_ports_exclude_clock_reset(self):
        rtl = lif_module()
        names = [p.name for p in rtl.input_ports]
        assert "clk" not in names
        assert "rst_n" not in names
        assert "I_t" in names

    def test_output_ports(self):
        rtl = lif_module()
        names = [p.name for p in rtl.output_ports]
        assert "spike_out" in names
        assert "v_out" in names

    def test_signed_ports(self):
        rtl = lif_module()
        it = next(p for p in rtl.ports if p.name == "I_t")
        assert it.is_signed is True

    def test_port_width(self):
        rtl = lif_module()
        it = next(p for p in rtl.ports if p.name == "I_t")
        assert it.width == 16

    def test_total_input_bits(self):
        rtl = lif_module()
        assert rtl.total_input_bits > 0

    def test_total_output_bits(self):
        rtl = lif_module()
        assert rtl.total_output_bits > 0

    def test_dense_module(self):
        rtl = dense_module()
        assert rtl.name == "sc_dense_layer_core"
        assert len(rtl.input_ports) == 1
        assert rtl.input_ports[0].name == "input_bus"
        assert rtl.input_ports[0].width == 8

    def test_no_module_raises(self):
        with pytest.raises(ValueError, match="No module"):
            RTLModule.from_verilog_source("// empty file")

    def test_sv_decl(self):
        p = ModulePort("foo", PortDirection.INPUT, PortType.LOGIC, 8, False)
        assert "input" in p.sv_decl
        assert "[7:0]" in p.sv_decl


# ── UVM Generator Tests ─────────────────────────────────────────────


class TestUVMGenerator:
    def test_generate_returns_benchmark(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert isinstance(bench, UVMBenchmark)
        assert bench.module_name == "sc_lif_neuron"

    def test_transaction_has_fields(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "I_t" in bench.transaction_sv
        assert "leak_k" in bench.transaction_sv
        assert "uvm_sequence_item" in bench.transaction_sv

    def test_transaction_has_constraints(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "constraint" in bench.transaction_sv

    def test_sequence_has_random(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "random_seq" in bench.sequence_sv
        assert "body" in bench.sequence_sv

    def test_sequence_has_corner(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "corner_seq" in bench.sequence_sv

    def test_sequence_has_lfsr(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "lfsr_seq" in bench.sequence_sv
        assert "lfsr[15]" in bench.sequence_sv

    def test_driver_has_drive_logic(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "driver" in bench.driver_sv
        assert "vif" in bench.driver_sv
        assert "I_t" in bench.driver_sv

    def test_monitor_samples_ports(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "monitor" in bench.monitor_sv
        assert "spike_out" in bench.monitor_sv
        assert "analysis_port" in bench.monitor_sv

    def test_scoreboard_checks(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "scoreboard" in bench.scoreboard_sv
        assert "transaction_count" in bench.scoreboard_sv

    def test_scoreboard_spike_check(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "spike_count" in bench.scoreboard_sv

    def test_coverage_coverpoints(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "covergroup" in bench.coverage_sv
        assert "coverpoint" in bench.coverage_sv

    def test_coverage_toggle(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "spike_out_toggle" in bench.coverage_sv

    def test_agent_connections(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "agent" in bench.agent_sv
        assert "drv" in bench.agent_sv
        assert "mon" in bench.agent_sv
        assert "sqr" in bench.agent_sv

    def test_env_wiring(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "env" in bench.env_sv
        assert "connect_phase" in bench.env_sv
        assert "agt.mon.ap.connect" in bench.env_sv

    def test_top_has_dut(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "sc_lif_neuron" in bench.top_sv
        assert "module tb_sc_lif_neuron_top" in bench.top_sv

    def test_top_has_clock_gen(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "always #5" in bench.top_sv

    def test_top_has_reset(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "rst_n = 0" in bench.top_sv
        assert "rst_n = 1" in bench.top_sv

    def test_top_has_test_class(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "uvm_test" in bench.top_sv
        assert "run_test" in bench.top_sv

    def test_top_has_interface(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "interface sc_lif_neuron_if" in bench.top_sv

    def test_sby_config_valid(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "[options]" in bench.sby_config
        assert "prove" in bench.sby_config
        assert "smtbmc" in bench.sby_config

    def test_filelist(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert len(bench.filelist) == 11
        assert "sc_lif_neuron_transaction.sv" in bench.filelist
        assert "sc_lif_neuron_bind.sv" in bench.filelist

    def test_to_dict_keys(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        d = bench.to_dict()
        assert "sc_lif_neuron_transaction.sv" in d
        assert "tb_sc_lif_neuron_top.sv" in d

    def test_spdx_in_all_files(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        skip_ext = (".sby", ".list")
        for fname, content in bench.to_dict().items():
            if fname.endswith(skip_ext) or fname == "Makefile":
                continue
            assert "SPDX" in content, f"missing SPDX in {fname}"

    def test_dense_layer_generation(self):
        gen = UVMGenerator()
        bench = gen.generate(dense_module())
        assert "sc_dense_layer_core" in bench.top_sv
        assert "input_bus" in bench.driver_sv


# ── Configuration Tests ──────────────────────────────────────────────


class TestConfiguration:
    def test_custom_stimulus(self):
        stim = StimulusConfig(num_transactions=500, bitstream_density_range=(0.2, 0.8))
        gen = UVMGenerator(stimulus=stim)
        bench = gen.generate(lif_module())
        assert "500" in bench.sequence_sv

    def test_custom_coverage(self):
        cov = CoverageSpec(bitstream_density_bins=20)
        gen = UVMGenerator(coverage=cov)
        bench = gen.generate(lif_module())
        assert "20" in bench.coverage_sv

    def test_scoreboard_config(self):
        sb = ScoreboardConfig(check_popcount=True, check_spike_timing=True)
        gen = UVMGenerator(scoreboard=sb)
        bench = gen.generate(lif_module())
        assert "popcount" in bench.scoreboard_sv or "spike" in bench.scoreboard_sv


# ── Formal Link Tests ────────────────────────────────────────────────


class TestFormalLinks:
    def test_generate_links(self):
        gen = UVMGenerator()
        rtl = lif_module()
        links = gen.generate_formal_links(rtl)
        assert len(links) > 0

    def test_link_has_assertion(self):
        gen = UVMGenerator()
        links = gen.generate_formal_links(lif_module())
        for link in links:
            assert "assert property" in link.assertion_sv

    def test_link_has_cover(self):
        gen = UVMGenerator()
        links = gen.generate_formal_links(lif_module())
        for link in links:
            assert "cover property" in link.cover_sv

    def test_link_references_reset(self):
        gen = UVMGenerator()
        links = gen.generate_formal_links(lif_module())
        any_rst = any("rst_n" in link.assertion_sv for link in links)
        assert any_rst


# ── Golden Model Scoreboard Tests ────────────────────────────────────


class TestGoldenModelScoreboard:
    def test_golden_comparison_enabled_with_explicit_reference(self):
        sb = ScoreboardConfig(
            check_golden_comparison=True,
            golden_expressions={"v_out": "txn.I_t"},
        )
        gen = UVMGenerator(scoreboard=sb)
        bench = gen.generate(lif_module())
        assert "golden_compute" in bench.scoreboard_sv
        assert "return txn.I_t;" in bench.scoreboard_sv

    def test_golden_vars_present(self):
        sb = ScoreboardConfig(
            check_golden_comparison=True,
            golden_expressions={"v_out": "txn.I_t"},
        )
        gen = UVMGenerator(scoreboard=sb)
        bench = gen.generate(lif_module())
        assert "expected_v_out" in bench.scoreboard_sv

    def test_mismatch_detection(self):
        sb = ScoreboardConfig(
            check_golden_comparison=True,
            golden_expressions={"v_out": "txn.I_t"},
        )
        gen = UVMGenerator(scoreboard=sb)
        bench = gen.generate(lif_module())
        assert "MISMATCH" in bench.scoreboard_sv

    def test_golden_comparison_requires_explicit_reference(self):
        sb = ScoreboardConfig(check_golden_comparison=True)
        gen = UVMGenerator(scoreboard=sb)
        with pytest.raises(ValueError, match="Missing golden reference expression"):
            gen.generate(lif_module())

    def test_golden_disabled(self):
        sb = ScoreboardConfig(check_golden_comparison=False)
        gen = UVMGenerator(scoreboard=sb)
        bench = gen.generate(lif_module())
        assert "golden_compute" not in bench.scoreboard_sv


# ── SCC Coverage Tests ───────────────────────────────────────────────


class TestSCCCoverage:
    def test_scc_bins_generated(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "scc" in bench.coverage_sv.lower()

    def test_scc_bins_count(self):
        cov = CoverageSpec(scc_bins=12)
        gen = UVMGenerator(coverage=cov)
        bench = gen.generate(lif_module())
        assert "12" in bench.coverage_sv


# ── Toggle / Activity Coverage Tests ─────────────────────────────────


class TestToggleCoverage:
    def test_activity_bins(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "activity" in bench.coverage_sv

    def test_toggle_disabled(self):
        cov = CoverageSpec(toggle_coverage=False)
        gen = UVMGenerator(coverage=cov)
        bench = gen.generate(lif_module())
        assert "activity" not in bench.coverage_sv


# ── Coverage Target Tests ────────────────────────────────────────────


class TestCoverageTarget:
    def test_target_percent_in_coverage(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "coverage_target" in bench.coverage_sv
        assert "95.0" in bench.coverage_sv

    def test_custom_target(self):
        cov = CoverageSpec(target_percent=99.0)
        gen = UVMGenerator(coverage=cov)
        bench = gen.generate(lif_module())
        assert "99.0" in bench.coverage_sv

    def test_warning_on_miss(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "uvm_warning" in bench.coverage_sv.lower() or "warning" in bench.coverage_sv.lower()


# ── Assertion Bind Module Tests ──────────────────────────────────────


class TestAssertionBind:
    def test_bind_generated(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert len(bench.bind_sv) > 0

    def test_bind_has_assertions(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "assert property" in bench.bind_sv

    def test_bind_has_cover(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "cover property" in bench.bind_sv

    def test_bind_has_reset_check(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "rst_n" in bench.bind_sv

    def test_bind_module_name(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "sc_lif_neuron_assertions" in bench.bind_sv

    def test_bind_in_dict(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        d = bench.to_dict()
        assert "sc_lif_neuron_bind.sv" in d


# ── Makefile Generator Tests ─────────────────────────────────────────


class TestMakefileGenerator:
    def test_makefile_generated(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert len(bench.makefile) > 0

    def test_makefile_has_targets(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "compile:" in bench.makefile
        assert "sim:" in bench.makefile
        assert "coverage:" in bench.makefile
        assert "clean:" in bench.makefile

    def test_makefile_has_regression(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "regression:" in bench.makefile

    def test_sim_targets_defined(self):
        assert "vcs" in SIM_TARGETS
        assert "questa" in SIM_TARGETS
        assert "xcelium" in SIM_TARGETS


# ── Regression List Tests ────────────────────────────────────────────


class TestRegressionList:
    def test_regression_list_generated(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert len(bench.regression_list) > 0

    def test_regression_has_tests(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "random" in bench.regression_list
        assert "corner" in bench.regression_list
        assert "lfsr" in bench.regression_list

    def test_regression_in_dict(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        d = bench.to_dict()
        assert "regression.list" in d


# ── Multi-DUT Tests ──────────────────────────────────────────────────


class TestMultiDUT:
    def test_generate_multi(self):
        gen = UVMGenerator()
        benchmarks = gen.generate_multi([lif_module(), dense_module()])
        assert len(benchmarks) == 2
        assert benchmarks[0].module_name == "sc_lif_neuron"
        assert benchmarks[1].module_name == "sc_dense_layer_core"

    def test_multi_independent(self):
        gen = UVMGenerator()
        benchmarks = gen.generate_multi([lif_module(), dense_module()])
        assert benchmarks[0].top_sv != benchmarks[1].top_sv


PARAMLESS_VERILOG_WITH_BLANK_PORT = """\
module sc_paramless (
    input  wire clk,
    ,
    output wire done
);
endmodule
"""


def test_from_verilog_source_handles_paramless_module_and_blank_port_entries():
    # No `#(...)` block exercises the parameter-less port-section branch, and the
    # stray comma yields a blank port entry that must be skipped, not parsed.
    module = RTLModule.from_verilog_source(PARAMLESS_VERILOG_WITH_BLANK_PORT)

    assert module.name == "sc_paramless"
    port_names = {port.name for port in module.ports}
    assert {"clk", "done"} <= port_names
    assert "" not in port_names
