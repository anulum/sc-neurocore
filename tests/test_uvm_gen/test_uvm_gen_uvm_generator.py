# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUVMGenerator from former test_uvm_gen.py

"""Focused suite: TestUVMGenerator from former test_uvm_gen.py."""

from __future__ import annotations

from uvm_gen_support import *  # noqa: F403


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
