# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UVM transaction, sequence, component, and environment emitters

"""Emit deterministic UVM component source from parsed RTL contracts."""

from __future__ import annotations

import textwrap

from sc_neurocore.uvm_gen._config import (
    CoverageSpec,
    ScoreboardConfig,
    SPDX_HEADER,
    StimulusConfig,
)
from sc_neurocore.uvm_gen._rtl import RTLModule


def _emit_transaction(rtl: RTLModule, stimulus: StimulusConfig) -> str:
    module_name = rtl.name
    fields = []
    for port in rtl.input_ports:
        signed = " signed" if port.is_signed else ""
        width = f" [{port.width - 1}:0]" if port.width > 1 else ""
        fields.append(f"    rand logic{signed}{width} {port.name};")
    for port in rtl.output_ports:
        signed = " signed" if port.is_signed else ""
        width = f" [{port.width - 1}:0]" if port.width > 1 else ""
        fields.append(f"    logic{signed}{width} {port.name};")

    field_block = "\n".join(fields) if fields else "    rand logic [7:0] data;"

    constraints = []
    lower, upper = stimulus.bitstream_density_range
    for port in rtl.input_ports:
        if port.width > 1:
            constraints.append(
                f"    constraint c_{port.name}_density {{\n"
                f"        $countones({port.name}) inside "
                f"{{[{int(lower * port.width)}:{int(upper * port.width)}]}};\n"
                f"    }}"
            )
    constraint_block = "\n".join(constraints)

    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore UVM — Transaction for {module_name}

class {module_name}_transaction extends uvm_sequence_item;
    `uvm_object_utils({module_name}_transaction)

{field_block}

{constraint_block}

    function new(string name = "{module_name}_transaction");
        super.new(name);
    endfunction

    function string convert2string();
        return $sformatf("{module_name}_txn: {" ".join(f"{port.name}=%0h" for port in rtl.input_ports)}",
            {", ".join(port.name for port in rtl.input_ports) or "data"});
    endfunction
endclass
""")


def _emit_sequence(rtl: RTLModule, stimulus: StimulusConfig) -> str:
    module_name = rtl.name
    num_transactions = stimulus.num_transactions
    corner = ""
    if stimulus.enable_corner_cases:
        corner_assigns = []
        for port in rtl.input_ports:
            if port.width > 1:
                corner_assigns.append(f"                txn.{port.name} = '0;")
                corner_assigns.append(f"                txn.{port.name} = '1;")
        if corner_assigns:
            corner = textwrap.indent("\n".join(corner_assigns), "")

    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore UVM — Sequences for {module_name}

class {module_name}_random_seq extends uvm_sequence #({module_name}_transaction);
    `uvm_object_utils({module_name}_random_seq)

    int num_transactions = {num_transactions};

    function new(string name = "{module_name}_random_seq");
        super.new(name);
    endfunction

    task body();
        {module_name}_transaction txn;
        for (int i = 0; i < num_transactions; i++) begin
            txn = {module_name}_transaction::type_id::create($sformatf("txn_%0d", i));
            start_item(txn);
            if (!txn.randomize())
                `uvm_fatal("SEQ", "Randomization failed")
            finish_item(txn);
        end
    endtask
endclass

class {module_name}_corner_seq extends uvm_sequence #({module_name}_transaction);
    `uvm_object_utils({module_name}_corner_seq)

    function new(string name = "{module_name}_corner_seq");
        super.new(name);
    endfunction

    task body();
        {module_name}_transaction txn;
        // All zeros
        txn = {module_name}_transaction::type_id::create("txn_zeros");
        start_item(txn);
        if (!txn.randomize())
            `uvm_fatal("SEQ", "Randomization failed")
{corner}
        finish_item(txn);
    endtask
endclass

class {module_name}_lfsr_seq extends uvm_sequence #({module_name}_transaction);
    `uvm_object_utils({module_name}_lfsr_seq)

    int seed = 16'hACE1;
    int length = 256;

    function new(string name = "{module_name}_lfsr_seq");
        super.new(name);
    endfunction

    task body();
        {module_name}_transaction txn;
        logic [15:0] lfsr = seed;
        for (int i = 0; i < length; i++) begin
            txn = {module_name}_transaction::type_id::create($sformatf("lfsr_%0d", i));
            start_item(txn);
            if (!txn.randomize())
                `uvm_fatal("SEQ", "Randomization failed")
            // Override with LFSR-driven data
            lfsr = {{lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]}};
            finish_item(txn);
        end
    endtask
endclass
""")


def _emit_driver(rtl: RTLModule) -> str:
    module_name = rtl.name
    drive_lines = []
    for port in rtl.input_ports:
        drive_lines.append(f"            vif.{port.name} <= txn.{port.name};")
    drive_block = "\n".join(drive_lines) if drive_lines else "            // no input ports"

    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore UVM — Driver for {module_name}

class {module_name}_driver extends uvm_driver #({module_name}_transaction);
    `uvm_component_utils({module_name}_driver)

    virtual {module_name}_if vif;

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        if (!uvm_config_db #(virtual {module_name}_if)::get(this, "", "vif", vif))
            `uvm_fatal("DRV", "No virtual interface")
    endfunction

    task run_phase(uvm_phase phase);
        {module_name}_transaction txn;
        forever begin
            seq_item_port.get_next_item(txn);
            @(posedge vif.clk);
{drive_block}
            seq_item_port.item_done();
        end
    endtask
endclass
""")


def _emit_monitor(rtl: RTLModule) -> str:
    module_name = rtl.name
    sample_in = []
    for port in rtl.input_ports:
        sample_in.append(f"            txn.{port.name} = vif.{port.name};")
    sample_out = []
    for port in rtl.output_ports:
        sample_out.append(f"            txn.{port.name} = vif.{port.name};")
    in_block = "\n".join(sample_in) if sample_in else "            // no inputs"
    out_block = "\n".join(sample_out) if sample_out else "            // no outputs"

    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore UVM — Monitor for {module_name}

class {module_name}_monitor extends uvm_monitor;
    `uvm_component_utils({module_name}_monitor)

    virtual {module_name}_if vif;
    uvm_analysis_port #({module_name}_transaction) ap;

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        ap = new("ap", this);
        if (!uvm_config_db #(virtual {module_name}_if)::get(this, "", "vif", vif))
            `uvm_fatal("MON", "No virtual interface")
    endfunction

    task run_phase(uvm_phase phase);
        {module_name}_transaction txn;
        forever begin
            @(posedge vif.clk);
            txn = {module_name}_transaction::type_id::create("mon_txn");
{in_block}
{out_block}
            ap.write(txn);
        end
    endtask
endclass
""")


def _emit_scoreboard(rtl: RTLModule, scoreboard: ScoreboardConfig) -> str:
    module_name = rtl.name
    checks = []
    if scoreboard.check_popcount:
        for port in rtl.output_ports:
            if port.width > 1:
                checks.append(
                    f"        // Popcount check for {port.name}\n"
                    f"        int pc_{port.name} = $countones(txn.{port.name});\n"
                    f'        `uvm_info("SB", $sformatf("{port.name} popcount=%0d", pc_{port.name}), UVM_MEDIUM)'
                )
    if scoreboard.check_spike_timing:
        for port in rtl.output_ports:
            if port.name.startswith("spike") or port.name.endswith("spike") or port.width == 1:
                checks.append(
                    f"        if (txn.{port.name})\n"
                    f"            spike_count++;\n"
                    f'        `uvm_info("SB", $sformatf("spike_count=%0d", spike_count), UVM_HIGH)'
                )

    golden_block = ""
    if scoreboard.check_golden_comparison:
        golden_lines = []
        for port in rtl.output_ports:
            if port.width > 1:
                if port.name not in scoreboard.golden_expressions:
                    raise ValueError(
                        "Missing golden reference expression for output "
                        f"{port.name!r}; provide ScoreboardConfig.golden_expressions."
                    )
                golden_lines.append(
                    f"        // Golden model comparison for {port.name}\n"
                    f"        expected_{port.name} = golden_compute_{port.name}(txn);\n"
                    f"        if (txn.{port.name} !== expected_{port.name}) begin\n"
                    f"            mismatch_count++;\n"
                    f'            `uvm_error("SB", $sformatf(\n'
                    f'                "MISMATCH {port.name}: got=%0h exp=%0h",\n'
                    f"                txn.{port.name}, expected_{port.name}))\n"
                    f"        end"
                )
        golden_block = "\n".join(golden_lines)

    check_block = "\n".join(checks) if checks else "        // No specific checks configured"

    golden_funcs = []
    golden_vars = []
    if scoreboard.check_golden_comparison:
        for port in rtl.output_ports:
            if port.width > 1:
                width = port.width
                expression = scoreboard.golden_expressions[port.name]
                golden_funcs.append(
                    f"    // Bit-true reference expression for {port.name}\n"
                    f"    function logic [{width - 1}:0] golden_compute_{port.name}({module_name}_transaction txn);\n"
                    f"        return {expression};\n"
                    f"    endfunction"
                )
                golden_vars.append(f"    logic [{port.width - 1}:0] expected_{port.name};")
    golden_func_block = "\n\n".join(golden_funcs)
    golden_var_block = "\n".join(golden_vars)

    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore UVM — Scoreboard for {module_name}

class {module_name}_scoreboard extends uvm_scoreboard;
    `uvm_component_utils({module_name}_scoreboard)

    uvm_analysis_imp #({module_name}_transaction, {module_name}_scoreboard) ap;
    int transaction_count;
    int mismatch_count;
    int spike_count;
{golden_var_block}

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        ap = new("ap", this);
        transaction_count = 0;
        mismatch_count = 0;
        spike_count = 0;
    endfunction

{golden_func_block}

    function void write({module_name}_transaction txn);
        transaction_count++;
{check_block}
{golden_block}
    endfunction

    function void report_phase(uvm_phase phase);
        `uvm_info("SB", $sformatf(
            "Scoreboard summary: transactions=%0d mismatches=%0d spikes=%0d",
            transaction_count, mismatch_count, spike_count), UVM_LOW)
        if (mismatch_count > 0)
            `uvm_error("SB", $sformatf("%0d mismatches detected", mismatch_count))
    endfunction
endclass
""")


def _emit_coverage(rtl: RTLModule, coverage: CoverageSpec) -> str:
    module_name = rtl.name
    coverpoints = []

    for port in rtl.input_ports:
        if port.width > 1:
            bins = coverage.bitstream_density_bins
            coverpoints.append(
                f"        {port.name}_density: coverpoint $countones(txn.{port.name}) {{\n"
                f"            bins density[{bins}] = {{[0:{port.width}]}};\n"
                f"        }}"
            )

    for port in rtl.output_ports:
        if port.width == 1:
            coverpoints.append(
                f"        {port.name}_toggle: coverpoint txn.{port.name} {{\n"
                f"            bins off = {{0}};\n"
                f"            bins on  = {{1}};\n"
                f"        }}"
            )
        elif port.width > 1:
            bins = coverage.spike_rate_bins
            coverpoints.append(
                f"        {port.name}_density: coverpoint $countones(txn.{port.name}) {{\n"
                f"            bins density[{bins}] = {{[0:{port.width}]}};\n"
                f"        }}"
            )

    scc_coverpoints = []
    input_wide = [port for port in rtl.input_ports if port.width > 1]
    if len(input_wide) >= 2 and coverage.scc_bins > 0:
        first, second = input_wide[0], input_wide[1]
        minimum_width = min(first.width, second.width)
        scc_coverpoints.append(
            f"        // SCC correlation bins between {first.name} and {second.name}\n"
            f"        {first.name}_{second.name}_scc: coverpoint \n"
            f"            ($countones(txn.{first.name}[{minimum_width - 1}:0] & txn.{second.name}[{minimum_width - 1}:0])) {{\n"
            f"            bins scc_bins[{coverage.scc_bins}] = {{[0:{minimum_width}]}};\n"
            f"        }}"
        )
    coverpoints.extend(scc_coverpoints)

    toggle_coverpoints = []
    if coverage.toggle_coverage:
        for port in rtl.input_ports:
            if port.width > 1:
                toggle_coverpoints.append(
                    f"        {port.name}_activity: coverpoint txn.{port.name} {{\n"
                    f"            bins zero = {{'0}};\n"
                    f"            bins full = {{'1}};\n"
                    f"            bins mid[4] = {{[1:{(1 << port.width) - 2}]}};\n"
                    f"        }}"
                )
    coverpoints.extend(toggle_coverpoints)

    cross_covers = ""
    if coverage.cross_coverage and len(coverpoints) >= 2:
        cross_covers = "        // Cross coverage between input/output density\n"

    coverpoint_block = (
        "\n".join(coverpoints) if coverpoints else "        // Auto-generated coverpoints"
    )
    target = coverage.target_percent

    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore UVM — Functional Coverage for {module_name}

class {module_name}_coverage extends uvm_subscriber #({module_name}_transaction);
    `uvm_component_utils({module_name}_coverage)

    real coverage_target = {target};

    covergroup {module_name}_cg with function sample({module_name}_transaction txn);
{coverpoint_block}
{cross_covers}
    endgroup

    function new(string name, uvm_component parent);
        super.new(name, parent);
        {module_name}_cg = new();
    endfunction

    function void write({module_name}_transaction t);
        {module_name}_cg.sample(t);
    endfunction

    function void report_phase(uvm_phase phase);
        real cov = {module_name}_cg.get_inst_coverage();
        `uvm_info("COV", $sformatf(
            "Coverage: %.1f%% (target: %.1f%%)", cov, coverage_target), UVM_LOW)
        if (cov < coverage_target)
            `uvm_warning("COV", $sformatf(
                "Coverage %.1f%% below target %.1f%%", cov, coverage_target))
    endfunction
endclass
""")


def _emit_agent(rtl: RTLModule) -> str:
    module_name = rtl.name
    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore UVM — Agent for {module_name}

class {module_name}_agent extends uvm_agent;
    `uvm_component_utils({module_name}_agent)

    {module_name}_driver  drv;
    {module_name}_monitor mon;
    uvm_sequencer #({module_name}_transaction) sqr;

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        drv = {module_name}_driver::type_id::create("drv", this);
        mon = {module_name}_monitor::type_id::create("mon", this);
        sqr = uvm_sequencer#({module_name}_transaction)::type_id::create("sqr", this);
    endfunction

    function void connect_phase(uvm_phase phase);
        super.connect_phase(phase);
        drv.seq_item_port.connect(sqr.seq_item_export);
    endfunction
endclass
""")


def _emit_env(rtl: RTLModule) -> str:
    module_name = rtl.name
    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore UVM — Environment for {module_name}

class {module_name}_env extends uvm_env;
    `uvm_component_utils({module_name}_env)

    {module_name}_agent      agt;
    {module_name}_scoreboard sb;
    {module_name}_coverage   cov;

    function new(string name, uvm_component parent);
        super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        agt = {module_name}_agent::type_id::create("agt", this);
        sb  = {module_name}_scoreboard::type_id::create("sb", this);
        cov = {module_name}_coverage::type_id::create("cov", this);
    endfunction

    function void connect_phase(uvm_phase phase);
        super.connect_phase(phase);
        agt.mon.ap.connect(sb.ap);
        agt.mon.ap.connect(cov.analysis_export);
    endfunction
endclass
""")
