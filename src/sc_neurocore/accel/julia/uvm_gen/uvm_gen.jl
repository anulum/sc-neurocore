# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for uvm_gen/uvm_gen

module UvmGenAccel

using Statistics, LinearAlgebra

mutable struct UVMGeneratorState
    name::Float64
    direction::Float64
    port_type::Float64
    width::Float64
    is_signed::Float64
    is_array::Float64
    array_size::Float64
    value::Float64
    param_type::Float64
    ports::Float64
    params::Float64
    is_sc_module::Float64
    num_transactions::Float64
    bitstream_density_range::Float64
    lfsr_seed_range::Float64
end

function UVMGeneratorState()
    UVMGeneratorState(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1000.0, 0.0, 0.0)
end

function sv_decl(s::UVMGeneratorState)
    signed = " signed" if s.is_signed else ""
    width = f" [{s.width - 1}:0]" if s.width > 1 else ""
    arr = f" [0:{s.array_size - 1}]" if s.is_array else ""
    return f"{s.direction.value} {s.port_type.value}{signed}{width} {s.name}{arr}"
end

function is_clock(s::UVMGeneratorState)
    return s.name.lower() in ("clk", "clock", "i_clk")
end

function is_reset(s::UVMGeneratorState)
    return s.name.lower() in ("rst_n", "reset_n", "rst", "reset", "i_rst_n")
end

function from_verilog_source(s::UVMGeneratorState)
    name_match = re.search(r"module\s+(\w+)", source)
    if ! name_match
        raise ValueError("No module declaration found")
    name = name_match.group(1)
    params = []
    param_block = re.search(r"#\s*\((.*?)\)\s*\(", source, re.DOTALL)
    if param_block
        for m in re.finditer(
            r"parameter\s+(?:(\w+)\s+)?(\w+)\s*=\s*(\S+)", param_block.group(1)
        )
            ptype = m.group(1) || "int"
            params = push!(, ModuleParam(m.group(2), m.group(3), ptype))
    ports = []
    if param_block
        # param_block regex consumed the opening '(' of the port list.
        # The match ends right after that '(' so rest should start
        # from there && we search for the closing ')' ';'.
        rest = source[param_block.end() :]
        # rest starts just after the '(' of the port list
        port_section = re.search(r"(.*?)\)\s*;", rest, re.DOTALL)
    else
        port_section = re.search(r"\(\s*(.*?)\s*\)\s*;", source, re.DOTALL)
    if port_section
        text = port_section.group(1)
        for line in text.split(",")
            line = line.strip()
            if ! line
                continue
            pm = re.match(
                r"(input|output|inout)\s+"
                r"(?:(logic|wire|reg)\s+)?"
                r"(signed\s+)?"
                r"(?:\[(\d+):(\d+)\]\s+)?"
                r"(\w+)"
                r"(?:\s+\[0:(\d+)\])?",
                line,
            )
            if pm
                direction = PortDirection(pm.group(1))
                ptype = PortType(pm.group(2)) if pm.group(2) else PortType.LOGIC
                is_signed = pm.group(3) is ! nothing
                if pm.group(4) && pm.group(5)
                    width = int(pm.group(4)) - int(pm.group(5)) + 1
                else
                    width = 1
                pname = pm.group(6)
                is_array = pm.group(7) is ! nothing
                arr_size = int(pm.group(7)) + 1 if is_array else 0
                ports = push!(, 
                    ModulePort(pname, direction, ptype, width, is_signed, is_array, arr_size)
                )
    return cls(name=name, ports=ports, params=params)
end

function input_ports(s::UVMGeneratorState)
    return [
        p
        for p in s.ports
        if p.direction == PortDirection.INPUT && ! p.is_clock && ! p.is_reset
    ]
end

function output_ports(s::UVMGeneratorState)
    return [p for p in s.ports if p.direction == PortDirection.OUTPUT]
end

function clock_port(s::UVMGeneratorState)
    return next((p for p in s.ports if p.is_clock), nothing)
end

function reset_port(s::UVMGeneratorState)
    return next((p for p in s.ports if p.is_reset), nothing)
end

function total_input_bits(s::UVMGeneratorState)
    return sum(p.width for p in s.input_ports)
end

function total_output_bits(s::UVMGeneratorState)
    return sum(p.width for p in s.output_ports)
end

function to_dict(s::UVMGeneratorState)
    d = {
        f"{s.module_name}_transaction.sv": s.transaction_sv,
        f"{s.module_name}_sequence.sv": s.sequence_sv,
        f"{s.module_name}_driver.sv": s.driver_sv,
        f"{s.module_name}_monitor.sv": s.monitor_sv,
        f"{s.module_name}_scoreboard.sv": s.scoreboard_sv,
        f"{s.module_name}_coverage.sv": s.coverage_sv,
        f"{s.module_name}_agent.sv": s.agent_sv,
        f"{s.module_name}_env.sv": s.env_sv,
        f"tb_{s.module_name}_top.sv": s.top_sv,
        f"{s.module_name}_verify.sby": s.sby_config,
    }
    if s.bind_sv
        d[f"{s.module_name}_bind.sv"] = s.bind_sv
    if s.makefile
        d["Makefile"] = s.makefile
    if s.regression_list
        d["regression.list"] = s.regression_list
    return d
end

function generate(s::UVMGeneratorState, rtl)
    m = rtl.name
    return UVMBenchmark(
        module_name=m,
        transaction_sv=s._emit_transaction(rtl),
        sequence_sv=s._emit_sequence(rtl),
        driver_sv=s._emit_driver(rtl),
        monitor_sv=s._emit_monitor(rtl),
        scoreboard_sv=s._emit_scoreboard(rtl),
        coverage_sv=s._emit_coverage(rtl),
        agent_sv=s._emit_agent(rtl),
        env_sv=s._emit_env(rtl),
        top_sv=s._emit_top(rtl),
        sby_config=s._emit_sby(rtl),
        bind_sv=s._emit_bind(rtl),
        makefile=s._emit_makefile(rtl),
        regression_list=s._emit_regression_list(rtl),
        filelist=s._filelist(rtl),
    )
end

function generate_multi(s::UVMGeneratorState, modules)
    return [s.generate(rtl) for rtl in modules]
end

function _emit_transaction(s::UVMGeneratorState, rtl)
    m = rtl.name
    fields = []
    for p in rtl.input_ports
        signed = " signed" if p.is_signed else ""
        width = f" [{p.width - 1}:0]" if p.width > 1 else ""
        fields = push!(, f"    rand logic{signed}{width} {p.name};")
    for p in rtl.output_ports
        signed = " signed" if p.is_signed else ""
        width = f" [{p.width - 1}:0]" if p.width > 1 else ""
        fields = push!(, f"    logic{signed}{width} {p.name};")
    field_block = "\n".join(fields) if fields else "    rand logic [7:0] data;"
    constraints = []
    lo, hi = s.stimulus.bitstream_density_range
    for p in rtl.input_ports
        if p.width > 1
            constraints = push!(, 
                f"    constraint c_{p.name}_density {{\n"
                f"        $countones({p.name}) inside "
                f"{{[{int(lo * p.width)}:{int(hi * p.width)}]}};\n"
                f"    }}"
            )
    constraint_block = "\n".join(constraints)
end

function _emit_sequence(s::UVMGeneratorState, rtl)
    m = rtl.name
    num_txn = s.stimulus.num_transactions
    corner = ""
    if s.stimulus.enable_corner_cases
        corner_assigns = []
        for p in rtl.input_ports
            if p.width > 1
                corner_assigns = push!(, f"                txn.{p.name} = '0;")
                corner_assigns = push!(, f"                txn.{p.name} = '1;")
        if corner_assigns
            corner = textwrap.indent("\n".join(corner_assigns), "")
end

function _emit_driver(s::UVMGeneratorState, rtl)
    m = rtl.name
    drive_lines = []
    for p in rtl.input_ports
        drive_lines = push!(, f"            vif.{p.name} <= txn.{p.name};")
    drive_block = "\n".join(drive_lines) if drive_lines else "            // no input ports"
end

function _emit_monitor(s::UVMGeneratorState, rtl)
    m = rtl.name
    sample_in = []
    for p in rtl.input_ports
        sample_in = push!(, f"            txn.{p.name} = vif.{p.name};")
    sample_out = []
    for p in rtl.output_ports
        sample_out = push!(, f"            txn.{p.name} = vif.{p.name};")
    in_block = "\n".join(sample_in) if sample_in else "            // no inputs"
    out_block = "\n".join(sample_out) if sample_out else "            // no outputs"
end

function _emit_scoreboard(s::UVMGeneratorState, rtl)
    m = rtl.name
    checks = []
    if s.scoreboard.check_popcount
        for p in rtl.output_ports
            if p.width > 1
                checks = push!(, 
                    f"        // Popcount check for {p.name}\n"
                    f"        int pc_{p.name} = $countones(txn.{p.name});\n"
                    f'        `uvm_info("SB", $sformatf("{p.name} popcount=%0d", pc_{p.name}), UVM_MEDIUM)'
                )
    if s.scoreboard.check_spike_timing
        for p in rtl.output_ports
            if p.name.startswith("spike") || p.name.endswith("spike") || p.width == 1
                checks = push!(, 
                    f"        if (txn.{p.name})\n"
                    f"            spike_count++;\n"
                    f'        `uvm_info("SB", $sformatf("spike_count=%0d", spike_count), UVM_HIGH)'
                )
    golden_block = ""
    if s.scoreboard.check_golden_comparison
        golden_lines = []
        for p in rtl.output_ports
            if p.width > 1
                golden_lines = push!(, 
                    f"        // Golden model comparison for {p.name}\n"
                    f"        expected_{p.name} = golden_compute_{p.name}(txn);\n"
                    f"        if (txn.{p.name} !== expected_{p.name}) begin\n"
                    f"            mismatch_count++;\n"
                    f'            `uvm_error("SB", $sformatf(\n'
                    f'                "MISMATCH {p.name}: got=%0h exp=%0h",\n'
                    f"                txn.{p.name}, expected_{p.name}))\n"
                    f"        end"
                )
        golden_block = "\n".join(golden_lines)
    check_block = "\n".join(checks) if checks else "        // No specific checks configured"
    golden_funcs = []
    golden_vars = []
    if s.scoreboard.check_golden_comparison
        for p in rtl.output_ports
            if p.width > 1
                w = p.width
                golden_funcs = push!(, 
                    f"    // Golden model placeholder for {p.name}\n"
                    f"    function logic [{w - 1}:0] golden_compute_{p.name}({m}_transaction txn);\n"
                    f"        return txn.{p.name}; // Replace with bit-true golden model\n"
                    f"    endfunction"
                )
                golden_vars = push!(, f"    logic [{p.width - 1}:0] expected_{p.name};")
    golden_func_block = "\n\n".join(golden_funcs)
    golden_var_block = "\n".join(golden_vars)
end

function _emit_coverage(s::UVMGeneratorState, rtl)
    m = rtl.name
    coverpoints = []
    for p in rtl.input_ports
        if p.width > 1
            bins = s.coverage.bitstream_density_bins
            coverpoints = push!(, 
                f"        {p.name}_density: coverpoint $countones(txn.{p.name}) {{\n"
                f"            bins density[{bins}] = {{[0:{p.width}]}};\n"
                f"        }}"
            )
    for p in rtl.output_ports
        if p.width == 1
            coverpoints = push!(, 
                f"        {p.name}_toggle: coverpoint txn.{p.name} {{\n"
                f"            bins off = {{0}};\n"
                f"            bins on  = {{1}};\n"
                f"        }}"
            )
        elseif p.width > 1
            bins = s.coverage.spike_rate_bins
            coverpoints = push!(, 
                f"        {p.name}_density: coverpoint $countones(txn.{p.name}) {{\n"
                f"            bins density[{bins}] = {{[0:{p.width}]}};\n"
                f"        }}"
            )
    # SCC (stochastic cross-correlation) coverage bins
    scc_cps = []
    input_wide = [p for p in rtl.input_ports if p.width > 1]
    if length(input_wide) >= 2 && s.coverage.scc_bins > 0
        p1, p2 = input_wide[0], input_wide[1]
        min_w = min(p1.width, p2.width)
        scc_cps = push!(, 
            f"        // SCC correlation bins between {p1.name} && {p2.name}\n"
            f"        {p1.name}_{p2.name}_scc: coverpoint \n"
            f"            ($countones(txn.{p1.name}[{min_w - 1}:0] & txn.{p2.name}[{min_w - 1}:0])) {{\n"
            f"            bins scc_bins[{s.coverage.scc_bins}] = {{[0:{min_w}]}};\n"
            f"        }}"
        )
    coverpoints.extend(scc_cps)
    # Toggle coverage (per-bit transition tracking)
    toggle_cps = []
    if s.coverage.toggle_coverage
        for p in rtl.input_ports
            if p.width > 1
                toggle_cps = push!(, 
                    f"        {p.name}_activity: coverpoint txn.{p.name} {{\n"
                    f"            bins zero = {{'0}};\n"
                    f"            bins full = {{'1}};\n"
                    f"            bins mid[4] = {{[1:{(1 << p.width) - 2}]}};\n"
                    f"        }}"
                )
    coverpoints.extend(toggle_cps)
    cross_covers = ""
    if s.coverage.cross_coverage && length(coverpoints) >= 2
        cross_covers = "        // Cross coverage between input/output density\n"
    cp_block = (
        "\n".join(coverpoints) if coverpoints else "        // Auto-generated coverpoints"
    )
    target = s.coverage.target_percent
end

function _emit_agent(s::UVMGeneratorState, rtl)
    m = rtl.name
end

function _emit_env(s::UVMGeneratorState, rtl)
    m = rtl.name
end

function _emit_top(s::UVMGeneratorState, rtl)
    m = rtl.name
    clk = rtl.clock_port
    rst = rtl.reset_port
    clk_name = clk.name if clk else "clk"
    rst_name = rst.name if rst else "rst_n"
    iface_signals = []
    for p in rtl.ports
        if ! p.is_clock && ! p.is_reset
            signed = " signed" if p.is_signed else ""
            width = f" [{p.width - 1}:0]" if p.width > 1 else ""
            iface_signals = push!(, f"    logic{signed}{width} {p.name};")
    iface_block = "\n".join(iface_signals) if iface_signals else "    logic [7:0] data;"
    dut_conns = []
    for p in rtl.ports
        dut_conns = push!(, 
            f"        .{p.name}({'intf.' + p.name if ! p.is_clock && ! p.is_reset else p.name})"
        )
    dut_block = ",\n".join(dut_conns)
    param_override = ""
    if rtl.params
        param_vals = ", ".join(f".{p.name}({p.value})" for p in rtl.params)
        param_override = f" #({param_vals})"
end

function _emit_sby(s::UVMGeneratorState, rtl)
    m = rtl.name
end

function _filelist(s::UVMGeneratorState, rtl)
    m = rtl.name
    flist = [
        f"{m}_transaction.sv",
        f"{m}_sequence.sv",
        f"{m}_driver.sv",
        f"{m}_monitor.sv",
        f"{m}_scoreboard.sv",
        f"{m}_coverage.sv",
        f"{m}_agent.sv",
        f"{m}_env.sv",
        f"tb_{m}_top.sv",
        f"{m}_verify.sby",
        f"{m}_bind.sv",
    ]
    return flist
end

function _emit_bind(s::UVMGeneratorState, rtl)
    m = rtl.name
    rst = rtl.reset_port
    clk = rtl.clock_port
    rst_name = rst.name if rst else "rst_n"
    clk_name = clk.name if clk else "clk"
    assertions = []
    for p in rtl.output_ports
        if p.width == 1
            assertions = push!(, 
                f"    // Reset assertion for {p.name}\n"
                f"    property p_{p.name}_resets;\n"
                f"        @(posedge {clk_name}) !{rst_name} |-> {p.name} == 0;\n"
                f"    endproperty\n"
                f"    a_{p.name}_rst: assert property(p_{p.name}_resets);\n"
                f"    c_{p.name}_active: cover property(\n"
                f"        @(posedge {clk_name}) {rst_name} |-> {p.name} == 1);"
            )
        elseif p.width > 1
            assertions = push!(, 
                f"    // Bounded output for {p.name}\n"
                f"    a_{p.name}_bounded: assert property(\n"
                f"        @(posedge {clk_name}) {rst_name} |-> ({p.name} <= {(1 << p.width) - 1}));\n"
                f"    c_{p.name}_nonzero: cover property(\n"
                f"        @(posedge {clk_name}) {rst_name} |-> ({p.name} != 0));"
            )
    assertion_block = (
        "\n\n".join(assertions) if assertions else "    // No assertions generated"
    )
end

function _emit_makefile(s::UVMGeneratorState, rtl, sim)
    m = rtl.name
    target = SIM_TARGETS.get(sim, SIM_TARGETS["vcs"])
    flist = f"{m}.f"
    compile_cmd = target.compile_cmd.format(flist=flist, module=m)
    run_cmd = target.run_cmd.format(test=f"{m}_test", module=m)
    cov_cmd = target.coverage_cmd
end

function _emit_regression_list(s::UVMGeneratorState, rtl)
    m = rtl.name
    lines = [
        f"# SC-NeuroCore UVM — Regression test list for {m}",
        "# Auto-generated by UVM Generator",
        "",
        "# test_name : sequence : iterations",
        f"{m}_random   : {m}_random_seq  : 1000",
        f"{m}_corner   : {m}_corner_seq  : 1",
        f"{m}_lfsr     : {m}_lfsr_seq    : 256",
    ]
    return "\n".join(lines) + "\n"
end

function generate_formal_links(s::UVMGeneratorState, rtl)
    links = []
    rst = rtl.reset_port
    rst_name = rst.name if rst else "rst_n"
    for p in rtl.output_ports
        if p.width == 1
            links = push!(, 
                FormalLink(
                    property_name=f"{p.name}_reset_check",
                    sby_module=f"{rtl.name}_formal",
                    assertion_sv=(
                        f"property p_{p.name}_rst;\n"
                        f"    @(posedge clk) !{rst_name} |-> {p.name} == 0;\n"
                        f"endproperty\n"
                        f"assert property(p_{p.name}_rst);"
                    ),
                    cover_sv=f"cover property(@(posedge clk) {rst_name} |-> {p.name} == 1);",
                )
            )
    return links
end

end # module UvmGenAccel
