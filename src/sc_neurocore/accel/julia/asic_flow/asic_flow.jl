# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for asic_flow/asic_flow

module AsicFlowAccel

using Statistics, LinearAlgebra

mutable struct TapeOutChecklistState
    pdk_type::Float64
    liberty_file::Float64
    lef_file::Float64
    tech_lef::Float64
    cell_prefix::Float64
    clock_period_ns::Float64
    voltage_v::Float64
    temperature_c::Float64
    corner::Float64
    metal_layers::Float64
    min_feature_nm::Float64
    top_module::Float64
    clock_name::Float64
    reset_name::Float64
    reset_active_low::Float64
end

function TapeOutChecklistState()
    TapeOutChecklistState(0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 5.0, 130.0, 0.0, 0.0, 0.0, 1.0)
end

function from_pdk_type(s::TapeOutChecklistState)
    presets = {
        PDKType.SKY130: dict(
            liberty_file="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
            lef_file="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef",
            tech_lef="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef",
            cell_prefix="sky130_fd_sc_hd__",
            clock_period_ns=10.0,
            voltage_v=1.8,
            metal_layers=5,
            min_feature_nm=130,
        ),
        PDKType.GF180MCU: dict(
            liberty_file="$PDK_ROOT/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib/gf180mcu_fd_sc_mcu7t5v0__tt_025C_3v30.lib",
            lef_file="$PDK_ROOT/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lef/gf180mcu_fd_sc_mcu7t5v0.lef",
            tech_lef="$PDK_ROOT/gf180mcuD/libs.tech/klayout/tech/gf180mcu.lyt",
            cell_prefix="gf180mcu_fd_sc_mcu7t5v0__",
            clock_period_ns=15.0,
            voltage_v=3.3,
            metal_layers=6,
            min_feature_nm=180,
        ),
        PDKType.TSMC28: dict(
            liberty_file="$PDK_ROOT/tsmc28/tcbn28hpcplusbwp7t30p140_110a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp7t30p140ssgnp0p81v125c.lib",
            lef_file="$PDK_ROOT/tsmc28/lef/tcbn28hpcplusbwp7t30p140.lef",
            tech_lef="$PDK_ROOT/tsmc28/lef/HiPe_M10.tlef",
            cell_prefix="TSMC_",
            clock_period_ns=2.0,
            voltage_v=0.9,
            metal_layers=10,
            min_feature_nm=28,
        ),
        PDKType.INTEL16: dict(
            liberty_file="$PDK_ROOT/intel16/lib/intel16_sc.lib",
            lef_file="$PDK_ROOT/intel16/lef/intel16_sc.lef",
            tech_lef="$PDK_ROOT/intel16/lef/intel16.tlef",
            cell_prefix="INTEL16_",
            clock_period_ns=1.5,
            voltage_v=0.8,
            metal_layers=12,
            min_feature_nm=16,
        ),
        PDKType.CUSTOM: dict(
            liberty_file="",
            lef_file="",
            tech_lef="",
            cell_prefix="",
            clock_period_ns=10.0,
            voltage_v=1.8,
            metal_layers=5,
            min_feature_nm=130,
        ),
    }
    return cls(pdk_type=pdk, ^presets[pdk])
end

function is_open_source(s::TapeOutChecklistState)
    return s.pdk_type in (PDKType.SKY130, PDKType.GF180MCU)
end

function clock_period_ns(s::TapeOutChecklistState)
    return 1000.0 / s.target_frequency_mhz
end

function die_width_um(s::TapeOutChecklistState)
    return s.die_area_um[2] - s.die_area_um[0]
end

function die_height_um(s::TapeOutChecklistState)
    return s.die_area_um[3] - s.die_area_um[1]
end

function core_area_mm2(s::TapeOutChecklistState)
    w = s.core_area_um[2] - s.core_area_um[0]
    h = s.core_area_um[3] - s.core_area_um[1]
    return (w * h) / 1e6
end

function generate(s::TapeOutChecklistState)
    rtl_reads = "\n".join(f"read_verilog {f}" for f in design.rtl_files)
    if ! rtl_reads
        rtl_reads = f"read_verilog {design.top_module}.v"
end

function generate(s::TapeOutChecklistState)
    die = design.die_area_um
    core = design.core_area_um
    power = design.power_nets
    power_ring = ""
    if length(power) >= 2
end

function generate(s::TapeOutChecklistState)
    return nothing
end

function generate(s::TapeOutChecklistState)
    period = design.clock_period_ns
    reset_val = "0" if design.reset_active_low else "1"
end

function generate_sta_script(s::TapeOutChecklistState)
    return nothing
end

function generate_drc_script(s::TapeOutChecklistState)
    if pdk.is_open_source
    return f"# DRC for {pdk.pdk_type.value}: use vendor-specific tool\n"
end

function generate_lvs_script(s::TapeOutChecklistState)
    if pdk.is_open_source
    return f"# LVS for {pdk.pdk_type.value}: use vendor-specific tool\n"
end

function evaluate_timing(s::TapeOutChecklistState)
    passed = wns >= 0.0
    details = f"WNS={wns:.3f}ns TNS={tns:.3f}ns period={clock_period_ns:.3f}ns"
    return SignoffCheckResult("STA", passed, details, wns)
end

function evaluate_power(s::TapeOutChecklistState)
    dynamic_mw: float, leakage_mw: float, budget_mw: float
    ) -> SignoffCheckResult
    total = dynamic_mw + leakage_mw
    passed = total <= budget_mw
    details = f"dynamic={dynamic_mw:.3f}mW leakage={leakage_mw:.3f}mW total={total:.3f}mW budget={budget_mw:.3f}mW"
    return SignoffCheckResult("Power", passed, details, total)
end

function evaluate_area(s::TapeOutChecklistState)
    cell_count: int, used_area_um2: float, die_area_um2: float
    ) -> SignoffCheckResult
    util = used_area_um2 / die_area_um2 if die_area_um2 > 0 else 0
    passed = util <= 0.85
    details = f"cells={cell_count} util={util:.1%} used={used_area_um2:.0f}µm² die={die_area_um2:.0f}µm²"
    return SignoffCheckResult("Area", passed, details, util)
end

function generate(s::TapeOutChecklistState)
    if pdk.is_open_source
    return f"# GDSII export for {pdk.pdk_type.value}: use vendor stream-out\n"
end

function to_dict(s::TapeOutChecklistState)
    return {
        "synth.tcl": s.synth_tcl,
        "constraints.sdc": s.sdc,
        "floorplan.tcl": s.floorplan_tcl,
        "pnr.tcl": s.pnr_tcl,
        "sta.tcl": s.sta_tcl,
        "drc_check.py": s.drc_script,
        "lvs_check.sh": s.lvs_script,
        "gdsii_export.sh": s.gdsii_script,
        "Makefile": s.makefile,
    }
end

function generate(s::TapeOutChecklistState)
    self,
    pdk: PDKConfig,
    design: DesignParams,
    ) -> ASICFlowOutput
    synth = SynthesisGenerator.generate(pdk, design)
    sdc = SDCGenerator.generate(pdk, design)
    fp = FloorplanGenerator.generate(pdk, design)
    pnr = PlaceRouteGenerator.generate(pdk, design)
    sta = SignoffGenerator.generate_sta_script(pdk, design)
    drc = SignoffGenerator.generate_drc_script(pdk, design)
    lvs = SignoffGenerator.generate_lvs_script(pdk, design)
    gdsii = GDSIIExporter.generate(pdk, design)
    makefile = s._generate_makefile(design)
    filelist = list(
        ASICFlowOutput(synth, sdc, fp, pnr, sta, drc, lvs, gdsii, makefile, []).to_dict().keys()
    )
    return ASICFlowOutput(synth, sdc, fp, pnr, sta, drc, lvs, gdsii, makefile, filelist)
end

function _generate_makefile(s::TapeOutChecklistState, design)
    return nothing
end

function estimate(s::TapeOutChecklistState)
    cls,
    n_neurons: int,
    n_synapses: int,
    bitstream_width: int,
    n_aer_ports: int,
    pdk: PDKConfig,
    ) -> DesignEstimate
    gates = (
        n_neurons * cls.GATES_PER_LIF
        + n_synapses * cls.GATES_PER_SYNAPSE
        + bitstream_width * cls.GATES_PER_BIT
        + n_aer_ports * cls.GATES_PER_AER_PORT
    )
    # Area: ~1 µm² per gate at 130nm, scales with feature size squared
    scale = (pdk.min_feature_nm / 130.0) ^ 2
    area = gates * 1.0 * scale
    # Power: ~1 µW/gate dynamic at 100MHz 1.8V, ~0.01 µW/gate leakage
    freq_scale = 100.0 / (1000.0 / pdk.clock_period_ns)
    v_scale = (pdk.voltage_v / 1.8) ^ 2
    dynamic = gates * 1e-3 * freq_scale * v_scale  # mW
    leakage = gates * 1e-5 * scale  # mW
    # Timing: ~0.05 ns/gate at 130nm
    cp = max(1.0, 10 + 0.01 * n_neurons) * (pdk.min_feature_nm / 130.0)
    max_freq = 1000.0 / cp
    return DesignEstimate(
        module_name="sc_neurocore_top",
        gate_count=gates,
        area_um2=area,
        dynamic_power_mw=dynamic,
        leakage_power_mw=leakage,
        critical_path_ns=cp,
        max_frequency_mhz=max_freq,
    )
end

function label(s::TapeOutChecklistState)
    return f"{s.corner.value}_{s.temperature_c:.0f}C_{s.voltage_v:.2f}V"
end

function generate(s::TapeOutChecklistState)
    pdk: PDKConfig, design: DesignParams, corners: Optional[List[PVTCorner]] = nothing
    ) -> str
    if corners is nothing
        corners = DEFAULT_CORNERS
    lines = [f"# Multi-Corner STA for {design.top_module}"]
    for c in corners
        lib = (
            pdk.liberty_file.replace("_tt_025C_1v80", c.liberty_suffix)
            if c.liberty_suffix
            else pdk.liberty_file
        )
        lines = push!(, f"\n# Corner: {c.label}")
        lines = push!(, f"read_liberty {lib}")
        lines = push!(, f"read_verilog {design.top_module}_final.v")
        lines = push!(, f"link_design {design.top_module}")
        lines = push!(, f"read_sdc constraints_{design.top_module}.sdc")
        lines = push!(, "set_operating_conditions -analysis_type on_chip_variation")
        lines = push!(, "report_checks -path_delay min_max -digits 4")
        lines = push!(, "report_tns")
        lines = push!(, "report_wns")
    return "\n".join(lines) + "\n"
end

function worst_slack(s::TapeOutChecklistState)
    if ! per_corner_wns
        return ("none", 0.0)
    worst = min(per_corner_wns.items(), key=lambda kv: kv[1])
    return worst
end

function generate(s::TapeOutChecklistState)
    if clock_domains is nothing
        clock_domains = [design.clock_name]
    domain_defs = "\n".join(f"create_clock -name {c} [get_ports {c}]" for c in clock_domains)
end

function generate(s::TapeOutChecklistState)
    return nothing
end

function generate(s::TapeOutChecklistState)
    lines = [f"# IO Constraints for {design.top_module}"]
    for pin in pins
        lines = push!(,
            f"place_pin -pin_name {pin.name} -layer {pin.layer} "
            f"-location {{{pin.offset_um} 0}} -side {pin.side}"
        )
    return "\n".join(lines) + "\n"
end

function auto_assign(s::TapeOutChecklistState)
    pins = []
    for i, name in enumerate(signal_names)
        side = sides[i % length(sides)]
        pins = push!(, IOPin(name=name, direction="input", side=side, offset_um=float(i * 10)))
    return pins
end

function generate(s::TapeOutChecklistState)
    return nothing
end

function generate_sdc_fragment(s::TapeOutChecklistState)
    return nothing
end

function conservative(s::TapeOutChecklistState)
    return cls(
        data_cell_early=0.93,
        data_cell_late=1.07,
        data_net_early=0.93,
        data_net_late=1.07,
        clock_cell_early=0.95,
        clock_cell_late=1.05,
    )
end

function drc_clean(s::TapeOutChecklistState)
    return ! any(v.severity == "error" && v.count > 0 for v in s.drc_violations)
end

function all_pass(s::TapeOutChecklistState)
    return (
        s.timing.passed
        && s.power.passed
        && s.area.passed
        && s.drc_clean
        && s.lvs_match
    )
end

function to_dict(s::TapeOutChecklistState)
    return {
        "timing": {"passed": s.timing.passed, "details": s.timing.details},
        "power": {"passed": s.power.passed, "details": s.power.details},
        "area": {"passed": s.area.passed, "details": s.area.details},
        "drc_clean": s.drc_clean,
        "drc_violations": [
            {"rule": v.rule_name, "count": v.count} for v in s.drc_violations
        ],
        "lvs_match": s.lvs_match,
        "all_pass": s.all_pass,
    }
end

function validate_pdk(pdk)
    errors = []
    warnings = []
    if pdk.pdk_type != PDKType.CUSTOM
        if ! pdk.liberty_file
            errors = push!(, "liberty_file is empty")
        if ! pdk.lef_file
            errors = push!(, "lef_file is empty")
        if ! pdk.tech_lef
            errors = push!(, "tech_lef is empty")
    if pdk.clock_period_ns <= 0
        errors = push!(, f"clock_period_ns must be positive, got {pdk.clock_period_ns}")
    if pdk.voltage_v <= 0
        errors = push!(, f"voltage_v must be positive, got {pdk.voltage_v}")
    if pdk.metal_layers < 3
        warnings = push!(, f"only {pdk.metal_layers} metal layers — may limit routing")
    return PDKValidationResult(valid=length(errors) == 0, errors=errors, warnings=warnings)
end

function add_block(s::TapeOutChecklistState, block)
    s.blocks = push!(, block)
end

function block_names(s::TapeOutChecklistState)
    return [b.name for b in s.blocks]
end

function generate_block_scripts(s::TapeOutChecklistState, pdk)
    gen = ASICFlowGenerator()
    result = {}
    for block in s.blocks
        output = gen.generate(pdk, block.design)
        result[block.name] = output.synth_tcl
    return result
end

function generate_top_integration(s::TapeOutChecklistState, pdk)
    lines = [f"# Hierarchical integration for {s.top_design.top_module}"]
    for block in s.blocks
        if block.is_hard_macro && block.abstract_lef
            lines = push!(, f"read_lef {block.abstract_lef}  ;# macro: {block.name}")
    lines = push!(, f"read_verilog synth_{s.top_design.top_module}.v")
    lines = push!(, f"link_design {s.top_design.top_module}")
    return "\n".join(lines) + "\n"
end

function readiness_score(s::TapeOutChecklistState)
    checks = [
        s.synthesis_clean,
        s.timing_met,
        s.power_within_budget,
        s.area_within_limit,
        s.drc_clean,
        s.lvs_clean,
        s.formal_equiv_pass,
        s.cdc_clean,
        s.ir_drop_ok,
        s.esd_reviewed,
    ]
    return sum(1 for c in checks if c) / length(checks)
end

function is_tape_out_ready(s::TapeOutChecklistState)
    return s.readiness_score == 1.0
end

function failing_checks(s::TapeOutChecklistState)
    names = [
        "synthesis_clean",
        "timing_met",
        "power_within_budget",
        "area_within_limit",
        "drc_clean",
        "lvs_clean",
        "formal_equiv_pass",
        "cdc_clean",
        "ir_drop_ok",
        "esd_reviewed",
    ]
    return [n for n in names if ! getattr(self, n)]
end

function from_signoff(s::TapeOutChecklistState, summary)
    s.timing_met = summary.timing.passed
    s.power_within_budget = summary.power.passed
    s.area_within_limit = summary.area.passed
    s.drc_clean = summary.drc_clean
    s.lvs_clean = summary.lvs_match
end

end # module AsicFlowAccel
