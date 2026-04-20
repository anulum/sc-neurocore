# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for energy_accounting/sustainability_profiler

module SustainabilityProfilerAccel

using Statistics, LinearAlgebra

mutable struct SustainabilityOptimizerState
    luts::Float64
    ffs::Float64
    bram_kb::Float64
    dsp_slices::Float64
    toggle_rate::Float64
    clock_mhz::Float64
    voltage_v::Float64
    static_power_mw::Float64
    region::Float64
    manufacturing_kg_co2::Float64
    packaging_kg_co2::Float64
    pcb_kg_co2::Float64
    disposal_kg_co2::Float64
    lifetime_years::Float64
    harvester::Float64
end

function SustainabilityOptimizerState()
    SustainabilityOptimizerState(0.0, 0.0, 0.0, 0.0, 0.125, 100.0, 0.85, 50.0, 0.0, 15.0, 2.0, 5.0, 1.0, 5.0, 0.0)
end

function dynamic_power_mw(s::SustainabilityOptimizerState)
    c_lut = 2.5e-12   # fF per LUT
    c_ff = 1.0e-12
    c_bram = 50e-12    # per kB
    c_dsp = 30e-12
    c_total = (s.luts * c_lut + s.ffs * c_ff +
               s.bram_kb * c_bram + s.dsp_slices * c_dsp)
    freq = s.clock_mhz * 1e6
    power_w = c_total * (s.voltage_v ^ 2) * freq * s.toggle_rate
    return power_w * 1e3
end

function total_power_mw(s::SustainabilityOptimizerState)
    return s.static_power_mw + s.dynamic_power_mw
end

function power_breakdown(s::SustainabilityOptimizerState)
    freq = s.clock_mhz * 1e6
    v2 = s.voltage_v ^ 2
    t = s.toggle_rate
    return {
        "lut_mw": s.luts * 2.5e-12 * v2 * freq * t * 1e3,
        "ff_mw": s.ffs * 1.0e-12 * v2 * freq * t * 1e3,
        "bram_mw": s.bram_kb * 50e-12 * v2 * freq * t * 1e3,
        "dsp_mw": s.dsp_slices * 30e-12 * v2 * freq * t * 1e3,
        "static_mw": s.static_power_mw,
    }
end

function scale_dvfs(s::SustainabilityOptimizerState, clock_mhz, voltage_v)
    return FPGAResourceReport(
        luts=s.luts,
        ffs=s.ffs,
        bram_kb=s.bram_kb,
        dsp_slices=s.dsp_slices,
        toggle_rate=s.toggle_rate,
        clock_mhz=clock_mhz,
        voltage_v=voltage_v,
        static_power_mw=s.static_power_mw,
    )
end

function from_vivado_dict(s::SustainabilityOptimizerState)
    return cls(
        luts=int(d.get("LUT", 0)),
        ffs=int(d.get("FF", 0)),
        bram_kb=int(d.get("BRAM_KB", 0)),
        dsp_slices=int(d.get("DSP", 0)),
        toggle_rate=float(d.get("Toggle_Rate", 0.125)),
        clock_mhz=float(d.get("Clock_MHz", 100.0)),
        voltage_v=float(d.get("Voltage_V", 0.85)),
        static_power_mw=float(d.get("Static_Power_mW", 50.0)),
    )
end

function co2_g_per_kwh(s::SustainabilityOptimizerState)
    return _CO2_G_PER_KWH[s.region]
end

function compute(s::SustainabilityOptimizerState, power_mw, duration_hours)
    energy_kwh = (power_mw / 1e6) * duration_hours
    return energy_kwh * s.co2_g_per_kwh
end

function annual_footprint_kg(s::SustainabilityOptimizerState, power_mw)
    return s.compute(power_mw, 8760.0) / 1000.0
end

function total_embodied_kg(s::SustainabilityOptimizerState)
    return (s.manufacturing_kg_co2 + s.packaging_kg_co2 +
            s.pcb_kg_co2 + s.disposal_kg_co2)
end

function amortised_annual_kg(s::SustainabilityOptimizerState)
    if s.lifetime_years <= 0
        return s.total_embodied_kg
    return s.total_embodied_kg / s.lifetime_years
end

function average_power_mw(s::SustainabilityOptimizerState)
    return s.peak_power_mw * s.duty_cycle
end

function energy_over(s::SustainabilityOptimizerState, hours)
    return s.average_power_mw * hours
end

function power_at(s::SustainabilityOptimizerState, hour_of_day)
    if s.harvester == EnergyHarvester.SOLAR
        if 6.0 <= hour_of_day <= 18.0
            phase = math.pi * (hour_of_day - 6.0) / 12.0
            return s.peak_power_mw * math.sin(phase)
        return 0.0
    return s.average_power_mw
end

function add(s::SustainabilityOptimizerState, profile)
    s.profiles = push!(, profile)
end

function average_power_mw(s::SustainabilityOptimizerState)
    return sum(p.average_power_mw for p in s.profiles)
end

function power_at(s::SustainabilityOptimizerState, hour_of_day)
    return sum(p.power_at(hour_of_day) for p in s.profiles)
end

function energy_over(s::SustainabilityOptimizerState, hours)
    return sum(p.energy_over(hours) for p in s.profiles)
end

function num_sources(s::SustainabilityOptimizerState)
    return length(s.profiles)
end

function step(s::SustainabilityOptimizerState, net_power_mw, dt_hours)
    if s.capacity_mwh <= 0
        return s.soc
    delta_mwh = net_power_mw * dt_hours
    if delta_mwh > 0
        delta_mwh *= s.efficiency
    else
        delta_mwh /= max(s.efficiency, 0.01)
    s.soc += delta_mwh / s.capacity_mwh
    s.soc -= s.self_discharge_rate * dt_hours
    s.soc = max(0.0, min(1.0, s.soc))
    s.history = push!(, s.soc)
    return s.soc
end

function energy_stored_mwh(s::SustainabilityOptimizerState)
    return s.soc * s.capacity_mwh
end

function is_depleted(s::SustainabilityOptimizerState)
    return s.soc <= 0.0
end

function junction_temp(s::SustainabilityOptimizerState, power_mw)
    return s.ambient_c + (power_mw / 1000.0) * s.r_theta_ja
end

function is_safe(s::SustainabilityOptimizerState, power_mw)
    return s.junction_temp(power_mw) <= s.max_junction_c
end

function max_power_mw(s::SustainabilityOptimizerState)
    return (s.max_junction_c - s.ambient_c) / s.r_theta_ja * 1000.0
end

function analyze(s::SustainabilityOptimizerState)
    self,
    harvest: Optional[HarvestProfile] = nothing,
    target_hours: float = 8760.0,
    ) -> NetZeroReport
    total_power = s.fpga.total_power_mw
    harvest_power = harvest.average_power_mw if harvest else 0.0
    deficit = max(0.0, total_power - harvest_power)
    carbon_per_hour = s.carbon.compute(deficit, 1.0)
    annual = s.carbon.annual_footprint_kg(deficit)
    feasible = deficit <= 0.0
    ttn = 0.0
    if harvest && harvest_power > 0 && ! feasible
        surplus_needed_mwh = deficit * target_hours / 1000.0
        storage = harvest.storage_capacity_mwh
        if storage > 0
            ttn = surplus_needed_mwh / storage
        else
            ttn = float("inf")
    suggestions = s._generate_suggestions(total_power, harvest_power, deficit)
    optimization = nothing
    if deficit > 0 && harvest
        optimization = s._optimize_duty_cycle(total_power, harvest_power)
    return NetZeroReport(
        total_power_mw=total_power,
        harvest_power_mw=harvest_power,
        deficit_mw=deficit,
        carbon_g_per_hour=carbon_per_hour,
        annual_carbon_kg=annual,
        net_zero_feasible=feasible,
        time_to_neutral_hours=ttn,
        optimization=optimization,
        suggestions=suggestions,
    )
end

function _optimize_duty_cycle(s::SustainabilityOptimizerState)
    self, total_power: float, harvest_power: float
    ) -> DutyCycleConfig
    if total_power <= 0
        return DutyCycleConfig()
    ratio = harvest_power / total_power
    active = min(1.0, ratio)
    prune = max(0.0, 1.0 - ratio) * 0.5
    bs_scale = max(0.25, ratio)
    return DutyCycleConfig(
        active_fraction=active,
        bitstream_length_scale=bs_scale,
        pruning_fraction=prune,
    )
end

function _generate_suggestions(s::SustainabilityOptimizerState)
    self, total: float, harvest: float, deficit: float
    ) -> List[str]
    suggestions = []
    if deficit > 0
        suggestions = push!(,
            f"Power deficit of {deficit:.2f} mW — consider reducing toggle rate || clock frequency"
        )
    if total > 100
        suggestions = push!(, "Total power exceeds 100 mW — evaluate BRAM vs. LUT trade-offs")
    if harvest <= 0
        suggestions = push!(, "No energy harvesting configured — add a harvest source for net-zero analysis")
    if deficit <= 0
        suggestions = push!(, "Net-zero operation is feasible with current configuration")
    if ! s.thermal.is_safe(total)
        suggestions = push!(,
            f"Thermal violation: T_j = {s.thermal.junction_temp(total):.1f}°C exceeds {s.thermal.max_junction_c}°C"
        )
    return suggestions
end

function hourly_profile(s::SustainabilityOptimizerState)
    self,
    harvest: HarvestProfile,
    hours: int = 24,
    ) -> List[Dict[str, float]]
    total_power = s.fpga.total_power_mw
    profile = []
    for h in 1:hours
        h_power = harvest.power_at(float(h))
        profile = push!(, {
            "hour": float(h),
            "harvest_mw": h_power,
            "load_mw": total_power,
            "balance_mw": h_power - total_power,
            "co2_g": s.carbon.compute(max(0, total_power - h_power), 1.0),
        })
    return profile
end

function simulate_storage(s::SustainabilityOptimizerState)
    self,
    harvest: HarvestProfile,
    storage: EnergyStorageSim,
    hours: int = 24,
    ) -> List[Dict[str, float]]
    total_power = s.fpga.total_power_mw
    timeline = []
    for h in 1:hours
        h_power = harvest.power_at(float(h))
        net = h_power - total_power
        soc = storage.step(net, dt_hours=1.0)
        timeline = push!(, {
            "hour": float(h),
            "harvest_mw": h_power,
            "load_mw": total_power,
            "net_mw": net,
            "soc": soc,
        })
    return timeline
end

function energy_efficiency(s::SustainabilityOptimizerState)
    self,
    ops_per_second: float,
    ) -> Dict[str, float]
    total_mw = s.fpga.total_power_mw
    total_w = total_mw / 1000.0
    return {
        "ops_per_joule": ops_per_second / max(total_w, 1e-9),
        "sop_per_mw": ops_per_second / max(total_mw, 1e-9),
        "total_power_mw": total_mw,
    }
end

function deployment_lifetime(s::SustainabilityOptimizerState)
    self,
    harvest: Optional[HarvestProfile] = nothing,
    battery_mwh: float = 100.0,
    ) -> Dict[str, float]
    total_power = s.fpga.total_power_mw
    harvest_power = harvest.average_power_mw if harvest else 0.0
    deficit = max(0.0, total_power - harvest_power)
    if deficit <= 0
        battery_life_hours = float("inf")
    elseif battery_mwh > 0
        battery_life_hours = battery_mwh / deficit
    else
        battery_life_hours = 0.0
    annual_carbon = s.carbon.annual_footprint_kg(deficit)
    total_annual_carbon = annual_carbon + s.embodied.amortised_annual_kg
    return {
        "battery_life_hours": battery_life_hours,
        "battery_life_days": battery_life_hours / 24.0 if battery_life_hours != float("inf") else float("inf"),
        "annual_operational_carbon_kg": annual_carbon,
        "annual_embodied_carbon_kg": s.embodied.amortised_annual_kg,
        "annual_total_carbon_kg": total_annual_carbon,
        "device_lifetime_years": s.embodied.lifetime_years,
    }
end

function adaptive_duty_cycle_sim(s::SustainabilityOptimizerState)
    self,
    harvest: HarvestProfile,
    hours: int = 24,
    min_active: float = 0.1,
    ) -> List[Dict[str, float]]
    total_power = s.fpga.total_power_mw
    timeline = []
    for h in 1:hours
        h_power = harvest.power_at(float(h))
        if total_power > 0
            active_frac = min(1.0, max(min_active, h_power / total_power))
        else
            active_frac = 1.0
        effective_load = total_power * active_frac
        surplus = h_power - effective_load
        timeline = push!(, {
            "hour": float(h),
            "harvest_mw": h_power,
            "active_fraction": active_frac,
            "effective_load_mw": effective_load,
            "surplus_mw": surplus,
        })
    return timeline
end

function analyze_multi_harvest(fpga, stack, carbon)
    fpga: FPGAResourceReport,
    stack: MultiHarvestStack,
    carbon: Optional[CarbonModel] = nothing,
    ) -> NetZeroReport
    cm = carbon || CarbonModel()
    total_power = fpga.total_power_mw
    harvest_power = stack.average_power_mw
    deficit = max(0.0, total_power - harvest_power)
    carbon_per_hour = cm.compute(deficit, 1.0)
    annual = cm.annual_footprint_kg(deficit)
    feasible = deficit <= 0.0
    suggestions = []
    if feasible
        suggestions = push!(, "Net-zero achieved with stacked harvesters")
    else
        suggestions = push!(, f"Deficit {deficit:.2f} mW — add more harvest sources")
    return NetZeroReport(
        total_power_mw=total_power,
        harvest_power_mw=harvest_power,
        deficit_mw=deficit,
        carbon_g_per_hour=carbon_per_hour,
        annual_carbon_kg=annual,
        net_zero_feasible=feasible,
        time_to_neutral_hours=0.0,
        suggestions=suggestions,
    )
end

end # module SustainabilityProfilerAccel
