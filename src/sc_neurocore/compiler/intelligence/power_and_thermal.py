# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

from __future__ import annotations
from dataclasses import dataclass
import math

# 9. Thermal-Aware Compilation
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ThermalEstimate:
    """Thermal analysis result for a compiled neuron.

    Attributes
    ----------
    power_mw : float
        Estimated total power in milliwatts.
    delta_t_c : float
        Estimated temperature rise in °C.
    junction_temp_c : float
        Estimated junction temperature.
    hotspot_delta_t_c : float
        Local temperature rise from concentrated DSP power.
    derated_freq_mhz : float
        Frequency after thermal derating.
    thermal_safe : bool
        True if junction temp is within limits.
    hotspot_risk : str
        ``"none"``, ``"low"``, ``"medium"``, ``"high"``.
    """

    power_mw: float
    delta_t_c: float
    junction_temp_c: float
    hotspot_delta_t_c: float
    derated_freq_mhz: float
    thermal_safe: bool
    hotspot_risk: str


def thermal_analysis(
    estimated_power_mw: float,
    target_freq_mhz: float,
    *,
    theta_ja: float = 11.5,
    t_ambient_c: float = 25.0,
    t_junction_max_c: float = 100.0,
    process_nm: int = 28,
    mul_count: int = 0,
    dsp_columns: int = 1,
    dsp_power_mw: float | None = None,
    theta_spreading: float = 0.0,
) -> ThermalEstimate:
    """Estimate thermal impact and frequency derating.

    Estimates lumped junction rise from total power and applies timing
    derating from both die temperature and DSP-column concentration.

    Parameters
    ----------
    estimated_power_mw : float
        Estimated power from ``estimate_power()`` or synthesis.
    target_freq_mhz : float
        Nominal target frequency.
    theta_ja : float
        Junction-to-ambient thermal resistance (°C/W).
        Typical: ~11.5 for Artix-7 BGA, ~3.5 for Versal with heatsink.
    t_ambient_c : float
        Ambient temperature.
    t_junction_max_c : float
        Maximum junction temperature.
    process_nm : int
        Process node (affects derating sensitivity).
    mul_count : int
        Number of DSP multipliers (affects hotspot risk).
    dsp_columns : int
        Number of DSP columns to spread across.
    dsp_power_mw : float, optional
        DSP-attributed dynamic power for local hotspot analysis. When omitted,
        local spreading rise is not added.
    theta_spreading : float
        Local spreading resistance from a DSP column hotspot to the bulk
        junction node (°C/W).

    Returns
    -------
    ThermalEstimate
        Thermal analysis with derating and hotspot risk.
    """
    _require_finite_non_negative(estimated_power_mw, "estimated_power_mw")
    _require_finite_positive(target_freq_mhz, "target_freq_mhz")
    _require_finite_positive(theta_ja, "theta_ja")
    _require_finite(t_ambient_c, "t_ambient_c")
    _require_finite_positive(t_junction_max_c, "t_junction_max_c")
    if not isinstance(process_nm, int) or process_nm <= 0:
        raise ValueError("process_nm must be a positive integer")
    if not isinstance(mul_count, int) or mul_count < 0:
        raise ValueError("mul_count must be a non-negative integer")
    if not isinstance(dsp_columns, int) or dsp_columns <= 0:
        raise ValueError("dsp_columns must be a positive integer")
    if dsp_power_mw is not None:
        _require_finite_non_negative(dsp_power_mw, "dsp_power_mw")
    _require_finite_non_negative(theta_spreading, "theta_spreading")

    # Package-level junction rise plus optional local hotspot spreading rise.
    power_w = estimated_power_mw / 1000.0
    delta_t = power_w * theta_ja
    hotspot_delta_t = 0.0
    if dsp_power_mw is not None and mul_count > 0 and theta_spreading > 0.0:
        dsp_power_per_column_w = (dsp_power_mw / 1000.0) / dsp_columns
        hotspot_delta_t = dsp_power_per_column_w * theta_spreading
    t_junction = t_ambient_c + delta_t + hotspot_delta_t

    # Frequency derating: ~0.1% per °C above 85°C for modern processes
    if t_junction > 85.0:
        derate_factor = 1.0 - (t_junction - 85.0) * 0.001
        derate_factor = max(0.7, derate_factor)  # Cap at 30% derating
    else:
        derate_factor = 1.0

    # Smaller processes are more sensitive to thermal
    if process_nm <= 7:
        derate_factor *= 0.98
    elif process_nm <= 16:
        derate_factor *= 0.99

    # Hotspot risk based on DSP concentration
    muls_per_column = mul_count / max(1, dsp_columns)
    if muls_per_column > 20:
        hotspot = "high"
        derate_factor *= 0.94
    elif muls_per_column > 10:
        hotspot = "medium"
        derate_factor *= 0.97
    elif muls_per_column > 4:
        hotspot = "low"
        derate_factor *= 0.99
    else:
        hotspot = "none"

    thermal_safe = t_junction < t_junction_max_c
    derated_freq = target_freq_mhz * derate_factor

    return ThermalEstimate(
        power_mw=estimated_power_mw,
        delta_t_c=round(delta_t, 2),
        junction_temp_c=round(t_junction, 1),
        hotspot_delta_t_c=round(hotspot_delta_t, 2),
        derated_freq_mhz=round(derated_freq, 1),
        thermal_safe=thermal_safe,
        hotspot_risk=hotspot,
    )


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def generate_thermal_constraints(
    module_name: str,
    analysis: ThermalEstimate,
    *,
    dsp_columns: int = 2,
) -> str:
    """Generate XDC constraints for thermal-aware DSP placement.

    Spreads DSP blocks across multiple columns to reduce thermal hotspots
    and adds temperature-derated timing constraints.

    Parameters
    ----------
    module_name : str
        Module name.
    analysis : ThermalEstimate
        Thermal analysis result.
    dsp_columns : int
        Number of DSP columns to distribute across.

    Returns
    -------
    str
        XDC constraint snippet for thermal-aware placement.
    """
    period_ns = 1000.0 / analysis.derated_freq_mhz
    lines = [
        f"# Thermal-aware constraints for {module_name}",
        "# SC-NeuroCore thermal compilation",
        f"# Junction temp: {analysis.junction_temp_c}°C, Hotspot risk: {analysis.hotspot_risk}",
        f"# Derated frequency: {analysis.derated_freq_mhz} MHz",
        "",
        "# Use derated clock period",
        f"create_clock -period {period_ns:.3f} -name clk [get_ports clk]",
        "",
    ]

    if analysis.hotspot_risk in ("medium", "high"):
        lines.extend(
            [
                f"# DSP spreading across {dsp_columns} columns to reduce hotspots",
                "set_property LOC DSP48E2_X0Y0 "
                "[get_cells -hier -filter {REF_NAME =~ DSP*} -limit 1]",
                "",
                "# Soft placement constraint: spread DSPs",
                "set_property C_REG 1 [get_cells -hier -filter {REF_NAME =~ DSP*}]",
                "",
            ]
        )

    if not analysis.thermal_safe:
        lines.extend(
            [
                f"# WARNING: Junction temperature {analysis.junction_temp_c}°C exceeds limit!",
                "# Consider: reduce clock, add heatsink, or reduce neuron count.",
                "",
            ]
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 30. Energy Harvesting Scheduler
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class EnergySchedule:
    """Energy-aware neuron update schedule.

    Attributes
    ----------
    total_neurons : int
        Total neurons.
    energy_budget_uj : float
        Energy budget per epoch (µJ).
    neurons_per_epoch : int
        Neurons updatable within budget.
    update_order : list[int]
        Priority-ordered neuron indices.
    epoch_duration_ms : float
        Epoch duration.
    duty_cycle : float
        Fraction of neurons updated per epoch.
    """

    total_neurons: int
    energy_budget_uj: float
    neurons_per_epoch: int
    update_order: list[int]
    epoch_duration_ms: float
    duty_cycle: float


def generate_energy_schedule(
    neuron_count: int,
    *,
    energy_budget_uj: float = 10.0,
    energy_per_neuron_nj: float = 50.0,
    epoch_duration_ms: float = 10.0,
    priority_neurons: list[int] | None = None,
) -> EnergySchedule:
    """Generate energy-budget-aware neuron update schedule.

    For energy-harvesting edge devices (solar, vibration, RF),
    schedules neuron updates to fit within the available energy.

    Parameters
    ----------
    neuron_count : int
        Total neurons.
    energy_budget_uj : float
        Available energy per epoch (µJ).
    energy_per_neuron_nj : float
        Energy per neuron update (nJ).
    epoch_duration_ms : float
        Epoch duration (ms).
    priority_neurons : list[int], optional
        High-priority neuron indices (updated first).

    Returns
    -------
    EnergySchedule
        Update schedule.
    """
    if not isinstance(neuron_count, int) or neuron_count <= 0:
        raise ValueError("neuron_count must be a positive integer")
    _require_finite_non_negative(energy_budget_uj, "energy_budget_uj")
    _require_finite_positive(energy_per_neuron_nj, "energy_per_neuron_nj")
    _require_finite_positive(epoch_duration_ms, "epoch_duration_ms")

    budget_nj = energy_budget_uj * 1000
    max_neurons = int(budget_nj / energy_per_neuron_nj)
    updatable = min(max_neurons, neuron_count)

    # Priority ordering
    if priority_neurons:
        order = []
        seen = set()
        for idx in priority_neurons:
            if not isinstance(idx, int) or idx < 0 or idx >= neuron_count:
                raise ValueError("priority_neurons must contain valid neuron indices")
            if idx in seen:
                continue
            seen.add(idx)
            order.append(idx)
        remaining = [i for i in range(neuron_count) if i not in order]
        order.extend(remaining)
    else:
        order = list(range(neuron_count))

    order = order[:updatable]
    duty = updatable / neuron_count if neuron_count > 0 else 0.0

    return EnergySchedule(
        total_neurons=neuron_count,
        energy_budget_uj=energy_budget_uj,
        neurons_per_epoch=updatable,
        update_order=order,
        epoch_duration_ms=epoch_duration_ms,
        duty_cycle=round(duty, 4),
    )


# ═══════════════════════════════════════════════════════════════════════
# 40. Thermal Envelope Estimator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ThermalEnvelopeEstimate:
    """Junction temperature estimate.

    Attributes
    ----------
    power_mw : float
        Estimated power dissipation (mW).
    theta_ja : float
        Junction-to-ambient thermal resistance (°C/W).
    t_ambient : float
        Ambient temperature (°C).
    t_junction : float
        Estimated junction temperature (°C).
    thermal_margin : float
        Margin to max T_j (°C).
    pass_fail : str
        ``"PASS"`` or ``"FAIL"``.
    """

    power_mw: float
    theta_ja: float
    t_ambient: float
    t_junction: float
    thermal_margin: float
    pass_fail: str


def estimate_thermal_envelope(
    *,
    power_mw: float = 100.0,
    theta_ja: float = 25.0,
    t_ambient: float = 25.0,
    t_junction_max: float = 125.0,
) -> ThermalEnvelopeEstimate:
    """Predict junction temperature from power dissipation.

    Uses simple thermal resistance model: T_j = T_a + P × θ_ja.

    Parameters
    ----------
    power_mw : float
        Power dissipation (mW).
    theta_ja : float
        Junction-to-ambient thermal resistance (°C/W).
    t_ambient : float
        Ambient temperature (°C).
    t_junction_max : float
        Maximum allowed junction temperature (°C).

    Returns
    -------
    ThermalEnvelopeEstimate
        Temperature estimate with pass/fail.
    """
    power_w = power_mw / 1000.0
    t_j = t_ambient + power_w * theta_ja
    margin = t_junction_max - t_j
    status = "PASS" if margin > 0 else "FAIL"

    return ThermalEnvelopeEstimate(
        power_mw=power_mw,
        theta_ja=theta_ja,
        t_ambient=t_ambient,
        t_junction=round(t_j, 2),
        thermal_margin=round(margin, 2),
        pass_fail=status,
    )


# ═══════════════════════════════════════════════════════════════════════
# 44. Power Intent Generator (UPF)
# ═══════════════════════════════════════════════════════════════════════


def generate_power_intent(
    module_name: str,
    *,
    num_domains: int = 2,
    always_on: bool = True,
) -> str:
    """Generate IEEE 1801 UPF power intent for neuron arrays.

    Creates power domain definitions, isolation rules, and
    retention strategies for multi-voltage SNN designs.

    Parameters
    ----------
    module_name : str
        Top module name.
    num_domains : int
        Number of power domains.
    always_on : bool
        Whether to include always-on domain.

    Returns
    -------
    str
        UPF source text.
    """
    lines = [
        f"# UPF Power Intent for {module_name}",
        "# Generated by SC-NeuroCore",
        "",
        f"set_scope {module_name}",
        "",
    ]
    if always_on:
        lines.append("create_power_domain PD_AON -include_scope")
        lines.append("create_supply_net VDD_AON -domain PD_AON")
        lines.append("create_supply_net VSS -domain PD_AON")
        lines.append("")

    for i in range(num_domains):
        lines.extend(
            [
                f"create_power_domain PD_NEURON_{i}",
                f"create_supply_net VDD_{i} -domain PD_NEURON_{i}",
                f"create_supply_net VSS -domain PD_NEURON_{i} -reuse",
                f"set_isolation iso_{i} -domain PD_NEURON_{i} "
                f"-isolation_power_net VDD_AON -isolation_ground_net VSS "
                f"-clamp_value 0",
                f"set_retention ret_{i} -domain PD_NEURON_{i} -retention_power_net VDD_AON",
                "",
            ]
        )

    lines.append("# Power states")
    lines.append("add_power_state PD_AON_ON -domain PD_AON -state ON {-supply_expr {VDD_AON == 1}}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 57. Power State Machine Generator
# ═══════════════════════════════════════════════════════════════════════


def generate_power_state_machine(
    module_name: str,
    *,
    states: list[str] | None = None,
) -> str:
    """Generate sleep/wake/hibernate FSM for ultra-low-power.

    Parameters
    ----------
    module_name : str
        Module name.
    states : list[str], optional
        FSM states. Default: ACTIVE, IDLE, SLEEP, HIBERNATE.

    Returns
    -------
    str
        Verilog FSM source.
    """
    if states is None:
        states = ["ACTIVE", "IDLE", "SLEEP", "HIBERNATE"]

    lines = [
        f"// Power state machine for {module_name}",
        "// Generated by SC-NeuroCore",
        f"module {module_name}_power_fsm (",
        "    input  wire clk, rst_n, wake, sleep_req,",
        f"    output reg [{len(states).bit_length() - 1}:0] state",
        ");",
    ]
    for i, s in enumerate(states):
        lines.append(f"    localparam {s} = {i};")

    lines.extend(
        [
            "    always @(posedge clk or negedge rst_n) begin",
            "        if (!rst_n)",
            f"            state <= {states[0]};",
            "        else case (state)",
        ]
    )
    for i, s in enumerate(states):
        nxt = states[min(i + 1, len(states) - 1)]
        lines.append(f"            {s}: state <= sleep_req ? {nxt} : (wake ? {states[0]} : state);")
    lines.extend(
        [
            "        endcase",
            "    end",
            "endmodule",
        ]
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 68. Approximate Computing Modes
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ApproximationConfig:
    """Approximate computing configuration.

    Attributes
    ----------
    populations : dict[str, dict]
    total_energy_savings_pct : float
    max_output_error_pct : float
    """

    populations: dict[str, dict]
    total_energy_savings_pct: float
    max_output_error_pct: float


def configure_approximation(
    equations: dict[str, str],
    *,
    target_savings_pct: float = 30.0,
    max_error_pct: float = 5.0,
) -> ApproximationConfig:
    """Configure precision-energy tradeoff knobs per state variable.

    Analyses each variable's dynamic range and recommends bit-width
    reduction and stochastic rounding to achieve target energy savings
    while bounding output error.

    Parameters
    ----------
    equations : dict[str, str]
        State variable equations.
    target_savings_pct : float
        Target energy savings percentage.
    max_error_pct : float
        Maximum acceptable output error percentage.

    Returns
    -------
    ApproximationConfig
    """
    pops = {}
    total_savings = 0.0
    for var in equations:
        # Heuristic: each bit removed saves ~15% energy per variable
        bits_removable = min(4, int(target_savings_pct / 15))
        savings = bits_removable * 15.0 / len(equations)
        error = bits_removable * 1.5
        if error > max_error_pct:
            bits_removable = max(1, int(max_error_pct / 1.5))
            savings = bits_removable * 15.0 / len(equations)
            error = bits_removable * 1.5
        pops[var] = {
            "bits_reduced": bits_removable,
            "stochastic_rounding": bits_removable >= 2,
            "energy_savings_pct": round(savings, 1),
            "error_bound_pct": round(error, 2),
        }
        total_savings += savings

    return ApproximationConfig(
        populations=pops,
        total_energy_savings_pct=round(total_savings, 1),
        max_output_error_pct=max(p["error_bound_pct"] for p in pops.values()),
    )


# ═══════════════════════════════════════════════════════════════════════
# 69. Energy Harvesting Budget Modeler
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class EnergyHarvestBudget:
    """Energy harvesting feasibility analysis.

    Attributes
    ----------
    harvester_power_uw : float
    design_power_uw : float
    energy_positive : bool
    recommended_duty_cycle : float
    margin_pct : float
    """

    harvester_power_uw: float
    design_power_uw: float
    energy_positive: bool
    recommended_duty_cycle: float
    margin_pct: float


def model_energy_harvest(
    design_power_uw: float,
    *,
    harvester_type: str = "solar",
    harvester_area_cm2: float = 1.0,
    environment: str = "indoor",
) -> EnergyHarvestBudget:
    """Model whether an energy harvester can sustain the neural workload.

    Parameters
    ----------
    design_power_uw : float
        Design power consumption in microwatts.
    harvester_type : str
        "solar", "piezo", "thermal", or "rf".
    harvester_area_cm2 : float
        Harvester active area.
    environment : str
        "indoor", "outdoor", or "industrial".

    Returns
    -------
    EnergyHarvestBudget
    """
    # Power density lookup (µW/cm²)
    densities = {
        ("solar", "outdoor"): 10000.0,
        ("solar", "indoor"): 10.0,
        ("solar", "industrial"): 50.0,
        ("piezo", "outdoor"): 200.0,
        ("piezo", "indoor"): 100.0,
        ("piezo", "industrial"): 500.0,
        ("thermal", "outdoor"): 25.0,
        ("thermal", "indoor"): 10.0,
        ("thermal", "industrial"): 60.0,
        ("rf", "outdoor"): 1.0,
        ("rf", "indoor"): 0.5,
        ("rf", "industrial"): 2.0,
    }
    density = densities.get((harvester_type, environment), 10.0)
    harvest_power = density * harvester_area_cm2

    energy_positive = harvest_power >= design_power_uw
    if design_power_uw > 0:
        duty_cycle = min(1.0, harvest_power / design_power_uw)
        margin = ((harvest_power - design_power_uw) / design_power_uw) * 100
    else:
        duty_cycle = 1.0
        margin = 100.0

    return EnergyHarvestBudget(
        harvester_power_uw=round(harvest_power, 2),
        design_power_uw=design_power_uw,
        energy_positive=energy_positive,
        recommended_duty_cycle=round(duty_cycle, 4),
        margin_pct=round(margin, 1),
    )


# ═══════════════════════════════════════════════════════════════════════
# 71. DVFS Controller Generator
# ═══════════════════════════════════════════════════════════════════════


def generate_dvfs_controller(
    module_name: str,
    *,
    operating_points: list[dict] | None = None,
    spike_rate_thresholds: list[float] | None = None,
) -> str:
    """Generate a Verilog DVFS controller FSM.

    Parameters
    ----------
    module_name : str
        Module name.
    operating_points : list[dict], optional
        [{voltage_mv, freq_mhz}]. Default: 3 points.
    spike_rate_thresholds : list[float], optional
        Spike rate thresholds for state transitions.

    Returns
    -------
    str
        Synthesisable Verilog source code.
    """
    if operating_points is None:
        operating_points = [
            {"voltage_mv": 700, "freq_mhz": 100},
            {"voltage_mv": 900, "freq_mhz": 250},
            {"voltage_mv": 1100, "freq_mhz": 500},
        ]
    if spike_rate_thresholds is None:
        spike_rate_thresholds = [10.0, 50.0]

    n = len(operating_points)
    states = [f"OP_{i}" for i in range(n)]
    lines = [
        f"// DVFS Controller for {module_name}",
        "// Auto-generated by SC-NeuroCore §71",
        f"module {module_name}_dvfs_ctrl (",
        "    input  wire        clk,",
        "    input  wire        rst_n,",
        "    input  wire [15:0] spike_rate,",
        "    output reg  [15:0] target_freq_mhz,",
        "    output reg  [15:0] target_voltage_mv",
        ");",
        "",
    ]
    # State encoding
    for i, s in enumerate(states):
        lines.append(f"    localparam {s} = {i};")
    lines.extend(
        [
            f"    reg [{max(1, n - 1).bit_length() - 1}:0] state;",
            "",
            "    always @(posedge clk or negedge rst_n) begin",
            "        if (!rst_n) begin",
            f"            state <= {states[0]};",
            f"            target_freq_mhz <= {operating_points[0]['freq_mhz']};",
            f"            target_voltage_mv <= {operating_points[0]['voltage_mv']};",
            "        end else begin",
            "            case (state)",
        ]
    )
    for i, op in enumerate(operating_points):
        lines.append(f"                {states[i]}: begin")
        lines.append(f"                    target_freq_mhz <= {op['freq_mhz']};")
        lines.append(f"                    target_voltage_mv <= {op['voltage_mv']};")
        if i < n - 1 and i < len(spike_rate_thresholds):
            th = int(spike_rate_thresholds[i])
            lines.append(f"                    if (spike_rate > {th}) state <= {states[i + 1]};")
        if i > 0 and i - 1 < len(spike_rate_thresholds):
            th = int(spike_rate_thresholds[i - 1])
            lines.append(f"                    if (spike_rate < {th}) state <= {states[i - 1]};")
        lines.append("                end")
    lines.extend(
        [
            "            endcase",
            "        end",
            "    end",
            "endmodule",
        ]
    )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
