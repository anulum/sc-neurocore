# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sustainable Neuromorphics Profiler

"""Integrated power/carbon modelling with energy-harvesting simulation.

Quantifies and optimises the carbon footprint of neuromorphic SC
deployments end-to-end.  Ingests FPGA resource reports (LUT/FF/toggle
counts), models dynamic/static power, computes CO₂ emissions per grid
region, and simulates energy-harvesting sources (piezo, RF, solar,
bio-fuel, thermoelectric) to determine net-zero feasibility.

Includes a ``SustainabilityOptimizer`` that auto-duty-cycles bitstreams
and prunes during low-harvest periods to achieve energy-neutral operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── FPGA Resource Modelling ──────────────────────────────────────────


@dataclass
class FPGAResourceReport:
    """Vivado-style utilisation report."""

    luts: int = 0
    ffs: int = 0
    bram_kb: int = 0
    dsp_slices: int = 0
    toggle_rate: float = 0.125  # average toggle fraction (0–1)
    clock_mhz: float = 100.0
    voltage_v: float = 0.85  # core voltage
    static_power_mw: float = 50.0  # leakage (package-dependent)

    @property
    def dynamic_power_mw(self) -> float:
        """Estimate dynamic power: P = C_eff * V² * f * activity.

        Uses an empirical per-resource capacitance model.
        """
        c_lut = 2.5e-12  # fF per LUT
        c_ff = 1.0e-12
        c_bram = 50e-12  # per kB
        c_dsp = 30e-12

        c_total = (
            self.luts * c_lut + self.ffs * c_ff + self.bram_kb * c_bram + self.dsp_slices * c_dsp
        )
        freq = self.clock_mhz * 1e6
        power_w = c_total * (self.voltage_v**2) * freq * self.toggle_rate
        return power_w * 1e3

    @property
    def total_power_mw(self) -> float:
        return self.static_power_mw + self.dynamic_power_mw

    def power_breakdown(self) -> Dict[str, float]:
        """Per-component dynamic power breakdown in mW."""
        freq = self.clock_mhz * 1e6
        v2 = self.voltage_v**2
        t = self.toggle_rate
        return {
            "lut_mw": self.luts * 2.5e-12 * v2 * freq * t * 1e3,
            "ff_mw": self.ffs * 1.0e-12 * v2 * freq * t * 1e3,
            "bram_mw": self.bram_kb * 50e-12 * v2 * freq * t * 1e3,
            "dsp_mw": self.dsp_slices * 30e-12 * v2 * freq * t * 1e3,
            "static_mw": self.static_power_mw,
        }

    def scale_dvfs(self, clock_mhz: float, voltage_v: float) -> FPGAResourceReport:
        """Return a new report with scaled clock and voltage (DVFS)."""
        return FPGAResourceReport(
            luts=self.luts,
            ffs=self.ffs,
            bram_kb=self.bram_kb,
            dsp_slices=self.dsp_slices,
            toggle_rate=self.toggle_rate,
            clock_mhz=clock_mhz,
            voltage_v=voltage_v,
            static_power_mw=self.static_power_mw,
        )

    @classmethod
    def from_vivado_dict(cls, d: Dict[str, Any]) -> FPGAResourceReport:
        """Parse from a Vivado utilisation report dictionary.

        Expected keys: 'LUT', 'FF', 'BRAM_KB', 'DSP', 'Toggle_Rate',
        'Clock_MHz', 'Voltage_V', 'Static_Power_mW'.
        """
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


# ── Carbon Model ─────────────────────────────────────────────────────


class GridRegion(Enum):
    EU = "eu"
    US = "us"
    CN = "cn"
    GLOBAL = "global"
    RENEWABLE = "renewable"


_CO2_G_PER_KWH: Dict[GridRegion, float] = {
    GridRegion.EU: 230.0,
    GridRegion.US: 380.0,
    GridRegion.CN: 540.0,
    GridRegion.GLOBAL: 420.0,
    GridRegion.RENEWABLE: 20.0,
}


@dataclass
class CarbonModel:
    """CO₂ emissions model for a given power profile."""

    region: GridRegion = GridRegion.GLOBAL

    @property
    def co2_g_per_kwh(self) -> float:
        return _CO2_G_PER_KWH[self.region]

    def compute(self, power_mw: float, duration_hours: float) -> float:
        """Return grams of CO₂ emitted."""
        energy_kwh = (power_mw / 1e6) * duration_hours
        return energy_kwh * self.co2_g_per_kwh

    def annual_footprint_kg(self, power_mw: float) -> float:
        """Annual CO₂ in kilograms (24/7 operation)."""
        return self.compute(power_mw, 8760.0) / 1000.0


@dataclass
class EmbodiedCarbon:
    """Manufacturing + end-of-life carbon footprint."""

    manufacturing_kg_co2: float = 15.0  # FPGA chip fabrication
    packaging_kg_co2: float = 2.0
    pcb_kg_co2: float = 5.0
    disposal_kg_co2: float = 1.0
    lifetime_years: float = 5.0

    @property
    def total_embodied_kg(self) -> float:
        return (
            self.manufacturing_kg_co2
            + self.packaging_kg_co2
            + self.pcb_kg_co2
            + self.disposal_kg_co2
        )

    @property
    def amortised_annual_kg(self) -> float:
        """Annual embodied carbon amortised over device lifetime."""
        if self.lifetime_years <= 0:
            return self.total_embodied_kg
        return self.total_embodied_kg / self.lifetime_years


# ── Energy Harvesting ────────────────────────────────────────────────


class EnergyHarvester(Enum):
    PIEZO = "piezo"
    RF = "rf"
    SOLAR = "solar"
    BIOFUEL = "biofuel"
    THERMOELECTRIC = "thermoelectric"


_HARVEST_PEAK_MW: Dict[EnergyHarvester, float] = {
    EnergyHarvester.PIEZO: 0.5,
    EnergyHarvester.RF: 0.1,
    EnergyHarvester.SOLAR: 50.0,
    EnergyHarvester.BIOFUEL: 2.0,
    EnergyHarvester.THERMOELECTRIC: 5.0,
}


@dataclass
class HarvestProfile:
    """Energy-harvesting power curve."""

    harvester: EnergyHarvester
    peak_power_mw: float = 0.0
    duty_cycle: float = 0.5
    storage_capacity_mwh: float = 0.01

    def __post_init__(self):
        if self.peak_power_mw <= 0:
            self.peak_power_mw = _HARVEST_PEAK_MW.get(self.harvester, 1.0)

    @property
    def average_power_mw(self) -> float:
        return self.peak_power_mw * self.duty_cycle

    def energy_over(self, hours: float) -> float:
        """Total energy harvested in mWh."""
        return self.average_power_mw * hours

    def power_at(self, hour_of_day: float) -> float:
        """Instantaneous power at a given hour (0–24).

        Solar follows a sinusoidal day curve; others are constant.
        """
        if self.harvester == EnergyHarvester.SOLAR:
            if 6.0 <= hour_of_day <= 18.0:
                phase = math.pi * (hour_of_day - 6.0) / 12.0
                return self.peak_power_mw * math.sin(phase)
            return 0.0
        return self.average_power_mw


class MultiHarvestStack:
    """Combines multiple energy-harvesting sources."""

    def __init__(self, profiles: Optional[List[HarvestProfile]] = None):
        self.profiles: List[HarvestProfile] = profiles or []

    def add(self, profile: HarvestProfile) -> None:
        self.profiles.append(profile)

    @property
    def average_power_mw(self) -> float:
        return sum(p.average_power_mw for p in self.profiles)

    def power_at(self, hour_of_day: float) -> float:
        return sum(p.power_at(hour_of_day) for p in self.profiles)

    def energy_over(self, hours: float) -> float:
        return sum(p.energy_over(hours) for p in self.profiles)

    @property
    def num_sources(self) -> int:
        return len(self.profiles)


# ── Energy Storage Simulation ────────────────────────────────────────


@dataclass
class EnergyStorageSim:
    """Battery/supercap state-of-charge simulation."""

    capacity_mwh: float = 10.0
    initial_soc: float = 0.5  # 0–1
    efficiency: float = 0.9  # round-trip
    self_discharge_rate: float = 0.001  # per hour

    def __post_init__(self):
        self.soc: float = self.initial_soc
        self.history: List[float] = [self.soc]

    def step(self, net_power_mw: float, dt_hours: float = 1.0) -> float:
        """Advance one time step. Returns clamped SoC.

        net_power_mw > 0 means charging (harvest surplus).
        net_power_mw < 0 means discharging (load exceeds harvest).
        """
        if self.capacity_mwh <= 0:
            return self.soc
        delta_mwh = net_power_mw * dt_hours
        if delta_mwh > 0:
            delta_mwh *= self.efficiency
        else:
            delta_mwh /= max(self.efficiency, 0.01)
        self.soc += delta_mwh / self.capacity_mwh
        self.soc -= self.self_discharge_rate * dt_hours
        self.soc = max(0.0, min(1.0, self.soc))
        self.history.append(self.soc)
        return self.soc

    @property
    def energy_stored_mwh(self) -> float:
        return self.soc * self.capacity_mwh

    @property
    def is_depleted(self) -> bool:
        return self.soc <= 0.0


# ── Thermal Model ────────────────────────────────────────────────────


@dataclass
class ThermalModel:
    """Simple junction-temperature model.

    T_j = T_ambient + P_total * R_theta_ja
    """

    ambient_c: float = 25.0
    r_theta_ja: float = 15.0  # °C/W (junction-to-ambient thermal resistance)
    max_junction_c: float = 85.0

    def junction_temp(self, power_mw: float) -> float:
        return self.ambient_c + (power_mw / 1000.0) * self.r_theta_ja

    def is_safe(self, power_mw: float) -> bool:
        return self.junction_temp(power_mw) <= self.max_junction_c

    def max_power_mw(self) -> float:
        """Maximum power before thermal limit."""
        return (self.max_junction_c - self.ambient_c) / self.r_theta_ja * 1000.0


# ── Sustainability Optimizer ─────────────────────────────────────────


@dataclass
class DutyCycleConfig:
    """Optimised duty-cycle configuration for a deployment."""

    active_fraction: float = 1.0
    bitstream_length_scale: float = 1.0
    pruning_fraction: float = 0.0


@dataclass
class NetZeroReport:
    """Results of a net-zero feasibility analysis."""

    total_power_mw: float
    harvest_power_mw: float
    deficit_mw: float
    carbon_g_per_hour: float
    annual_carbon_kg: float
    net_zero_feasible: bool
    time_to_neutral_hours: float
    optimization: Optional[DutyCycleConfig] = None
    suggestions: List[str] = field(default_factory=list)


class SustainabilityOptimizer:
    """Optimises SC deployment for net-zero energy operation."""

    def __init__(
        self,
        fpga: FPGAResourceReport,
        carbon: Optional[CarbonModel] = None,
        embodied: Optional[EmbodiedCarbon] = None,
        thermal: Optional[ThermalModel] = None,
    ):
        self.fpga = fpga
        self.carbon = carbon or CarbonModel()
        self.embodied = embodied or EmbodiedCarbon()
        self.thermal = thermal or ThermalModel()

    def analyze(
        self,
        harvest: Optional[HarvestProfile] = None,
        target_hours: float = 8760.0,
    ) -> NetZeroReport:
        """Run full sustainability analysis."""
        total_power = self.fpga.total_power_mw
        harvest_power = harvest.average_power_mw if harvest else 0.0
        deficit = max(0.0, total_power - harvest_power)

        carbon_per_hour = self.carbon.compute(deficit, 1.0)
        annual = self.carbon.annual_footprint_kg(deficit)

        feasible = deficit <= 0.0
        ttn = 0.0
        if harvest and harvest_power > 0 and not feasible:
            surplus_needed_mwh = deficit * target_hours / 1000.0
            storage = harvest.storage_capacity_mwh
            if storage > 0:
                ttn = surplus_needed_mwh / storage
            else:
                ttn = float("inf")

        suggestions = self._generate_suggestions(total_power, harvest_power, deficit)

        optimization = None
        if deficit > 0 and harvest:
            optimization = self._optimize_duty_cycle(total_power, harvest_power)

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

    def _optimize_duty_cycle(self, total_power: float, harvest_power: float) -> DutyCycleConfig:
        """Compute optimal duty-cycle to match harvest."""
        if total_power <= 0:
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

    def _generate_suggestions(self, total: float, harvest: float, deficit: float) -> List[str]:
        suggestions = []
        if deficit > 0:
            suggestions.append(
                f"Power deficit of {deficit:.2f} mW — consider reducing toggle rate or clock frequency"
            )
        if total > 100:
            suggestions.append("Total power exceeds 100 mW — evaluate BRAM vs. LUT trade-offs")
        if harvest <= 0:
            suggestions.append(
                "No energy harvesting configured — add a harvest source for net-zero analysis"
            )
        if deficit <= 0:
            suggestions.append("Net-zero operation is feasible with current configuration")
        if not self.thermal.is_safe(total):
            suggestions.append(
                f"Thermal violation: T_j = {self.thermal.junction_temp(total):.1f}°C exceeds {self.thermal.max_junction_c}°C"
            )
        return suggestions

    def hourly_profile(
        self,
        harvest: HarvestProfile,
        hours: int = 24,
    ) -> List[Dict[str, float]]:
        """Generate an hourly power-balance profile."""
        total_power = self.fpga.total_power_mw
        profile = []
        for h in range(hours):
            h_power = harvest.power_at(float(h))
            profile.append(
                {
                    "hour": float(h),
                    "harvest_mw": h_power,
                    "load_mw": total_power,
                    "balance_mw": h_power - total_power,
                    "co2_g": self.carbon.compute(max(0, total_power - h_power), 1.0),
                }
            )
        return profile

    def simulate_storage(
        self,
        harvest: HarvestProfile,
        storage: EnergyStorageSim,
        hours: int = 24,
    ) -> List[Dict[str, float]]:
        """Simulate battery SoC over time with harvest and load."""
        total_power = self.fpga.total_power_mw
        timeline = []
        for h in range(hours):
            h_power = harvest.power_at(float(h))
            net = h_power - total_power
            soc = storage.step(net, dt_hours=1.0)
            timeline.append(
                {
                    "hour": float(h),
                    "harvest_mw": h_power,
                    "load_mw": total_power,
                    "net_mw": net,
                    "soc": soc,
                }
            )
        return timeline

    def energy_efficiency(
        self,
        ops_per_second: float,
    ) -> Dict[str, float]:
        """Compute energy efficiency metrics."""
        total_mw = self.fpga.total_power_mw
        total_w = total_mw / 1000.0
        return {
            "ops_per_joule": ops_per_second / max(total_w, 1e-9),
            "sop_per_mw": ops_per_second / max(total_mw, 1e-9),
            "total_power_mw": total_mw,
        }

    def deployment_lifetime(
        self,
        harvest: Optional[HarvestProfile] = None,
        battery_mwh: float = 100.0,
    ) -> Dict[str, float]:
        """Estimate deployment lifetime and maintenance intervals."""
        total_power = self.fpga.total_power_mw
        harvest_power = harvest.average_power_mw if harvest else 0.0
        deficit = max(0.0, total_power - harvest_power)

        if deficit <= 0:
            battery_life_hours = float("inf")
        elif battery_mwh > 0:
            battery_life_hours = battery_mwh / deficit
        else:
            battery_life_hours = 0.0

        annual_carbon = self.carbon.annual_footprint_kg(deficit)
        total_annual_carbon = annual_carbon + self.embodied.amortised_annual_kg

        return {
            "battery_life_hours": battery_life_hours,
            "battery_life_days": battery_life_hours / 24.0
            if battery_life_hours != float("inf")
            else float("inf"),
            "annual_operational_carbon_kg": annual_carbon,
            "annual_embodied_carbon_kg": self.embodied.amortised_annual_kg,
            "annual_total_carbon_kg": total_annual_carbon,
            "device_lifetime_years": self.embodied.lifetime_years,
        }

    def adaptive_duty_cycle_sim(
        self,
        harvest: HarvestProfile,
        hours: int = 24,
        min_active: float = 0.1,
    ) -> List[Dict[str, float]]:
        """Simulate time-varying adaptive duty cycling.

        Adjusts active fraction per hour based on available harvest power.
        """
        total_power = self.fpga.total_power_mw
        timeline = []
        for h in range(hours):
            h_power = harvest.power_at(float(h))
            if total_power > 0:
                active_frac = min(1.0, max(min_active, h_power / total_power))
            else:
                active_frac = 1.0
            effective_load = total_power * active_frac
            surplus = h_power - effective_load
            timeline.append(
                {
                    "hour": float(h),
                    "harvest_mw": h_power,
                    "active_fraction": active_frac,
                    "effective_load_mw": effective_load,
                    "surplus_mw": surplus,
                }
            )
        return timeline


# ── Multi-Source Analysis Helper ─────────────────────────────────────


def analyze_multi_harvest(
    fpga: FPGAResourceReport,
    stack: MultiHarvestStack,
    carbon: Optional[CarbonModel] = None,
) -> NetZeroReport:
    """Run sustainability analysis with a stacked harvest profile."""
    cm = carbon or CarbonModel()
    total_power = fpga.total_power_mw
    harvest_power = stack.average_power_mw
    deficit = max(0.0, total_power - harvest_power)
    carbon_per_hour = cm.compute(deficit, 1.0)
    annual = cm.annual_footprint_kg(deficit)
    feasible = deficit <= 0.0
    suggestions = []
    if feasible:
        suggestions.append("Net-zero achieved with stacked harvesters")
    else:
        suggestions.append(f"Deficit {deficit:.2f} mW — add more harvest sources")
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
