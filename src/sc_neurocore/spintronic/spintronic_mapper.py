# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spintronic / Magnetic-Domain SC Mapper

"""Maps SC bitstreams to spintronic domain-wall and skyrmion logic.

Bridges deterministic SC arithmetic to magnetic-domain computing:
domain-wall motion, skyrmion nucleation/annihilation, and spin-orbit
torque (SOT) switching. Includes device variability models and
co-simulation hooks for micromagnetic solvers (MuMax3).

Device technologies:
- **Domain-Wall (DW)**: Nanotrack racetrack with SOT-driven motion
- **Skyrmion (SKY)**: Skyrmion nucleation/annihilation gates
- **STT-MTJ**: Spin-transfer torque magnetic tunnel junctions
- **SOT-MRAM**: Spin-orbit torque MRAM cells

Pipeline:
    SC bitstream → SpintronicMapper → DeviceArray → MuMax3 co-sim

Compatible with:
- ``memristor/memristor_mapper.py`` — shares variability injection pattern
- ``hdl_gen/verilog_generator.py`` — RTL target for spintronic arrays
- ``chiplet_gen/`` — spintronic dies as chiplet nodes
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Device Technology ────────────────────────────────────────────────


class SpintronicTech(Enum):
    DOMAIN_WALL = "domain_wall"
    SKYRMION = "skyrmion"
    STT_MTJ = "stt_mtj"
    SOT_MRAM = "sot_mram"


@dataclass
class MaterialParams:
    """Micromagnetic material parameters."""

    saturation_magnetisation_a_m: float  # A/m (Ms)
    exchange_stiffness_j_m: float  # J/m (Aex)
    dmi_strength_j_m2: float  # J/m² (D, Dzyaloshinskii-Moriya)
    perpendicular_anisotropy_j_m3: float  # J/m³ (Ku)
    damping_alpha: float  # Gilbert damping
    temperature_k: float = 300.0

    @classmethod
    def cofeb_mgo(cls) -> MaterialParams:
        """CoFeB/MgO — standard STT-MTJ / SOT-MRAM stack."""
        return cls(
            saturation_magnetisation_a_m=1.2e6,
            exchange_stiffness_j_m=1.5e-11,
            dmi_strength_j_m2=0.0,
            perpendicular_anisotropy_j_m3=8e5,
            damping_alpha=0.01,
        )

    @classmethod
    def pt_co_multilayer(cls) -> MaterialParams:
        """Pt/Co multilayer — skyrmion host."""
        return cls(
            saturation_magnetisation_a_m=5.8e5,
            exchange_stiffness_j_m=1.5e-11,
            dmi_strength_j_m2=3.5e-3,
            perpendicular_anisotropy_j_m3=6e5,
            damping_alpha=0.015,
        )

    @classmethod
    def w_cofeb(cls) -> MaterialParams:
        """W/CoFeB — SOT switching with heavy-metal underlayer."""
        return cls(
            saturation_magnetisation_a_m=1.1e6,
            exchange_stiffness_j_m=1.3e-11,
            dmi_strength_j_m2=0.5e-3,
            perpendicular_anisotropy_j_m3=7e5,
            damping_alpha=0.02,
        )


@dataclass
class SpintronicDeviceConfig:
    """Configuration for a single spintronic device."""

    tech: SpintronicTech = SpintronicTech.SOT_MRAM
    material: MaterialParams = field(default_factory=MaterialParams.cofeb_mgo)
    width_nm: float = 80.0
    length_nm: float = 200.0
    thickness_nm: float = 1.2
    switching_current_ua: float = 50.0
    switching_time_ns: float = 1.0
    write_resistance_ohm: float = 10000.0
    parallel_resistance_ohm: float = 5000.0
    retention_years: float = 10.0
    tmr_ratio: float = 1.5
    error_rate: float = 1e-6

    def __post_init__(self) -> None:
        if self.width_nm <= 0:
            raise ValueError("width_nm must be positive")
        if self.length_nm <= 0:
            raise ValueError("length_nm must be positive")
        if self.thickness_nm <= 0:
            raise ValueError("thickness_nm must be positive")
        if self.switching_current_ua <= 0:
            raise ValueError("switching_current_ua must be positive")
        if self.switching_time_ns <= 0:
            raise ValueError("switching_time_ns must be positive")
        if self.write_resistance_ohm <= 0:
            raise ValueError("write_resistance_ohm must be positive")
        if self.parallel_resistance_ohm <= 0:
            raise ValueError("parallel_resistance_ohm must be positive")
        if self.tmr_ratio < 0:
            raise ValueError("tmr_ratio must be non-negative")

    @classmethod
    def from_tech(cls, tech: SpintronicTech) -> SpintronicDeviceConfig:
        presets: Dict[SpintronicTech, Dict[str, Any]] = {
            SpintronicTech.DOMAIN_WALL: dict(
                material=MaterialParams.pt_co_multilayer(),
                width_nm=60.0,
                length_nm=1000.0,
                thickness_nm=0.8,
                switching_current_ua=100.0,
                switching_time_ns=5.0,
                write_resistance_ohm=12000.0,
                parallel_resistance_ohm=8000.0,
            ),
            SpintronicTech.SKYRMION: dict(
                material=MaterialParams.pt_co_multilayer(),
                width_nm=50.0,
                length_nm=500.0,
                thickness_nm=0.8,
                switching_current_ua=30.0,
                switching_time_ns=2.0,
                write_resistance_ohm=9000.0,
                parallel_resistance_ohm=7000.0,
            ),
            SpintronicTech.STT_MTJ: dict(
                material=MaterialParams.cofeb_mgo(),
                width_nm=40.0,
                length_nm=40.0,
                thickness_nm=1.2,
                switching_current_ua=80.0,
                switching_time_ns=3.0,
                write_resistance_ohm=10000.0,
                parallel_resistance_ohm=5000.0,
            ),
            SpintronicTech.SOT_MRAM: dict(
                material=MaterialParams.w_cofeb(),
                width_nm=80.0,
                length_nm=200.0,
                thickness_nm=1.0,
                switching_current_ua=50.0,
                switching_time_ns=0.5,
                write_resistance_ohm=4000.0,
                parallel_resistance_ohm=3000.0,
            ),
        }
        return cls(tech=tech, **presets[tech])

    @property
    def area_nm2(self) -> float:
        return self.width_nm * self.length_nm

    @property
    def switching_energy_fj(self) -> float:
        """Write-path switching energy, E = I² × R_write × t."""
        i_a = self.switching_current_ua * 1e-6
        return i_a**2 * self.write_resistance_ohm * self.switching_time_ns * 1e6  # fJ

    @property
    def thermal_stability(self) -> float:
        """Thermal stability factor Δ = Ku × V / (kB × T).

        Needs to be > 40 for 10-year retention.
        """
        kb = 1.38064852e-23
        volume_m3 = (self.width_nm * self.length_nm * self.thickness_nm) * 1e-27
        t = self.material.temperature_k
        return self.material.perpendicular_anisotropy_j_m3 * volume_m3 / (kb * t)

    @property
    def read_disturb_probability(self) -> float:
        """Probability of read disturb (accidental flip during read).

        Approximation: P_rd ∝ exp(-Δ) for in-plane STT read.
        """
        delta = self.thermal_stability
        return float(np.exp(-delta)) if delta < 100 else 0.0

    @property
    def endurance_cycles(self) -> int:
        """Estimated write endurance cycles."""
        endurance_map = {
            SpintronicTech.DOMAIN_WALL: 10**15,
            SpintronicTech.SKYRMION: 10**15,
            SpintronicTech.STT_MTJ: 10**12,
            SpintronicTech.SOT_MRAM: 10**15,
        }
        return endurance_map.get(self.tech, 10**12)


# ── Variability Model ───────────────────────────────────────────────


@dataclass
class VariabilityModel:
    """Process variability for spintronic devices.

    Models dimension, anisotropy, and DMI variations from fab data.
    """

    width_sigma_pct: float = 3.0
    length_sigma_pct: float = 3.0
    ku_sigma_pct: float = 5.0
    dmi_sigma_pct: float = 8.0
    damping_sigma_pct: float = 10.0
    ms_sigma_pct: float = 2.0

    def apply(
        self, device: SpintronicDeviceConfig, rng: np.random.Generator
    ) -> SpintronicDeviceConfig:
        """Return a variability-injected copy of the device."""
        import copy

        d = copy.deepcopy(device)
        d.width_nm *= 1 + rng.normal(0, self.width_sigma_pct / 100)
        d.length_nm *= 1 + rng.normal(0, self.length_sigma_pct / 100)
        d.material.perpendicular_anisotropy_j_m3 *= 1 + rng.normal(0, self.ku_sigma_pct / 100)
        d.material.dmi_strength_j_m2 *= 1 + rng.normal(0, self.dmi_sigma_pct / 100)
        d.material.damping_alpha *= 1 + rng.normal(0, self.damping_sigma_pct / 100)
        d.material.saturation_magnetisation_a_m *= 1 + rng.normal(0, self.ms_sigma_pct / 100)
        d.width_nm = max(10.0, d.width_nm)
        d.length_nm = max(10.0, d.length_nm)
        d.material.damping_alpha = max(0.001, d.material.damping_alpha)
        return d


# ── Spintronic Array ─────────────────────────────────────────────────


@dataclass
class SpintronicCell:
    """One cell in a spintronic array."""

    row: int
    col: int
    device: SpintronicDeviceConfig
    state: int = 0  # 0 = P (parallel), 1 = AP (anti-parallel)
    weight_q88: int = 256  # Q8.8 = 1.0

    @property
    def resistance_ohm(self) -> float:
        r_p = self.device.parallel_resistance_ohm
        return r_p * (1 + self.state * self.device.tmr_ratio)


class SpintronicArray:
    """Crossbar array of spintronic devices for SC computation."""

    def __init__(
        self,
        rows: int,
        cols: int,
        tech: SpintronicTech = SpintronicTech.SOT_MRAM,
        variability: Optional[VariabilityModel] = None,
        rng_seed: int = 42,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.tech = tech
        self.rng = np.random.default_rng(rng_seed)
        base_device = SpintronicDeviceConfig.from_tech(tech)
        var = variability or VariabilityModel()
        self.cells: List[List[SpintronicCell]] = []
        for r in range(rows):
            row = []
            for c in range(cols):
                dev = var.apply(base_device, self.rng)
                row.append(SpintronicCell(r, c, dev))
            self.cells.append(row)

    @property
    def total_cells(self) -> int:
        return self.rows * self.cols

    @property
    def total_area_um2(self) -> float:
        return sum(c.device.area_nm2 for row in self.cells for c in row) / 1e6

    def program_weights(self, weights_q88: np.ndarray[Any, Any]) -> None:
        """Program Q8.8 weights into the array."""
        for r in range(min(self.rows, weights_q88.shape[0])):
            for c in range(min(self.cols, weights_q88.shape[1])):
                w = int(weights_q88[r, c])
                self.cells[r][c].weight_q88 = w
                self.cells[r][c].state = 1 if w > 128 else 0

    def read_weights(self) -> np.ndarray[Any, Any]:
        """Read weights back from the array."""
        w = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                w[r, c] = self.cells[r][c].weight_q88
        return w

    def power_breakdown(self, bitstream_length: int = 256) -> Dict[str, float]:
        """Per-array power breakdown in femtojoules."""
        switch_energy = (
            sum(c.device.switching_energy_fj for row in self.cells for c in row) * bitstream_length
        )
        leakage_fj = (
            sum(
                1.0 / c.resistance_ohm * 0.1  # 100 mV read bias, 1 ns
                for row in self.cells
                for c in row
            )
            * bitstream_length
            * 1e6
        )
        return {
            "switching_fj": switch_energy,
            "leakage_fj": leakage_fj,
            "total_fj": switch_energy + leakage_fj,
        }


# ── SC Bitstream → Spintronic Mapper ─────────────────────────────────


@dataclass
class MappingResult:
    """Result of mapping SC bitstreams to a spintronic array."""

    array_rows: int
    array_cols: int
    tech: SpintronicTech
    total_area_um2: float
    total_energy_fj: float
    total_switching_time_ns: float
    bit_error_rate: float
    mc_yield: float = 1.0


class SpintronicMapper:
    """Maps SC bitstreams + weights to a spintronic crossbar.

    Performs:
    1. Weight quantisation to device states (P/AP)
    2. Array sizing from network dimensions
    3. Energy/timing estimation
    4. Monte-Carlo variability yield analysis
    """

    def __init__(
        self,
        tech: SpintronicTech = SpintronicTech.SOT_MRAM,
        variability: Optional[VariabilityModel] = None,
        rng_seed: int = 42,
    ) -> None:
        self.tech = tech
        self.variability = variability or VariabilityModel()
        self.rng = np.random.default_rng(rng_seed)

    def map_network(
        self,
        weights_q88: np.ndarray[Any, Any],
        bitstream_length: int = 256,
    ) -> Tuple[SpintronicArray, MappingResult]:
        """Map a weight matrix to a spintronic array."""
        rows, cols = weights_q88.shape
        array = SpintronicArray(
            rows,
            cols,
            self.tech,
            self.variability,
            int(self.rng.integers(0, 2**31)),
        )
        array.program_weights(weights_q88)

        base = SpintronicDeviceConfig.from_tech(self.tech)
        total_e = base.switching_energy_fj * rows * cols * bitstream_length
        total_t = base.switching_time_ns * bitstream_length
        ber = base.error_rate * rows * cols

        return array, MappingResult(
            rows,
            cols,
            self.tech,
            array.total_area_um2,
            total_e,
            total_t,
            ber,
        )

    def monte_carlo_yield(
        self,
        weights_q88: np.ndarray[Any, Any],
        n_trials: int = 100,
        tolerance_q88: int = 16,
    ) -> float:
        """Run Monte-Carlo yield analysis."""
        passing = 0
        for _ in range(n_trials):
            seed = int(self.rng.integers(0, 2**31))
            array = SpintronicArray(
                weights_q88.shape[0],
                weights_q88.shape[1],
                self.tech,
                self.variability,
                seed,
            )
            array.program_weights(weights_q88)
            readback = array.read_weights()
            max_error = int(
                np.max(np.abs(readback.astype(np.int32) - weights_q88.astype(np.int32)))
            )
            if max_error <= tolerance_q88:
                passing += 1
        return passing / n_trials


# ── MuMax3 Co-Simulation Script Generator ────────────────────────────


class MuMax3ScriptGenerator:
    """Generates MuMax3 (.mx3) scripts for micromagnetic co-simulation."""

    @staticmethod
    def generate_switching(
        device: SpintronicDeviceConfig,
        current_density_a_m2: float = 1e12,
        duration_ns: float = 5.0,
    ) -> str:
        m = device.material
        return f"""\
// SC-NeuroCore MuMax3 Co-Simulation — SOT Switching
// Tech: {device.tech.value}

SetGridSize(128, 32, 1)
SetCellSize({device.width_nm / 128:.2f}e-9, {device.length_nm / 32:.2f}e-9, {device.thickness_nm:.1f}e-9)

Msat  = {m.saturation_magnetisation_a_m:.1f}
Aex   = {m.exchange_stiffness_j_m:.2e}
Ku1   = {m.perpendicular_anisotropy_j_m3:.2e}
AnisU = vector(0, 0, 1)
Dind  = {m.dmi_strength_j_m2:.2e}
alpha = {m.damping_alpha:.4f}

Temp = {m.temperature_k:.0f}

m = uniform(0, 0, 1)

J = vector({current_density_a_m2:.2e}, 0, 0)
xi = 0.05
Pol = 0.56

Run({duration_ns:.1f}e-9)

SaveAs(m, "final_state")
"""

    @staticmethod
    def generate_skyrmion(
        device: SpintronicDeviceConfig,
    ) -> str:
        m = device.material
        return f"""\
// SC-NeuroCore MuMax3 — Skyrmion Nucleation
SetGridSize(256, 256, 1)
SetCellSize({device.width_nm / 256:.2f}e-9, {device.width_nm / 256:.2f}e-9, {device.thickness_nm:.1f}e-9)

Msat  = {m.saturation_magnetisation_a_m:.1f}
Aex   = {m.exchange_stiffness_j_m:.2e}
Ku1   = {m.perpendicular_anisotropy_j_m3:.2e}
AnisU = vector(0, 0, 1)
Dind  = {m.dmi_strength_j_m2:.2e}
alpha = {m.damping_alpha:.4f}

Temp = {m.temperature_k:.0f}

m = uniform(0, 0, 1)
m.SetRegion(0, NeelSkyrmion(1, -1))

Relax()
SaveAs(m, "skyrmion_ground_state")
"""


# ── Verilog Wrapper Generator ────────────────────────────────────────


class SpintronicVerilogGenerator:
    """Generates Verilog wrappers for spintronic crossbar arrays."""

    @staticmethod
    def generate(
        array_name: str,
        rows: int,
        cols: int,
        tech: SpintronicTech,
    ) -> str:
        return f"""\
// SC-NeuroCore — Spintronic Array Wrapper ({tech.value})
// Auto-generated — do not edit

module {array_name} #(
    parameter ROWS = {rows},
    parameter COLS = {cols},
    parameter BITSTREAM_W = 256
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire [BITSTREAM_W-1:0]   bitstream_in  [0:COLS-1],
    output wire [BITSTREAM_W-1:0]   bitstream_out [0:ROWS-1],
    // Weight programming interface
    input  wire [$clog2(ROWS)-1:0]  prog_row,
    input  wire [$clog2(COLS)-1:0]  prog_col,
    input  wire [15:0]              prog_weight,  // Q8.8
    input  wire                     prog_en
);

    // Weight storage (maps to spintronic state via external driver)
    reg [15:0] weights [0:ROWS-1][0:COLS-1];

    // Programming
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // No reset for NVM — retain state
        end else if (prog_en) begin
            weights[prog_row][prog_col] <= prog_weight;
        end
    end

    // SC dot product per output row
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : row_gen
            wire [BITSTREAM_W-1:0] partial [0:COLS-1];
            for (c = 0; c < COLS; c = c + 1) begin : col_gen
                // SC AND gate: bitstream × weight-encoded bitstream
                assign partial[c] = bitstream_in[c]; // Simplified; actual uses LFSR threshold
            end
            // OR-reduction (majority) integration point for partial products
            assign bitstream_out[r] = partial[0]; // downstream mapper expands this tree
        end
    endgenerate

endmodule
"""


# ── Domain-Wall Racetrack Shift Register (Gap 1) ────────────────────


@dataclass
class RacetrackShiftRegister:
    """Domain-wall racetrack memory with current-driven bit shifting.

    Models a nanotrack with N bit positions; SOT current pulses shift
    all domain walls one position per clock edge.
    """

    n_positions: int
    bits: Optional[np.ndarray[Any, Any]] = None
    shift_current_ua: float = 100.0
    shift_time_ns: float = 0.5
    shift_error_rate: float = 1e-5

    def __post_init__(self) -> None:
        if self.bits is None:
            self.bits = np.zeros(self.n_positions, dtype=np.int8)

    def load(self, data: np.ndarray[Any, Any]) -> None:
        self.bits = np.array(data[: self.n_positions], dtype=np.int8)

    def shift_right(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> None:
        for _ in range(n):
            bits = self.bits if self.bits is not None else np.zeros(self.n_positions, dtype=np.int8)
            self.bits = np.roll(bits, 1).astype(np.int8, copy=False)
            self.bits[0] = 0
            if rng is not None and rng.random() < self.shift_error_rate:
                pos = rng.integers(0, self.n_positions)
                self.bits[pos] ^= 1

    def shift_left(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> None:
        for _ in range(n):
            bits = self.bits if self.bits is not None else np.zeros(self.n_positions, dtype=np.int8)
            self.bits = np.roll(bits, -1).astype(np.int8, copy=False)
            self.bits[-1] = 0
            if rng is not None and rng.random() < self.shift_error_rate:
                pos = rng.integers(0, self.n_positions)
                self.bits[pos] ^= 1

    @property
    def shift_energy_fj(self) -> float:
        r_ohm = 500.0
        i_a = self.shift_current_ua * 1e-6
        return i_a**2 * r_ohm * self.shift_time_ns * 1e6


# ── Skyrmion Hall Angle Correction (Gap 2) ──────────────────────────


@dataclass
class SkyrmionHallCorrector:
    """Corrects skyrmion trajectory for the skyrmion Hall effect.

    Skyrmions drift at an angle θ_H to the applied current direction.
    θ_H = arctan(4π Q / (α D)) where Q is topological charge.
    """

    topological_charge: int = 1
    damping_alpha: float = 0.015
    dmi_strength: float = 3.5e-3

    @property
    def hall_angle_deg(self) -> float:
        ratio = 4 * math.pi * abs(self.topological_charge) * self.damping_alpha
        return math.degrees(math.atan(ratio))

    def corrected_position(self, x_drive: float, track_width_nm: float) -> Tuple[float, float]:
        """Return (x, y) position accounting for Hall drift."""
        theta = math.radians(self.hall_angle_deg)
        y_drift = x_drive * math.tan(theta)
        y_clamped = max(-track_width_nm / 2, min(track_width_nm / 2, y_drift))
        return (x_drive, y_clamped)

    @property
    def needs_confinement(self) -> bool:
        return self.hall_angle_deg > 5.0


# ── Temperature-Dependent Switching (Gap 3) ─────────────────────────


def switching_current_vs_temperature(
    i_c0_ua: float,
    delta_0: float,
    temperature_k: float,
    temp_ref_k: float = 300.0,
) -> float:
    """Ic(T) = Ic0 × (1 - T/T_ref × kBT/(Δ0 × kBT_ref)).

    Simplified Néel-Arrhenius model for thermally activated switching.
    """
    if temp_ref_k <= 0 or delta_0 <= 0:
        return i_c0_ua
    ratio = temperature_k / temp_ref_k
    factor = max(0.01, 1.0 - ratio * (1.0 / delta_0))
    return i_c0_ua * factor


def switching_time_vs_temperature(
    t_sw0_ns: float,
    temperature_k: float,
    temp_ref_k: float = 300.0,
) -> float:
    """Switching time increases with temperature due to thermal fluctuations."""
    ratio = temperature_k / temp_ref_k
    return t_sw0_ns * (1.0 + 0.1 * (ratio - 1.0))


# ── Retention Failure Probability (Gap 4) ───────────────────────────


def retention_failure_probability(
    thermal_stability: float,
    time_seconds: float,
    attempt_freq_hz: float = 1e9,
) -> float:
    """P_fail = 1 - exp(-t × f0 × exp(-Δ))."""
    if thermal_stability > 100:
        return 0.0
    exponent = -thermal_stability
    rate = attempt_freq_hz * math.exp(exponent)
    p = 1.0 - math.exp(-time_seconds * rate)
    return max(0.0, min(1.0, p))


# ── Multi-Level Cell (MLC) Support (Gap 5) ──────────────────────────


@dataclass
class MLCConfig:
    """Multi-level cell configuration for spintronic devices."""

    bits_per_cell: int = 2
    levels: int = 0

    def __post_init__(self) -> None:
        self.levels = 2**self.bits_per_cell

    @property
    def resistance_margins(self) -> List[float]:
        """Resistance levels evenly spaced between R_P and R_AP."""
        r_p, r_ap = 5000.0, 12500.0
        step = (r_ap - r_p) / (self.levels - 1) if self.levels > 1 else 0
        return [r_p + i * step for i in range(self.levels)]

    def quantize_weight(self, weight_float: float) -> int:
        """Quantize a [0, 1] weight to an MLC level."""
        level = int(round(weight_float * (self.levels - 1)))
        return max(0, min(self.levels - 1, level))

    def dequantize(self, level: int) -> float:
        return level / (self.levels - 1) if self.levels > 1 else 0.0

    @property
    def density_improvement(self) -> float:
        return float(self.bits_per_cell)


# ── Write-Verify Loop (Gap 6) ───────────────────────────────────────


@dataclass
class WriteVerifyResult:
    """Result of a write-verify cycle."""

    target_weight: int
    actual_weight: int
    attempts: int
    success: bool

    @property
    def error(self) -> int:
        return abs(self.target_weight - self.actual_weight)


def write_verify(
    cell: SpintronicCell,
    target_q88: int,
    max_attempts: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> WriteVerifyResult:
    """Program a cell with write-verify loop."""
    for attempt in range(1, max_attempts + 1):
        cell.weight_q88 = target_q88
        cell.state = 1 if target_q88 > 128 else 0
        if rng is not None:
            noise = int(rng.normal(0, 2))
            cell.weight_q88 = max(0, min(511, cell.weight_q88 + noise))
        if abs(cell.weight_q88 - target_q88) <= 4:
            return WriteVerifyResult(target_q88, cell.weight_q88, attempt, True)
    return WriteVerifyResult(target_q88, cell.weight_q88, max_attempts, False)


# ── Aging / Wear-Out Degradation (Gap 7) ────────────────────────────


@dataclass
class AgingModel:
    """Endurance degradation model for spintronic devices.

    TMR and thermal stability degrade with write cycles.
    """

    cycles_written: int = 0

    def tmr_degradation(self, initial_tmr: float, endurance_limit: int) -> float:
        """TMR degrades linearly as cycling approaches endurance limit."""
        if endurance_limit <= 0:
            return initial_tmr
        frac = min(1.0, self.cycles_written / endurance_limit)
        return initial_tmr * (1.0 - 0.3 * frac)  # up to 30% TMR loss

    def stability_degradation(self, initial_delta: float, endurance_limit: int) -> float:
        """Thermal stability degrades with oxide breakdown."""
        if endurance_limit <= 0:
            return initial_delta
        frac = min(1.0, self.cycles_written / endurance_limit)
        return initial_delta * (1.0 - 0.2 * frac)

    @property
    def is_worn_out(self) -> bool:
        return self.cycles_written > 0 and self.tmr_degradation(1.5, 10**12) < 0.5

    def write(self, n: int = 1) -> None:
        self.cycles_written += n


# ── Radiation Hardness Model (Gap 8) ────────────────────────────────


@dataclass
class RadiationModel:
    """SEU and TID models for spintronic devices in radiation environments."""

    seu_cross_section_cm2: float = 1e-14  # per device
    tid_threshold_krad: float = 1000.0

    def seu_rate(self, flux_particles_cm2_s: float, n_devices: int) -> float:
        """SEU events per second."""
        return self.seu_cross_section_cm2 * flux_particles_cm2_s * n_devices

    def tid_degradation(self, dose_krad: float) -> float:
        """Fraction of performance remaining after TID."""
        if dose_krad >= self.tid_threshold_krad:
            return 0.5  # 50% degradation at threshold
        return 1.0 - 0.5 * (dose_krad / self.tid_threshold_krad)

    @property
    def is_rad_hard(self) -> bool:
        """Spintronic devices are inherently radiation-hard (non-charge-based)."""
        return self.tid_threshold_krad >= 100.0


# ── Array Defect Map / Repair (Gap 9) ───────────────────────────────


@dataclass
class DefectEntry:
    """One defective cell in the array."""

    row: int
    col: int
    defect_type: str  # "stuck_p", "stuck_ap", "open", "short"


class DefectMap:
    """Tracks and remaps defective cells in a spintronic array."""

    def __init__(self) -> None:
        self.defects: List[DefectEntry] = []
        self.remap: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def add_defect(self, row: int, col: int, defect_type: str) -> None:
        self.defects.append(DefectEntry(row, col, defect_type))

    @property
    def defect_count(self) -> int:
        return len(self.defects)

    def defect_rate(self, total_cells: int) -> float:
        if total_cells <= 0:
            return 0.0
        return self.defect_count / total_cells

    def add_remap(self, bad: Tuple[int, int], spare: Tuple[int, int]) -> None:
        self.remap[bad] = spare

    def is_defective(self, row: int, col: int) -> bool:
        return any(d.row == row and d.col == col for d in self.defects)

    def effective_address(self, row: int, col: int) -> Tuple[int, int]:
        return self.remap.get((row, col), (row, col))


# ── MuMax3 Output Parser (Gap 10) ───────────────────────────────────


@dataclass
class MuMax3Result:
    """Parsed result from a MuMax3 simulation."""

    final_mx: float = 0.0
    final_my: float = 0.0
    final_mz: float = 0.0
    switched: bool = False
    energy_j: float = 0.0
    sim_time_ns: float = 0.0

    @property
    def magnetisation_magnitude(self) -> float:
        return math.sqrt(self.final_mx**2 + self.final_my**2 + self.final_mz**2)


class MuMax3OutputParser:
    """Parses MuMax3 table output for co-simulation integration."""

    @staticmethod
    def parse_table(text: str) -> MuMax3Result:
        """Parse a MuMax3 .table output (TSV with header)."""
        lines = [l.strip() for l in text.strip().split("\n") if l.strip() and not l.startswith("#")]
        if not lines:
            return MuMax3Result()
        last = lines[-1].split("\t")
        if len(last) < 4:
            last = lines[-1].split()
        try:
            t = float(last[0])
            mx = float(last[1])
            my = float(last[2])
            mz = float(last[3])
            switched = mz < 0  # switched if mz flipped
            return MuMax3Result(mx, my, mz, switched, sim_time_ns=t * 1e9)
        except (ValueError, IndexError):
            return MuMax3Result()

    @staticmethod
    def is_switching_successful(result: MuMax3Result) -> bool:
        return result.switched and result.magnetisation_magnitude > 0.9
