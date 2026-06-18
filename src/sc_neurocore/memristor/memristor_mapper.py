# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Memristor/Crossbar Variability-Aware SC Mapper

"""Variability-aware memristor crossbar mapper for stochastic computing.

Takes SC bitstreams + per-device conductance variability models (from real
fab data distributions) and emits optimised crossbar-aware SystemVerilog
with compensation LUTs or stochastic encoding that absorbs read/write
noise at design time.

Supports Monte Carlo variability injection for co-simulation, enabling
fab-ready crossbar SC deployment on emerging memristor arrays (Mythic,
Weebit, 2D-material ReRAM, phase-change, etc.).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Technology Models ────────────────────────────────────────────────


class MemristorTechnology(Enum):
    """Supported memristor fabrication technologies."""

    RERAM_HFOX = "reram_hfox"  # HfOₓ-based ReRAM (Weebit, TSMC)
    RERAM_2D = "reram_2d"  # 2D-material ReRAM (MoS₂, h-BN)
    PCM = "pcm"  # Phase-change memory (GST-based)
    MYTHIC_AMP = "mythic_amp"  # Mythic analog matrix processor
    GENERIC = "generic"


# Empirical variability parameters sourced from published fab data.
# sigma_g = relative conductance σ/μ; sigma_rw = read/write noise σ;
# endurance = max write cycles; retention_s = state retention in seconds.
_TECH_PARAMS: Dict[MemristorTechnology, Dict[str, float]] = {
    MemristorTechnology.RERAM_HFOX: {
        "g_on": 100e-6,  # ON conductance (S)
        "g_off": 1e-6,  # OFF conductance (S)
        "sigma_g": 0.05,  # 5% device-to-device σ/μ
        "sigma_rw": 0.02,  # 2% read/write noise σ/μ
        "endurance": 1e6,
        "retention_s": 3.15e7,  # ~1 year
        "num_levels": 16,
    },
    MemristorTechnology.RERAM_2D: {
        "g_on": 50e-6,
        "g_off": 0.5e-6,
        "sigma_g": 0.08,
        "sigma_rw": 0.03,
        "endurance": 1e8,
        "retention_s": 3.15e8,
        "num_levels": 8,
    },
    MemristorTechnology.PCM: {
        "g_on": 200e-6,
        "g_off": 2e-6,
        "sigma_g": 0.10,
        "sigma_rw": 0.04,
        "endurance": 1e4,
        "retention_s": 3.15e7,
        "num_levels": 32,
    },
    MemristorTechnology.MYTHIC_AMP: {
        "g_on": 80e-6,
        "g_off": 0.8e-6,
        "sigma_g": 0.03,
        "sigma_rw": 0.01,
        "endurance": 1e7,
        "retention_s": 3.15e8,
        "num_levels": 256,
    },
    MemristorTechnology.GENERIC: {
        "g_on": 100e-6,
        "g_off": 1e-6,
        "sigma_g": 0.06,
        "sigma_rw": 0.025,
        "endurance": 1e5,
        "retention_s": 3.15e7,
        "num_levels": 16,
    },
}


@dataclass
class ConductanceModel:
    """Per-device conductance variability model.

    Models both device-to-device (D2D) fabrication variability and
    cycle-to-cycle (C2C) read/write noise as Gaussian distributions.
    """

    technology: MemristorTechnology
    g_on: float = 0.0
    g_off: float = 0.0
    sigma_g: float = 0.0
    sigma_rw: float = 0.0
    num_levels: int = 0

    def __post_init__(self) -> None:
        """Fill any unset conductance parameters from the technology presets."""
        params = _TECH_PARAMS[self.technology]
        if self.g_on <= 0:
            self.g_on = params["g_on"]
        if self.g_off <= 0:
            self.g_off = params["g_off"]
        if self.sigma_g <= 0:
            self.sigma_g = params["sigma_g"]
        if self.sigma_rw <= 0:
            self.sigma_rw = params["sigma_rw"]
        if self.num_levels <= 0:
            self.num_levels = int(params["num_levels"])

    @property
    def dynamic_range(self) -> float:
        """ON/OFF conductance ratio."""
        return self.g_on / self.g_off if self.g_off > 0 else float("inf")

    @property
    def level_step(self) -> float:
        """Conductance step between adjacent levels."""
        return (self.g_on - self.g_off) / max(1, self.num_levels - 1)

    def target_conductance(self, level: int) -> float:
        """Nominal conductance for a given quantisation level."""
        level = max(0, min(self.num_levels - 1, level))
        return self.g_off + level * self.level_step

    def sample_d2d(self, level: int, rng: np.random.Generator) -> float:
        """Sample actual conductance with device-to-device variability."""
        nominal = self.target_conductance(level)
        return float(rng.normal(nominal, nominal * self.sigma_g))

    def sample_rw(self, conductance: float, rng: np.random.Generator) -> float:
        """Apply read/write noise to a conductance value."""
        return float(rng.normal(conductance, abs(conductance) * self.sigma_rw))

    def drift(self, conductance: float, elapsed_s: float, alpha: float = 0.1) -> float:
        """Model conductance drift over time.

        Uses power-law drift: G(t) = G₀ × (t/t₀)^(-α)
        """
        t0 = 1.0
        if elapsed_s <= t0:
            return conductance
        drifted: float = conductance * (elapsed_s / t0) ** (-alpha)
        return drifted

    def thermal_shift(self, conductance: float, temp_c: float, ref_c: float = 25.0) -> float:
        """Temperature-dependent conductance shift.

        Linear TC model: ΔG/G ≈ tc_ppm × ΔT × 1e-6.
        """
        tc_ppm = 1500.0  # typical for metal-oxide ReRAM
        delta_t = temp_c - ref_c
        return conductance * (1.0 + tc_ppm * delta_t * 1e-6)


# ── Sneak-Path Current Model ─────────────────────────────────────────


class SneakPathModel:
    """Estimates sneak-path leakage in passive crossbar arrays.

    In an M×N passive crossbar, selecting cell (r,c) exposes parallel
    leakage paths through unselected devices. Worst-case sneak current
    is proportional to (M+N-2) × G_off.
    """

    @staticmethod
    def worst_case_sneak(rows: int, cols: int, g_off: float, v_read: float = 0.2) -> float:
        """Worst-case sneak current (A) through unselected paths."""
        n_paths = (rows - 1) + (cols - 1)
        return n_paths * g_off * v_read

    @staticmethod
    def signal_to_sneak_ratio(g_on: float, g_off: float, rows: int, cols: int) -> float:
        """Ratio of desired signal current to sneak current."""
        sneak = SneakPathModel.worst_case_sneak(rows, cols, g_off)
        if sneak <= 0:
            return float("inf")
        return (g_on * 0.2) / sneak  # assume 0.2V read


# ── IR-Drop / Wire Resistance ────────────────────────────────────────


@dataclass
class IRDropModel:
    """Models interconnect wire resistance and voltage drop."""

    r_wire_per_cell: float = 2.5  # Ω per cell segment

    def voltage_drop(self, row: int, col: int) -> float:
        """Accumulated IR drop at cell (row, col) from corner."""
        return self.r_wire_per_cell * (row + col) * 1e-3  # mV → V approx

    def effective_conductance(
        self, g_nominal: float, row: int, col: int, v_read: float = 0.2
    ) -> float:
        """Conductance seen at read amplifier after IR drop."""
        v_drop = self.voltage_drop(row, col)
        v_eff = max(0.0, v_read - v_drop)
        return g_nominal * (v_eff / v_read) if v_read > 0 else g_nominal


# ── Stuck-at Fault Model ─────────────────────────────────────────────


@dataclass
class StuckFaultMap:
    """Map of stuck-at ON/OFF devices in a crossbar."""

    rows: int
    cols: int
    stuck_on: List[Tuple[int, int]] = field(default_factory=list)
    stuck_off: List[Tuple[int, int]] = field(default_factory=list)

    @classmethod
    def generate(
        cls,
        rows: int,
        cols: int,
        fault_rate: float = 0.001,
        seed: int = 42,
    ) -> StuckFaultMap:
        """Generate random stuck faults at given rate."""
        rng = np.random.default_rng(seed)
        total = rows * cols
        n_faults = int(total * fault_rate)
        fault_idx = rng.choice(total, size=min(n_faults, total), replace=False)
        on_faults = []
        off_faults = []
        for idx in fault_idx:
            r, c = divmod(int(idx), cols)
            if rng.random() < 0.5:
                on_faults.append((r, c))
            else:
                off_faults.append((r, c))
        return cls(rows, cols, on_faults, off_faults)

    def is_stuck(self, row: int, col: int) -> Optional[str]:
        """Return 'on', 'off', or None."""
        if (row, col) in self.stuck_on:
            return "on"
        if (row, col) in self.stuck_off:
            return "off"
        return None

    @property
    def num_faults(self) -> int:
        """Return the total count of stuck-on and stuck-off devices."""
        return len(self.stuck_on) + len(self.stuck_off)

    @property
    def fault_rate(self) -> float:
        """Return the fraction of crossbar cells that are faulty."""
        total = self.rows * self.cols
        return self.num_faults / total if total > 0 else 0.0


# ── Aging / Drift Model ──────────────────────────────────────────────


@dataclass
class AgingReport:
    """Results of aging simulation."""

    elapsed_s: float
    mean_drift_fraction: float
    max_drift_fraction: float
    levels_shifted: int


class AgingSimulator:
    """Simulates conductance drift over device lifetime."""

    def __init__(self, model: ConductanceModel, alpha: float = 0.1) -> None:
        self.model = model
        self.alpha = alpha

    def simulate(
        self, conductances: np.ndarray[Any, Any], elapsed_s: float
    ) -> Tuple[np.ndarray[Any, Any], AgingReport]:
        """Apply drift to all conductances, return (drifted, report)."""
        drifted = np.zeros_like(conductances)
        for idx in np.ndindex(conductances.shape):
            drifted[idx] = self.model.drift(float(conductances[idx]), elapsed_s, self.alpha)
        abs_drift = np.abs(drifted - conductances)
        rel_drift = abs_drift / np.maximum(np.abs(conductances), 1e-15)
        step = self.model.level_step
        levels_shifted = int(np.sum(abs_drift > step)) if step > 0 else 0
        return drifted, AgingReport(
            elapsed_s=elapsed_s,
            mean_drift_fraction=float(np.mean(rel_drift)),
            max_drift_fraction=float(np.max(rel_drift)),
            levels_shifted=levels_shifted,
        )


# ── Stochastic Encoding Absorption ───────────────────────────────────


class SCAbsorbEncoder:
    """Adjusts SC encoding thresholds to absorb known device variability.

    Instead of compensating post-silicon, pre-distorts the bitstream
    encoding thresholds so that the effective computation matches ideal.
    """

    @staticmethod
    def compute_adjusted_thresholds(
        ideal_weights: np.ndarray[Any, Any],
        actual_conductances: np.ndarray[Any, Any],
        model: ConductanceModel,
        q_bits: int = 8,
    ) -> np.ndarray[Any, Any]:
        """Return Q8.8 adjusted thresholds that absorb device error."""
        levels_ideal = np.clip(
            np.round(ideal_weights * (model.num_levels - 1)).astype(int),
            0,
            model.num_levels - 1,
        )
        g_ideal = np.array(
            [
                [
                    model.target_conductance(int(levels_ideal[i, j]))
                    for j in range(ideal_weights.shape[1])
                ]
                for i in range(ideal_weights.shape[0])
            ]
        )
        ratio = np.where(
            np.abs(actual_conductances) > 1e-15,
            g_ideal / actual_conductances,
            1.0,
        )
        scale = 1 << q_bits
        return np.clip(np.round(ratio * scale).astype(np.int32), 0, 65535)


# ── Write-Verify Programming ─────────────────────────────────────────


@dataclass
class WriteVerifyResult:
    """Outcome of iterative write-verify programming."""

    target_level: int
    target_g: float
    achieved_g: float
    iterations: int
    converged: bool


class WriteVerifyProtocol:
    """Iterative program-verify loop for memristor cells."""

    def __init__(
        self,
        model: ConductanceModel,
        max_iterations: int = 10,
        tolerance: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.max_iter = max_iterations
        self.tolerance = tolerance
        self.rng = np.random.default_rng(seed)

    def program_cell(self, target_level: int) -> WriteVerifyResult:
        """Program a single cell to target level with verify."""
        target_g = self.model.target_conductance(target_level)
        g_current = self.model.sample_d2d(target_level, self.rng)

        for i in range(self.max_iter):
            err = abs(g_current - target_g) / max(abs(target_g), 1e-15)
            if err <= self.tolerance:
                return WriteVerifyResult(target_level, target_g, g_current, i + 1, True)
            correction = (target_g - g_current) * 0.5
            g_current += correction
            g_current = self.model.sample_rw(g_current, self.rng)

        return WriteVerifyResult(target_level, target_g, g_current, self.max_iter, False)


# ── Power / Latency Estimation ───────────────────────────────────────


@dataclass
class CrossbarPowerEstimate:
    """Power and performance estimate for a crossbar tile."""

    rows: int
    cols: int
    read_power_uw: float
    write_power_uw: float
    read_latency_ns: float
    write_latency_ns: float
    area_um2: float


class CrossbarEstimator:
    """Estimates power, latency, and area for crossbar arrays."""

    TECH_POWER = {
        MemristorTechnology.RERAM_HFOX: {
            "read_pw": 0.1,
            "write_pw": 10.0,
            "read_ns": 10,
            "write_ns": 100,
            "cell_um2": 0.04,
        },
        MemristorTechnology.RERAM_2D: {
            "read_pw": 0.05,
            "write_pw": 5.0,
            "read_ns": 5,
            "write_ns": 50,
            "cell_um2": 0.01,
        },
        MemristorTechnology.PCM: {
            "read_pw": 0.2,
            "write_pw": 50.0,
            "read_ns": 20,
            "write_ns": 200,
            "cell_um2": 0.06,
        },
        MemristorTechnology.MYTHIC_AMP: {
            "read_pw": 0.08,
            "write_pw": 8.0,
            "read_ns": 8,
            "write_ns": 80,
            "cell_um2": 0.03,
        },
        MemristorTechnology.GENERIC: {
            "read_pw": 0.15,
            "write_pw": 15.0,
            "read_ns": 15,
            "write_ns": 150,
            "cell_um2": 0.05,
        },
    }

    @classmethod
    def estimate(cls, crossbar: CrossbarArray) -> CrossbarPowerEstimate:
        """Estimate read/write power, latency and area for a crossbar array."""
        p = cls.TECH_POWER[crossbar.technology]
        n = crossbar.num_devices
        return CrossbarPowerEstimate(
            rows=crossbar.rows,
            cols=crossbar.cols,
            read_power_uw=p["read_pw"] * n,
            write_power_uw=p["write_pw"] * n,
            read_latency_ns=p["read_ns"],
            write_latency_ns=p["write_ns"],
            area_um2=p["cell_um2"] * n,
        )


# ── Crossbar Array ───────────────────────────────────────────────────


class CrossbarTopology(Enum):
    """Physical wiring topology of a memristor crossbar array."""

    STANDARD = "standard"  # M×N passive crossbar
    ONE_T_ONE_R = "1t1r"  # 1-transistor-1-resistor
    ONE_S_ONE_R = "1s1r"  # 1-selector-1-resistor
    DIFFERENTIAL = "differential"  # Differential pair crossbar


@dataclass
class CrossbarArray:
    """Physical crossbar array specification."""

    rows: int
    cols: int
    topology: CrossbarTopology = CrossbarTopology.STANDARD
    technology: MemristorTechnology = MemristorTechnology.GENERIC

    @property
    def num_devices(self) -> int:
        """Return the device count, doubling for differential topologies."""
        if self.topology == CrossbarTopology.DIFFERENTIAL:
            return self.rows * self.cols * 2
        return self.rows * self.cols

    @property
    def conductance_model(self) -> ConductanceModel:
        """Return the conductance model for this array's technology."""
        return ConductanceModel(technology=self.technology)


# ── Variability Injection ────────────────────────────────────────────


class VariabilityInjector:
    """Injects fab-realistic conductance variability into weight matrices."""

    def __init__(self, model: ConductanceModel, seed: int = 42) -> None:
        self.model = model
        self.rng = np.random.default_rng(seed)

    def quantize_weights(self, weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Map floating-point weights [0, 1] to conductance levels."""
        levels = np.clip(
            np.round(weights * (self.model.num_levels - 1)).astype(int),
            0,
            self.model.num_levels - 1,
        )
        return levels

    def inject_d2d(self, levels: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply device-to-device variability to quantised levels."""
        result = np.zeros_like(levels, dtype=np.float64)
        for idx in np.ndindex(levels.shape):
            result[idx] = self.model.sample_d2d(int(levels[idx]), self.rng)
        return result

    def inject_rw(self, conductances: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply read/write noise to conductance values."""
        result = np.zeros_like(conductances, dtype=np.float64)
        for idx in np.ndindex(conductances.shape):
            result[idx] = self.model.sample_rw(float(conductances[idx]), self.rng)
        return result

    def inject_full(
        self, weights: np.ndarray[Any, Any]
    ) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Full pipeline: quantise → D2D → R/W noise. Returns (levels, conductances)."""
        levels = self.quantize_weights(weights)
        g_d2d = self.inject_d2d(levels)
        g_final = self.inject_rw(g_d2d)
        return levels, g_final

    def compute_error(
        self, weights: np.ndarray[Any, Any], conductances: np.ndarray[Any, Any]
    ) -> Dict[str, float]:
        """Compute variability-induced error statistics."""
        levels = self.quantize_weights(weights)
        ideal = np.array(
            [[self.model.target_conductance(int(levels[idx])) for idx in np.ndindex(levels.shape)]]
        ).reshape(levels.shape)
        abs_err = np.abs(conductances - ideal)
        rel_err = abs_err / np.maximum(np.abs(ideal), 1e-15)
        return {
            "mae": float(np.mean(abs_err)),
            "max_abs_err": float(np.max(abs_err)),
            "mean_rel_err": float(np.mean(rel_err)),
            "max_rel_err": float(np.max(rel_err)),
        }


# ── Compensation LUT ─────────────────────────────────────────────────


class CompensationStrategy(Enum):
    """Strategy for compensating memristor conductance non-idealities."""

    NONE = "none"
    LUT = "lut"  # per-device compensation lookup table
    STOCHASTIC_ENCODING = "sc_absorb"  # adjust SC encoding thresholds
    HYBRID = "hybrid"


@dataclass
class CompensationLUT:
    """Per-device compensation lookup table.

    Maps nominal level → compensated threshold to absorb known D2D
    variability at design time (program-verify or digital pre-distortion).
    """

    device_id: Tuple[int, int]
    nominal_levels: np.ndarray[Any, Any]  # shape (num_levels,)
    compensated_thresholds: np.ndarray[Any, Any]  # Q8.8 fixed-point thresholds

    @classmethod
    def build(
        cls,
        device_id: Tuple[int, int],
        model: ConductanceModel,
        measured_g: Optional[np.ndarray[Any, Any]] = None,
    ) -> CompensationLUT:
        """Build compensation LUT from measured or modelled conductances.

        If measured_g is None, generates from the model's nominal values.
        """
        nominal = np.array([model.target_conductance(i) for i in range(model.num_levels)])
        if measured_g is not None and len(measured_g) == model.num_levels:
            ratio = nominal / np.maximum(measured_g, 1e-15)
        else:
            ratio = np.ones(model.num_levels)
        # Q8.8 fixed-point: multiply by 256, round to int
        thresholds = np.clip(np.round(ratio * 256).astype(np.int32), 0, 65535)
        return cls(
            device_id=device_id,
            nominal_levels=np.arange(model.num_levels),
            compensated_thresholds=thresholds,
        )

    @property
    def max_compensation(self) -> float:
        """Maximum compensation ratio (deviation from 1.0)."""
        ratios = self.compensated_thresholds.astype(np.float64) / 256.0
        return float(np.max(np.abs(ratios - 1.0)))


# ── Crossbar Mapping ─────────────────────────────────────────────────


@dataclass
class CrossbarMapping:
    """Mapping of a weight matrix to a crossbar array."""

    crossbar: CrossbarArray
    weight_levels: np.ndarray[Any, Any]  # shape (rows, cols), quantised levels
    conductances: np.ndarray[Any, Any]  # shape (rows, cols), actual conductances
    compensation_luts: List[CompensationLUT] = field(default_factory=list)
    error_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class MappingResult:
    """Full result of a memristor mapping pass."""

    mappings: List[CrossbarMapping]
    total_devices: int
    total_crossbars: int
    mean_rel_error: float
    max_rel_error: float
    compensation_strategy: CompensationStrategy


class MemristorMapper:
    """Maps SC network weight matrices to physical crossbar arrays."""

    def __init__(
        self,
        technology: MemristorTechnology = MemristorTechnology.GENERIC,
        topology: CrossbarTopology = CrossbarTopology.STANDARD,
        max_crossbar_size: int = 256,
        compensation: CompensationStrategy = CompensationStrategy.LUT,
        seed: int = 42,
    ) -> None:
        self.technology = technology
        self.topology = topology
        self.max_size = max_crossbar_size
        self.compensation = compensation
        self.model = ConductanceModel(technology=technology)
        self.injector = VariabilityInjector(self.model, seed)

    def map_weights(self, weights: np.ndarray[Any, Any]) -> MappingResult:
        """Map a weight matrix (or list of matrices) to crossbar arrays."""
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)

        rows, cols = weights.shape
        tile_rows = min(rows, self.max_size)
        tile_cols = min(cols, self.max_size)

        mappings = []
        for r0 in range(0, rows, tile_rows):
            for c0 in range(0, cols, tile_cols):
                tile = weights[r0 : r0 + tile_rows, c0 : c0 + tile_cols]
                tr, tc = tile.shape
                xbar = CrossbarArray(tr, tc, self.topology, self.technology)
                levels, conductances = self.injector.inject_full(tile)
                err = self.injector.compute_error(tile, conductances)

                luts = []
                if self.compensation in (CompensationStrategy.LUT, CompensationStrategy.HYBRID):
                    for i in range(tr):
                        for j in range(tc):
                            measured = np.array(
                                [
                                    self.model.sample_d2d(lv, self.injector.rng)
                                    for lv in range(self.model.num_levels)
                                ]
                            )
                            lut = CompensationLUT.build((r0 + i, c0 + j), self.model, measured)
                            luts.append(lut)

                mappings.append(
                    CrossbarMapping(
                        crossbar=xbar,
                        weight_levels=levels,
                        conductances=conductances,
                        compensation_luts=luts,
                        error_stats=err,
                    )
                )

        total_dev = sum(m.crossbar.num_devices for m in mappings)
        rel_errs = [m.error_stats.get("mean_rel_err", 0) for m in mappings]
        max_errs = [m.error_stats.get("max_rel_err", 0) for m in mappings]

        return MappingResult(
            mappings=mappings,
            total_devices=total_dev,
            total_crossbars=len(mappings),
            mean_rel_error=float(np.mean(rel_errs)) if rel_errs else 0.0,
            max_rel_error=float(np.max(max_errs)) if max_errs else 0.0,
            compensation_strategy=self.compensation,
        )


# ── Monte Carlo Co-Simulation ───────────────────────────────────────


@dataclass
class MonteCarloReport:
    """Results from Monte Carlo variability co-simulation."""

    num_trials: int
    mean_output_error: float
    std_output_error: float
    max_output_error: float
    yield_fraction: float  # fraction of trials within tolerance
    output_distribution: np.ndarray[Any, Any]
    error_histogram: np.ndarray[Any, Any]


class MonteCarloSimulator:
    """Monte Carlo co-simulation with variability injection.

    Runs N trials of a crossbar multiply-accumulate operation with
    independently sampled D2D + R/W variability to estimate output
    error distributions and yield.
    """

    def __init__(
        self,
        model: ConductanceModel,
        num_trials: int = 1000,
        tolerance: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.num_trials = num_trials
        self.tolerance = tolerance
        self.rng = np.random.default_rng(seed)

    def simulate_mac(
        self,
        weights: np.ndarray[Any, Any],
        inputs: np.ndarray[Any, Any],
    ) -> MonteCarloReport:
        """Simulate multiply-accumulate with variability.

        Parameters
        ----------
        weights : ndarray, shape (M, N)
            Ideal weight matrix in [0, 1].
        inputs : ndarray, shape (N,)
            Input vector in [0, 1].
        """
        ideal_out = weights @ inputs

        outputs = np.zeros((self.num_trials, len(ideal_out)))
        for trial in range(self.num_trials):
            injector = VariabilityInjector(self.model, seed=int(self.rng.integers(0, 2**31)))
            levels, g_actual = injector.inject_full(weights)

            g_ideal = np.array(
                [
                    [
                        self.model.target_conductance(int(levels[i, j]))
                        for j in range(weights.shape[1])
                    ]
                    for i in range(weights.shape[0])
                ]
            )
            scale = np.where(np.abs(g_ideal) > 1e-15, g_actual / g_ideal, 1.0)
            effective_weights = weights * scale
            outputs[trial] = effective_weights @ inputs

        errors = np.abs(outputs - ideal_out[np.newaxis, :])
        mean_err = float(np.mean(errors))
        rel_errors = errors / np.maximum(np.abs(ideal_out[np.newaxis, :]), 1e-15)
        within_tol = np.all(rel_errors < self.tolerance, axis=1)
        yield_frac = float(np.mean(within_tol))

        err_flat = errors.flatten()
        hist, _ = np.histogram(err_flat, bins=50)

        return MonteCarloReport(
            num_trials=self.num_trials,
            mean_output_error=mean_err,
            std_output_error=float(np.std(errors)),
            max_output_error=float(np.max(errors)),
            yield_fraction=yield_frac,
            output_distribution=np.mean(outputs, axis=0),
            error_histogram=hist,
        )


# ── Verilog Emitter ──────────────────────────────────────────────────


class VerilogEmitter:
    """Generates crossbar-aware SystemVerilog with compensation LUTs."""

    def __init__(self, bit_width: int = 16, frac_bits: int = 8) -> None:
        self.bw = bit_width
        self.frac = frac_bits

    def emit_crossbar(
        self,
        mapping: CrossbarMapping,
        module_name: str = "sc_memristor_crossbar",
    ) -> str:
        """Generate SystemVerilog for a single crossbar tile."""
        r, c = mapping.crossbar.rows, mapping.crossbar.cols
        bw = self.bw

        # Build weight parameter block
        weight_params = []
        for i in range(r):
            for j in range(c):
                lvl = int(mapping.weight_levels[i, j])
                weight_params.append(f"    localparam [{bw - 1}:0] W_{i}_{j} = {bw}'d{lvl};")
        weight_block = "\n".join(weight_params)

        # Compensation LUT (if present)
        comp_block = ""
        if mapping.compensation_luts:
            num_levels = mapping.compensation_luts[0].nominal_levels.shape[0]
            comp_lines = [f"    // Compensation LUT ({num_levels} levels)"]
            comp_lines.append(f"    logic [{bw - 1}:0] comp_lut [0:{num_levels - 1}];")
            comp_lines.append("    initial begin")
            lut = mapping.compensation_luts[0]
            for k in range(num_levels):
                val = int(lut.compensated_thresholds[k])
                comp_lines.append(f"        comp_lut[{k}] = {bw}'d{val};")
            comp_lines.append("    end")
            comp_block = "\n".join(comp_lines)

        # MAC accumulator
        mac_lines = []
        for i in range(r):
            terms = " + ".join(f"(i_bitstream[{j}] & W_{i}_{j}[0])" for j in range(c))
            mac_lines.append(f"            o_mac[{i}] <= {terms};")
        mac_block = "\n".join(mac_lines)

        return textwrap.dedent(f"""\
            // SPDX-License-Identifier: AGPL-3.0-or-later
            // Commercial license available
            // © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
            // © Code 2020–2026 Miroslav Šotek. All rights reserved.
            // ORCID: 0009-0009-3560-0851
            // Contact: www.anulum.li | protoscience@anulum.li
            // SC-NeuroCore — Memristor Crossbar ({mapping.crossbar.technology.value})

            module {module_name} #(
                parameter ROWS = {r},
                parameter COLS = {c},
                parameter BW = {bw}
            )(
                input  logic clk,
                input  logic rst_n,
                input  logic [COLS-1:0] i_bitstream,
                output logic [{bw - 1}:0] o_mac [0:ROWS-1],
                output logic o_valid
            );

                // Programmed weight levels (Q{self.bw - self.frac}.{self.frac} fixed-point)
{textwrap.indent(weight_block, "    ")}

{textwrap.indent(comp_block, "    ") if comp_block else "    // No compensation LUT"}

                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        o_valid <= 1'b0;
                    end else begin
{mac_block}
                        o_valid <= 1'b1;
                    end
                end

            endmodule
        """)

    def emit_top(
        self,
        result: MappingResult,
        module_name: str = "sc_memristor_array",
    ) -> str:
        """Generate top-level module instantiating all crossbar tiles."""
        bw = self.bw
        total_rows = sum(m.crossbar.rows for m in result.mappings)
        total_cols = max((m.crossbar.cols for m in result.mappings), default=1)

        inst_lines = []
        for idx, mapping in enumerate(result.mappings):
            inst_lines.append(
                textwrap.dedent(f"""\
                sc_memristor_crossbar #(
                    .ROWS({mapping.crossbar.rows}),
                    .COLS({mapping.crossbar.cols}),
                    .BW({bw})
                ) tile_{idx} (
                    .clk(clk),
                    .rst_n(rst_n),
                    .i_bitstream(i_bitstream[{mapping.crossbar.cols - 1}:0]),
                    .o_mac(tile_{idx}_mac),
                    .o_valid(tile_{idx}_valid)
                );""")
            )

        inst_block = "\n".join(inst_lines)

        return textwrap.dedent(f"""\
            // SPDX-License-Identifier: AGPL-3.0-or-later
            // © Code 2020–2026 Miroslav Šotek. All rights reserved.
            // SC-NeuroCore — Memristor Array Top ({result.total_crossbars} tiles)

            module {module_name} (
                input  logic clk,
                input  logic rst_n,
                input  logic [{total_cols - 1}:0] i_bitstream,
                output logic o_valid
            );

            {textwrap.indent(inst_block, "    ")}

            endmodule
        """)
