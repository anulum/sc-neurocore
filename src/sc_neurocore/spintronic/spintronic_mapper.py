# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spintronic / Magnetic-Domain SC Mapper

"""Estimate how stochastic-computing weights map onto spintronic device arrays.

The module models four device technologies (domain-wall racetracks,
skyrmion tracks, spin-transfer-torque magnetic tunnel junctions and
spin-orbit-torque MRAM cells) with closed-form estimates: device presets,
Gaussian process variability, Q8.8 weight storage, switching energy, thermal
stability, retention, multi-level cells, write-verify, wear, radiation and
defect bookkeeping. It writes MuMax3 input scripts and reads MuMax3 table
output; it does not run a micromagnetic solver.

Device technologies:
- **Domain-Wall (DW)**: Nanotrack racetrack with SOT-driven motion
- **Skyrmion (SKY)**: Skyrmion nucleation/annihilation gates
- **STT-MTJ**: Spin-transfer torque magnetic tunnel junctions
- **SOT-MRAM**: Spin-orbit torque MRAM cells

Pipeline:
    weight matrix → SpintronicMapper → SpintronicArray and MappingResult;
    device → MuMax3ScriptGenerator → MuMax3 script

Several estimates are simpler than their names suggest, and the docstrings
say so where it matters: a cell reads back the weight it was programmed with,
so variability changes area and energy but never the read weights, and the
Monte-Carlo yield is 1.0 for any weights; the generated Verilog stores the
weights but passes input column 0 to every output row.

Related modules:
- ``memristor/memristor_mapper.py`` draws its device variability the same
  way, as Gaussian relative deviations
- ``hdl_gen/verilog_generator.py`` is the general Verilog generator; this
  module writes its own array wrapper
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Device Technology ────────────────────────────────────────────────


class SpintronicTech(Enum):
    """The device technology of a spintronic cell or array.

    Each member selects a preset of :meth:`SpintronicDeviceConfig.from_tech`
    and an entry of :attr:`SpintronicDeviceConfig.endurance_cycles`.
    """

    DOMAIN_WALL = "domain_wall"
    SKYRMION = "skyrmion"
    STT_MTJ = "stt_mtj"
    SOT_MRAM = "sot_mram"


@dataclass
class MaterialParams:
    """Micromagnetic material parameters.

    Attributes
    ----------
    saturation_magnetisation_a_m : float
        Saturation magnetisation Ms in A/m.
    exchange_stiffness_j_m : float
        Exchange stiffness Aex in J/m.
    dmi_strength_j_m2 : float
        Interfacial Dzyaloshinskii-Moriya strength D in J/m².
    perpendicular_anisotropy_j_m3 : float
        Uniaxial perpendicular anisotropy Ku in J/m³.
    damping_alpha : float
        Gilbert damping, dimensionless.
    temperature_k : float
        Temperature in kelvin; 300 by default.
    """

    saturation_magnetisation_a_m: float  # A/m (Ms)
    exchange_stiffness_j_m: float  # J/m (Aex)
    dmi_strength_j_m2: float  # J/m² (D, Dzyaloshinskii-Moriya)
    perpendicular_anisotropy_j_m3: float  # J/m³ (Ku)
    damping_alpha: float  # Gilbert damping
    temperature_k: float = 300.0

    @classmethod
    def cofeb_mgo(cls) -> MaterialParams:
        """CoFeB/MgO — standard STT-MTJ / SOT-MRAM stack.

        Returns
        -------
        MaterialParams
            Ms 1.2e6 A/m, Aex 1.5e-11 J/m, no DMI, Ku 8e5 J/m³, damping 0.01.
        """
        return cls(
            saturation_magnetisation_a_m=1.2e6,
            exchange_stiffness_j_m=1.5e-11,
            dmi_strength_j_m2=0.0,
            perpendicular_anisotropy_j_m3=8e5,
            damping_alpha=0.01,
        )

    @classmethod
    def pt_co_multilayer(cls) -> MaterialParams:
        """Pt/Co multilayer — skyrmion host.

        Returns
        -------
        MaterialParams
            Ms 5.8e5 A/m, Aex 1.5e-11 J/m, D 3.5e-3 J/m², Ku 6e5 J/m³,
            damping 0.015.
        """
        return cls(
            saturation_magnetisation_a_m=5.8e5,
            exchange_stiffness_j_m=1.5e-11,
            dmi_strength_j_m2=3.5e-3,
            perpendicular_anisotropy_j_m3=6e5,
            damping_alpha=0.015,
        )

    @classmethod
    def w_cofeb(cls) -> MaterialParams:
        """W/CoFeB — SOT switching with heavy-metal underlayer.

        Returns
        -------
        MaterialParams
            Ms 1.1e6 A/m, Aex 1.3e-11 J/m, D 0.5e-3 J/m², Ku 7e5 J/m³,
            damping 0.02.
        """
        return cls(
            saturation_magnetisation_a_m=1.1e6,
            exchange_stiffness_j_m=1.3e-11,
            dmi_strength_j_m2=0.5e-3,
            perpendicular_anisotropy_j_m3=7e5,
            damping_alpha=0.02,
        )


@dataclass
class SpintronicDeviceConfig:
    """Configuration for a single spintronic device.

    The defaults describe a CoFeB/MgO cell; :meth:`from_tech` gives the
    presets of the four technologies.

    Attributes
    ----------
    tech : SpintronicTech
        Device technology.
    material : MaterialParams
        Material stack of the free layer.
    width_nm, length_nm, thickness_nm : float
        Free-layer dimensions in nanometres; each must be positive.
    switching_current_ua : float
        Write current in microamperes; must be positive.
    switching_time_ns : float
        Write pulse duration in nanoseconds; must be positive.
    write_resistance_ohm : float
        Resistance of the write path in ohms; must be positive.
    parallel_resistance_ohm : float
        Resistance in the parallel state in ohms; must be positive.
    retention_years : float
        Stated retention; stored and not used by any estimate.
    tmr_ratio : float
        Tunnelling magnetoresistance ratio, (R_AP - R_P) / R_P; must not be
        negative.
    error_rate : float
        Write error probability per device, used by
        :meth:`SpintronicMapper.map_network`.

    Raises
    ------
    ValueError
        If a dimension, the switching current or time, or a resistance is not
        positive, or the TMR ratio is negative.
    """

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
        """Refuse non-positive dimensions, currents, times and resistances."""
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
        """Return the preset device of one technology.

        Parameters
        ----------
        tech : SpintronicTech
            The technology to build.

        Returns
        -------
        SpintronicDeviceConfig
            The preset's material, dimensions, switching current and time and
            both resistances; ``retention_years``, ``tmr_ratio`` and
            ``error_rate`` keep the class defaults. Domain-wall and skyrmion
            devices use the Pt/Co multilayer, STT-MTJ CoFeB/MgO and SOT-MRAM
            W/CoFeB.
        """
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
        """Footprint of the free layer, width × length, in nm²."""
        return self.width_nm * self.length_nm

    @property
    def switching_energy_fj(self) -> float:
        """Write-path switching energy, E = I² × R_write × t, in femtojoules."""
        i_a = self.switching_current_ua * 1e-6
        return i_a**2 * self.write_resistance_ohm * self.switching_time_ns * 1e6  # fJ

    @property
    def thermal_stability(self) -> float:
        """Thermal stability factor Δ = Ku × V / (kB × T).

        V is width × length × thickness; demagnetising and shape terms are not
        included. Needs to be > 40 for 10-year retention.
        """
        kb = 1.38064852e-23
        volume_m3 = (self.width_nm * self.length_nm * self.thickness_nm) * 1e-27
        t = self.material.temperature_k
        return self.material.perpendicular_anisotropy_j_m3 * volume_m3 / (kb * t)

    @property
    def read_disturb_probability(self) -> float:
        """Probability of read disturb (accidental flip during read).

        Computed as exp(-Δ) with Δ from :attr:`thermal_stability`, without a
        read-current or read-time factor, and 0.0 when Δ is 100 or more.
        """
        delta = self.thermal_stability
        return float(np.exp(-delta)) if delta < 100 else 0.0

    @property
    def endurance_cycles(self) -> int:
        """Write endurance of the technology from a fixed table.

        10**15 cycles for domain-wall, skyrmion and SOT-MRAM devices and
        10**12 for STT-MTJ; table values, not derived from the device's
        parameters.
        """
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

    Each parameter is scaled by ``1 + N(0, sigma / 100)``: width and length,
    and the material's anisotropy, DMI strength, damping and saturation
    magnetisation. Thickness, resistances and switching current and time are
    not varied.

    Attributes
    ----------
    width_sigma_pct, length_sigma_pct : float
        Standard deviation of the width and the length, in per cent.
    ku_sigma_pct, dmi_sigma_pct, damping_sigma_pct, ms_sigma_pct : float
        Standard deviation of the anisotropy, the DMI strength, the damping
        and the saturation magnetisation, in per cent.
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
        """Return a variability-injected copy of the device.

        Parameters
        ----------
        device : SpintronicDeviceConfig
            The nominal device; it is not changed.
        rng : numpy.random.Generator
            Source of the six Gaussian draws, taken in the order width, length,
            anisotropy, DMI, damping, magnetisation.

        Returns
        -------
        SpintronicDeviceConfig
            A deep copy with the varied parameters; width and length are
            floored at 10 nm and the damping at 0.001. The copy is not
            validated again, so a draw can make the anisotropy or the
            magnetisation negative.
        """
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
    """One cell in a spintronic array.

    Attributes
    ----------
    row, col : int
        Position of the cell in its array.
    device : SpintronicDeviceConfig
        The cell's own, possibly varied, device.
    state : int
        Magnetic state: 0 for parallel (P), 1 for anti-parallel (AP).
    weight_q88 : int
        The stored weight in Q8.8 fixed point (256 is 1.0). It is kept as a
        number beside the binary state and read back unchanged.
    """

    row: int
    col: int
    device: SpintronicDeviceConfig
    state: int = 0  # 0 = P (parallel), 1 = AP (anti-parallel)
    weight_q88: int = 256  # Q8.8 = 1.0

    @property
    def resistance_ohm(self) -> float:
        """Cell resistance in ohms: R_P in state 0, R_P × (1 + TMR) in state 1."""
        r_p = self.device.parallel_resistance_ohm
        return r_p * (1 + self.state * self.device.tmr_ratio)


class SpintronicArray:
    """Crossbar array of spintronic devices for SC computation.

    Every cell gets its own copy of the technology's preset device, varied by
    the variability model with draws from one generator seeded by
    ``rng_seed``, row by row.

    Parameters
    ----------
    rows, cols : int
        Array dimensions.
    tech : SpintronicTech
        Technology of every cell.
    variability : VariabilityModel or None
        Process variability; ``None`` uses the default model.
    rng_seed : int
        Seed of the array's random-number generator.

    Attributes
    ----------
    rows, cols : int
        Array dimensions.
    tech : SpintronicTech
        Technology of every cell.
    rng : numpy.random.Generator
        The generator the cells' variability was drawn from.
    cells : list of list of SpintronicCell
        The cells, indexed ``cells[row][col]``.
    """

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
        """Number of cells, rows × cols."""
        return self.rows * self.cols

    @property
    def total_area_um2(self) -> float:
        """Sum of the cells' free-layer footprints in µm², varied sizes included."""
        return sum(c.device.area_nm2 for row in self.cells for c in row) / 1e6

    def program_weights(self, weights_q88: np.ndarray[Any, Any]) -> None:
        """Program Q8.8 weights into the array.

        Each cell stores its weight as given and switches to the
        anti-parallel state when the weight is above 128 (0.5 in Q8.8),
        otherwise to the parallel state. The device's variability does not
        change the stored weight.

        Parameters
        ----------
        weights_q88 : numpy.ndarray
            Two-dimensional weights; only the part that overlaps the array is
            programmed, and cells outside the matrix keep their weights.
        """
        for r in range(min(self.rows, weights_q88.shape[0])):
            for c in range(min(self.cols, weights_q88.shape[1])):
                w = int(weights_q88[r, c])
                self.cells[r][c].weight_q88 = w
                self.cells[r][c].state = 1 if w > 128 else 0

    def read_weights(self) -> np.ndarray[Any, Any]:
        """Read weights back from the array.

        Returns
        -------
        numpy.ndarray
            The stored weights as an int32 array of shape (rows, cols): the
            programmed values themselves, not values derived from the cells'
            resistances.
        """
        w = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                w[r, c] = self.cells[r][c].weight_q88
        return w

    def power_breakdown(self, bitstream_length: int = 256) -> Dict[str, float]:
        """Per-array energy breakdown in femtojoules.

        Parameters
        ----------
        bitstream_length : int
            Number of bits each cell processes.

        Returns
        -------
        dict of str to float
            ``switching_fj``: every cell's switching energy once per bit.
            ``leakage_fj``: 0.1 / R × 10**6 fJ per cell and bit with R the
            cell's resistance in ohms; a 100 mV read bias held for 1 ns, as
            the code's comment says, would give 0.01 / R × 10**6 fJ.
            ``total_fj``: the sum of the two.
        """
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
    """Result of mapping SC bitstreams to a spintronic array.

    Attributes
    ----------
    array_rows, array_cols : int
        Dimensions of the array, those of the weight matrix.
    tech : SpintronicTech
        Technology of the array.
    total_area_um2 : float
        Sum of the varied cells' footprints in µm².
    total_energy_fj : float
        The nominal device's switching energy × cells × bitstream length.
    total_switching_time_ns : float
        The nominal device's switching time × bitstream length.
    bit_error_rate : float
        The nominal device's error rate × cells: an expected number of
        failing devices, which can exceed 1.
    mc_yield : float
        Yield of a Monte-Carlo analysis; :meth:`SpintronicMapper.map_network`
        leaves it at 1.0.
    """

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
    1. Weight storage, with the P/AP state set by a threshold at 0.5
    2. Array sizing from the weight matrix's shape
    3. Energy/timing estimation from the nominal device
    4. A Monte-Carlo yield over arrays of varied devices

    Parameters
    ----------
    tech : SpintronicTech
        Technology of the arrays.
    variability : VariabilityModel or None
        Process variability; ``None`` uses the default model.
    rng_seed : int
        Seed of the generator that seeds each array.

    Attributes
    ----------
    tech : SpintronicTech
        Technology of the arrays.
    variability : VariabilityModel
        The variability applied to every array.
    rng : numpy.random.Generator
        The generator that draws each array's seed.
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
        """Map a weight matrix to a spintronic array.

        Parameters
        ----------
        weights_q88 : numpy.ndarray
            Two-dimensional Q8.8 weights; the array takes their shape.
        bitstream_length : int
            Bits per stochastic stream, which scales energy and time.

        Returns
        -------
        tuple of SpintronicArray and MappingResult
            The programmed array, seeded from the mapper's generator, and the
            estimates: area from the array's varied cells, energy, time and
            error rate from the technology's nominal device.
        """
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
        """Run Monte-Carlo yield analysis.

        Each trial builds an array of varied devices, programs the weights
        and reads them back; a trial passes when no read weight differs from
        its target by more than ``tolerance_q88``. Because a cell reads back
        exactly the weight it stored, every trial passes for weights that fit
        the int32 read-back, whatever the variability.

        Parameters
        ----------
        weights_q88 : numpy.ndarray
            Two-dimensional Q8.8 weights.
        n_trials : int
            Number of arrays to build; must be positive.
        tolerance_q88 : int
            Largest accepted read error in Q8.8 units.

        Returns
        -------
        float
            The fraction of passing trials, 1.0 for such weights.

        Raises
        ------
        ZeroDivisionError
            If ``n_trials`` is 0.
        """
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
        """Return a MuMax3 script that drives a current through the device's free layer.

        The script sets a 128 × 32 × 1 grid over the device's width, length and
        thickness, the material's parameters and temperature, a uniform +z
        start, a current density along x with ``xi`` 0.05 and ``Pol`` 0.56,
        runs for the duration and saves the final magnetisation.

        Parameters
        ----------
        device : SpintronicDeviceConfig
            The device to simulate.
        current_density_a_m2 : float
            Current density in A/m².
        duration_ns : float
            Simulated time in nanoseconds.

        Returns
        -------
        str
            The script's text.
        """
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
        """Return a MuMax3 script that relaxes one Néel skyrmion in the device.

        The script sets a 256 × 256 × 1 grid whose cells are width / 256 in
        both in-plane directions (a square of the device's width), the
        material's parameters and temperature, a uniform +z state with a
        Néel skyrmion placed in region 0, relaxes it and saves the ground
        state.

        Parameters
        ----------
        device : SpintronicDeviceConfig
            The device to simulate.

        Returns
        -------
        str
            The script's text.
        """
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
        """Return a wrapper module with a weight-programming port and bitstream ports.

        The module stores 16-bit Q8.8 weights written through the programming
        port, keeps them through reset, and assigns input column 0 to every
        output row: the weights are not used in the datapath. Its ports are
        unpacked arrays, which need SystemVerilog.

        Parameters
        ----------
        array_name : str
            Name of the module.
        rows, cols : int
            Defaults of the ``ROWS`` and ``COLS`` parameters.
        tech : SpintronicTech
            Technology named in the header comment.

        Returns
        -------
        str
            The module's source text.
        """
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

    Attributes
    ----------
    n_positions : int
        Number of bit positions of the track.
    bits : numpy.ndarray or None
        The stored bits as int8; ``None`` becomes ``n_positions`` zeros.
    shift_current_ua : float
        Shift current in microamperes.
    shift_time_ns : float
        Duration of one shift pulse in nanoseconds.
    shift_error_rate : float
        Probability, per shift, that one random position flips.
    """

    n_positions: int
    bits: Optional[np.ndarray[Any, Any]] = None
    shift_current_ua: float = 100.0
    shift_time_ns: float = 0.5
    shift_error_rate: float = 1e-5

    def __post_init__(self) -> None:
        """Start an empty track with ``n_positions`` zero bits."""
        if self.bits is None:
            self.bits = np.zeros(self.n_positions, dtype=np.int8)

    def load(self, data: np.ndarray[Any, Any]) -> None:
        """Replace the stored bits by the first ``n_positions`` of ``data``.

        Shorter data is not padded, so the track then holds fewer bits.

        Parameters
        ----------
        data : numpy.ndarray
            The bits to store, converted to int8.
        """
        self.bits = np.array(data[: self.n_positions], dtype=np.int8)

    def shift_right(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> None:
        """Shift every bit towards the end of the track, ``n`` times.

        Each shift moves bit i to i + 1, drops the last bit and enters a 0 at
        position 0. With a generator, each shift flips one random position
        with probability ``shift_error_rate``.

        Parameters
        ----------
        n : int
            Number of shifts.
        rng : numpy.random.Generator or None
            Source of shift errors; ``None`` shifts without errors.
        """
        for _ in range(n):
            bits = self.bits if self.bits is not None else np.zeros(self.n_positions, dtype=np.int8)
            self.bits = np.roll(bits, 1).astype(np.int8, copy=False)
            self.bits[0] = 0
            if rng is not None and rng.random() < self.shift_error_rate:
                pos = rng.integers(0, self.n_positions)
                self.bits[pos] ^= 1

    def shift_left(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> None:
        """Shift every bit towards the start of the track, ``n`` times.

        Each shift moves bit i to i - 1, drops the first bit and enters a 0 at
        the last position. With a generator, each shift flips one random
        position with probability ``shift_error_rate``.

        Parameters
        ----------
        n : int
            Number of shifts.
        rng : numpy.random.Generator or None
            Source of shift errors; ``None`` shifts without errors.
        """
        for _ in range(n):
            bits = self.bits if self.bits is not None else np.zeros(self.n_positions, dtype=np.int8)
            self.bits = np.roll(bits, -1).astype(np.int8, copy=False)
            self.bits[-1] = 0
            if rng is not None and rng.random() < self.shift_error_rate:
                pos = rng.integers(0, self.n_positions)
                self.bits[pos] ^= 1

    @property
    def shift_energy_fj(self) -> float:
        """Energy of one shift pulse, I² × 500 Ω × t, in femtojoules.

        The track resistance is a fixed 500 Ω.
        """
        r_ohm = 500.0
        i_a = self.shift_current_ua * 1e-6
        return i_a**2 * r_ohm * self.shift_time_ns * 1e6


# ── Skyrmion Hall Angle Correction (Gap 2) ──────────────────────────


@dataclass
class SkyrmionHallCorrector:
    """Corrects skyrmion trajectory for the skyrmion Hall effect.

    Skyrmions drift at an angle θ_H to the applied current direction. The
    code computes θ_H = arctan(4π |Q| α) with Q the topological charge and α
    the damping. This is not the Thiele form arctan(4πQ / (α 𝒟)), in which the
    damping divides; ``dmi_strength`` is stored and not used.

    Attributes
    ----------
    topological_charge : int
        Topological charge Q; its sign is ignored.
    damping_alpha : float
        Gilbert damping α.
    dmi_strength : float
        DMI strength in J/m²; not used.
    """

    topological_charge: int = 1
    damping_alpha: float = 0.015
    dmi_strength: float = 3.5e-3

    @property
    def hall_angle_deg(self) -> float:
        """The drift angle in degrees, arctan(4π |Q| α)."""
        ratio = 4 * math.pi * abs(self.topological_charge) * self.damping_alpha
        return math.degrees(math.atan(ratio))

    def corrected_position(self, x_drive: float, track_width_nm: float) -> Tuple[float, float]:
        """Return the position a skyrmion reaches after a drive along the track.

        Parameters
        ----------
        x_drive : float
            Distance driven along the track.
        track_width_nm : float
            Width of the track; the transverse drift is clamped to its half.

        Returns
        -------
        tuple of float
            ``(x_drive, y)`` with y = x_drive × tan(θ_H), clamped to
            ±track_width_nm / 2.
        """
        theta = math.radians(self.hall_angle_deg)
        y_drift = x_drive * math.tan(theta)
        y_clamped = max(-track_width_nm / 2, min(track_width_nm / 2, y_drift))
        return (x_drive, y_clamped)

    @property
    def needs_confinement(self) -> bool:
        """Whether the drift angle exceeds 5 degrees."""
        return self.hall_angle_deg > 5.0


# ── Temperature-Dependent Switching (Gap 3) ─────────────────────────


def switching_current_vs_temperature(
    i_c0_ua: float,
    delta_0: float,
    temperature_k: float,
    temp_ref_k: float = 300.0,
) -> float:
    """Ic(T) = Ic0 × max(0.01, 1 - (T / T_ref) / Δ0).

    A linear thermal reduction of the switching current, floored at 1 % of
    Ic0.

    Parameters
    ----------
    i_c0_ua : float
        Switching current at the reference temperature, in microamperes.
    delta_0 : float
        Thermal stability factor at the reference temperature.
    temperature_k : float
        Temperature in kelvin.
    temp_ref_k : float
        Reference temperature in kelvin.

    Returns
    -------
    float
        The switching current in microamperes; ``i_c0_ua`` unchanged when the
        reference temperature or Δ0 is not positive.
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
    """Switching time increases with temperature due to thermal fluctuations.

    Computed as t_sw0 × (1 + 0.1 × (T / T_ref - 1)), a linear model; below the
    reference temperature the time decreases.

    Parameters
    ----------
    t_sw0_ns : float
        Switching time at the reference temperature, in nanoseconds.
    temperature_k : float
        Temperature in kelvin.
    temp_ref_k : float
        Reference temperature in kelvin; must not be 0.

    Returns
    -------
    float
        The switching time in nanoseconds.
    """
    ratio = temperature_k / temp_ref_k
    return t_sw0_ns * (1.0 + 0.1 * (ratio - 1.0))


# ── Retention Failure Probability (Gap 4) ───────────────────────────


def retention_failure_probability(
    thermal_stability: float,
    time_seconds: float,
    attempt_freq_hz: float = 1e9,
) -> float:
    """P_fail = 1 - exp(-t × f0 × exp(-Δ)).

    Parameters
    ----------
    thermal_stability : float
        Thermal stability factor Δ.
    time_seconds : float
        Storage time t in seconds.
    attempt_freq_hz : float
        Attempt frequency f0 in hertz.

    Returns
    -------
    float
        The probability, clamped to [0, 1]; 0.0 when Δ is above 100.
    """
    if thermal_stability > 100:
        return 0.0
    exponent = -thermal_stability
    rate = attempt_freq_hz * math.exp(exponent)
    p = 1.0 - math.exp(-time_seconds * rate)
    return max(0.0, min(1.0, p))


# ── Multi-Level Cell (MLC) Support (Gap 5) ──────────────────────────


@dataclass
class MLCConfig:
    """Multi-level cell configuration for spintronic devices.

    Attributes
    ----------
    bits_per_cell : int
        Bits stored per cell.
    levels : int
        Number of levels; always set to 2**bits_per_cell after construction,
        whatever value is given.
    """

    bits_per_cell: int = 2
    levels: int = 0

    def __post_init__(self) -> None:
        """Set ``levels`` to 2**bits_per_cell."""
        self.levels = 2**self.bits_per_cell

    @property
    def resistance_margins(self) -> List[float]:
        """Resistance levels evenly spaced between R_P and R_AP.

        R_P and R_AP are fixed at 5000 Ω and 12500 Ω, not taken from a
        device.
        """
        r_p, r_ap = 5000.0, 12500.0
        step = (r_ap - r_p) / (self.levels - 1) if self.levels > 1 else 0
        return [r_p + i * step for i in range(self.levels)]

    def quantize_weight(self, weight_float: float) -> int:
        """Quantise a [0, 1] weight to an MLC level.

        Parameters
        ----------
        weight_float : float
            The weight; values outside [0, 1] are clamped to the end levels.

        Returns
        -------
        int
            The nearest level, from 0 to ``levels - 1``.
        """
        level = int(round(weight_float * (self.levels - 1)))
        return max(0, min(self.levels - 1, level))

    def dequantize(self, level: int) -> float:
        """Return the weight of a level, level / (levels - 1).

        Parameters
        ----------
        level : int
            The level; it is not checked against the range.

        Returns
        -------
        float
            The weight; 0.0 for a single-level cell.
        """
        return level / (self.levels - 1) if self.levels > 1 else 0.0

    @property
    def density_improvement(self) -> float:
        """Storage density relative to a one-bit cell, equal to ``bits_per_cell``."""
        return float(self.bits_per_cell)


# ── Write-Verify Loop (Gap 6) ───────────────────────────────────────


@dataclass
class WriteVerifyResult:
    """Result of a write-verify cycle.

    Attributes
    ----------
    target_weight : int
        The Q8.8 weight that was to be written.
    actual_weight : int
        The weight the cell holds after the last attempt.
    attempts : int
        Number of write attempts made.
    success : bool
        Whether the last attempt landed within 4 of the target.
    """

    target_weight: int
    actual_weight: int
    attempts: int
    success: bool

    @property
    def error(self) -> int:
        """Absolute difference between the target and the written weight."""
        return abs(self.target_weight - self.actual_weight)


def write_verify(
    cell: SpintronicCell,
    target_q88: int,
    max_attempts: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> WriteVerifyResult:
    """Program a cell with write-verify loop.

    Each attempt writes the target afresh and sets the cell's state by the
    0.5 threshold; with a generator it then adds Gaussian noise of standard
    deviation 2, truncated to an integer, and clamps the weight to [0, 511].
    The loop stops at the first attempt within 4 of the target. Without a
    generator the first attempt always succeeds.

    Parameters
    ----------
    cell : SpintronicCell
        The cell to program; changed in place.
    target_q88 : int
        The Q8.8 weight to write.
    max_attempts : int
        Largest number of attempts.
    rng : numpy.random.Generator or None
        Source of write noise; ``None`` writes exactly.

    Returns
    -------
    WriteVerifyResult
        The target, the written weight, the attempts used and whether the
        last attempt succeeded.
    """
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

    Attributes
    ----------
    cycles_written : int
        Write cycles so far.
    """

    cycles_written: int = 0

    def tmr_degradation(self, initial_tmr: float, endurance_limit: int) -> float:
        """TMR degrades linearly as cycling approaches endurance limit.

        Parameters
        ----------
        initial_tmr : float
            TMR ratio of a new device.
        endurance_limit : int
            Cycles at which the loss reaches its maximum.

        Returns
        -------
        float
            initial_tmr × (1 - 0.3 × min(1, cycles / limit)): at most a 30 %
            loss; ``initial_tmr`` unchanged for a limit that is not positive.
        """
        if endurance_limit <= 0:
            return initial_tmr
        frac = min(1.0, self.cycles_written / endurance_limit)
        return initial_tmr * (1.0 - 0.3 * frac)  # up to 30% TMR loss

    def stability_degradation(self, initial_delta: float, endurance_limit: int) -> float:
        """Thermal stability degrades with oxide breakdown.

        Parameters
        ----------
        initial_delta : float
            Thermal stability factor of a new device.
        endurance_limit : int
            Cycles at which the loss reaches its maximum.

        Returns
        -------
        float
            initial_delta × (1 - 0.2 × min(1, cycles / limit)): at most a 20 %
            loss; ``initial_delta`` unchanged for a limit that is not
            positive.
        """
        if endurance_limit <= 0:
            return initial_delta
        frac = min(1.0, self.cycles_written / endurance_limit)
        return initial_delta * (1.0 - 0.2 * frac)

    @property
    def is_worn_out(self) -> bool:
        """Whether the TMR of a 1.5-TMR device rated for 10**12 cycles fell below 0.5.

        The TMR loss stops at 30 %, so the TMR never falls below 1.05 and this
        is always ``False``.
        """
        return self.cycles_written > 0 and self.tmr_degradation(1.5, 10**12) < 0.5

    def write(self, n: int = 1) -> None:
        """Count ``n`` more write cycles.

        Parameters
        ----------
        n : int
            Cycles to add.
        """
        self.cycles_written += n


# ── Radiation Hardness Model (Gap 8) ────────────────────────────────


@dataclass
class RadiationModel:
    """SEU and TID models for spintronic devices in radiation environments.

    Attributes
    ----------
    seu_cross_section_cm2 : float
        Single-event-upset cross-section per device, in cm².
    tid_threshold_krad : float
        Total ionising dose, in krad, at which performance has halved.
    """

    seu_cross_section_cm2: float = 1e-14  # per device
    tid_threshold_krad: float = 1000.0

    def seu_rate(self, flux_particles_cm2_s: float, n_devices: int) -> float:
        """SEU events per second.

        Parameters
        ----------
        flux_particles_cm2_s : float
            Particle flux in particles per cm² per second.
        n_devices : int
            Number of devices exposed.

        Returns
        -------
        float
            Cross-section × flux × devices.
        """
        return self.seu_cross_section_cm2 * flux_particles_cm2_s * n_devices

    def tid_degradation(self, dose_krad: float) -> float:
        """Fraction of performance remaining after TID.

        Parameters
        ----------
        dose_krad : float
            Total ionising dose in krad.

        Returns
        -------
        float
            1 - 0.5 × dose / threshold below the threshold, and 0.5 at and
            above it.
        """
        if dose_krad >= self.tid_threshold_krad:
            return 0.5  # 50% degradation at threshold
        return 1.0 - 0.5 * (dose_krad / self.tid_threshold_krad)

    @property
    def is_rad_hard(self) -> bool:
        """Whether the TID threshold is at least 100 krad.

        Spintronic storage is not charge-based, which is why the default
        threshold is high.
        """
        return self.tid_threshold_krad >= 100.0


# ── Array Defect Map / Repair (Gap 9) ───────────────────────────────


@dataclass
class DefectEntry:
    """One defective cell in the array.

    Attributes
    ----------
    row, col : int
        Position of the cell.
    defect_type : str
        Kind of defect, free text; the module uses no list of kinds.
    """

    row: int
    col: int
    defect_type: str  # "stuck_p", "stuck_ap", "open", "short"


class DefectMap:
    """Tracks and remaps defective cells in a spintronic array.

    Attributes
    ----------
    defects : list of DefectEntry
        The recorded defects, in the order added.
    remap : dict
        Spare address for each remapped ``(row, col)``.
    """

    def __init__(self) -> None:
        self.defects: List[DefectEntry] = []
        self.remap: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def add_defect(self, row: int, col: int, defect_type: str) -> None:
        """Record one defective cell.

        Parameters
        ----------
        row, col : int
            Position of the cell.
        defect_type : str
            Kind of defect, free text.
        """
        self.defects.append(DefectEntry(row, col, defect_type))

    @property
    def defect_count(self) -> int:
        """Number of recorded defects, repeated cells counted again."""
        return len(self.defects)

    def defect_rate(self, total_cells: int) -> float:
        """Return the recorded defects per cell.

        Parameters
        ----------
        total_cells : int
            Number of cells of the array.

        Returns
        -------
        float
            Defect count / total cells; 0.0 for an array without cells.
        """
        if total_cells <= 0:
            return 0.0
        return self.defect_count / total_cells

    def add_remap(self, bad: Tuple[int, int], spare: Tuple[int, int]) -> None:
        """Send accesses of one address to a spare address.

        Parameters
        ----------
        bad : tuple of int
            The ``(row, col)`` to replace.
        spare : tuple of int
            The ``(row, col)`` that replaces it.
        """
        self.remap[bad] = spare

    def is_defective(self, row: int, col: int) -> bool:
        """Return whether a defect is recorded at one position.

        Parameters
        ----------
        row, col : int
            Position of the cell.

        Returns
        -------
        bool
            ``True`` when any recorded defect has that position.
        """
        return any(d.row == row and d.col == col for d in self.defects)

    def effective_address(self, row: int, col: int) -> Tuple[int, int]:
        """Return the address an access to one position uses.

        Parameters
        ----------
        row, col : int
            Position of the cell.

        Returns
        -------
        tuple of int
            The spare address when the position is remapped, else the
            position itself.
        """
        return self.remap.get((row, col), (row, col))


# ── MuMax3 Output Parser (Gap 10) ───────────────────────────────────


@dataclass
class MuMax3Result:
    """Parsed result from a MuMax3 simulation.

    Attributes
    ----------
    final_mx, final_my, final_mz : float
        Average reduced magnetisation components of the last table row.
    switched : bool
        Whether the last row's z component is negative.
    energy_j : float
        Energy in joules; not read by the parser.
    sim_time_ns : float
        Simulated time of the last row in nanoseconds.
    """

    final_mx: float = 0.0
    final_my: float = 0.0
    final_mz: float = 0.0
    switched: bool = False
    energy_j: float = 0.0
    sim_time_ns: float = 0.0

    @property
    def magnetisation_magnitude(self) -> float:
        """Length of the average magnetisation vector (mx, my, mz)."""
        return math.sqrt(self.final_mx**2 + self.final_my**2 + self.final_mz**2)


class MuMax3OutputParser:
    """Parses MuMax3 table output for co-simulation integration."""

    @staticmethod
    def parse_table(text: str) -> MuMax3Result:
        """Parse a MuMax3 .table output (TSV with header).

        Lines starting with ``#`` and empty lines are skipped. The last
        remaining line is split at tabs, or at whitespace when it has fewer
        than four tab-separated fields, and read as time in seconds, mx, my
        and mz.

        Parameters
        ----------
        text : str
            The table's text.

        Returns
        -------
        MuMax3Result
            The last row's time and magnetisation, ``switched`` when mz is
            negative; an all-default result when no row is left or it does not
            parse.
        """
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
        """Return whether a result switched and stayed nearly saturated.

        Parameters
        ----------
        result : MuMax3Result
            A parsed result.

        Returns
        -------
        bool
            ``True`` when ``switched`` is set and the magnetisation's
            magnitude exceeds 0.9.
        """
        return result.switched and result.magnetisation_magnitude > 0.9
