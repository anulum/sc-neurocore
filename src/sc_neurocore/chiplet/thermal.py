# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet package thermal network solver

"""HotSpot-style steady-state and transient chiplet thermal analysis.

The conductance formulation follows Skadron et al., *IEEE TVLSI* 2006,
with inter-die coupling represented as symmetric bond conductance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from sc_neurocore.chiplet.topology import ChipletTopology, InterposerTech


FloatArray = npt.NDArray[np.float64]

_R_THERMAL_K_PER_W: dict[InterposerTech, float] = {
    InterposerTech.UCIE: 0.8,
    InterposerTech.BOW: 3.0,
    InterposerTech.EMIB: 0.5,
    InterposerTech.COWOS: 0.3,
    InterposerTech.ORGANIC: 8.0,
    InterposerTech.CUSTOM: 1.0,
}


@dataclass
class DieThermal:
    """Hold thermal properties and runtime state for one die.

    Parameters
    ----------
    die_id
        Non-negative package die identifier.
    temperature_c
        Current junction temperature in degrees Celsius.
    power_mw
        Dissipated power in milliwatts.
    heat_capacity_j_per_k
        Positive die heat capacity in joules per kelvin.
    r_to_ambient_k_per_w
        Positive junction-to-ambient resistance in kelvin per watt.
    r_spread_k_per_w
        Non-negative within-die spreading resistance in kelvin per watt.
    max_temperature_c
        Junction-temperature throttle threshold in degrees Celsius.
    """

    die_id: int
    temperature_c: float = 25.0
    power_mw: float = 100.0
    heat_capacity_j_per_k: float = 0.083
    r_to_ambient_k_per_w: float = 1.5
    r_spread_k_per_w: float = 0.2
    max_temperature_c: float = 105.0

    def __post_init__(self) -> None:
        """Validate the die identity and finite physical thermal properties."""
        if self.die_id < 0:
            raise ValueError("die_id must be >= 0")
        values = (
            self.temperature_c,
            self.power_mw,
            self.heat_capacity_j_per_k,
            self.r_to_ambient_k_per_w,
            self.r_spread_k_per_w,
            self.max_temperature_c,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("thermal properties must be finite")
        if self.power_mw < 0:
            raise ValueError("power_mw must be >= 0")
        if self.heat_capacity_j_per_k <= 0 or self.r_to_ambient_k_per_w <= 0:
            raise ValueError("heat capacity and ambient resistance must be > 0")
        if self.r_spread_k_per_w < 0:
            raise ValueError("r_spread_k_per_w must be >= 0")

    @property
    def is_throttled(self) -> bool:
        """Return whether the current temperature meets the throttle threshold."""
        return self.temperature_c >= self.max_temperature_c


@dataclass
class PackageThermalReport:
    """Contain steady-state, transient, and conductance evidence for a package."""

    die_temps: dict[int, float] = field(default_factory=dict)
    max_temp: float = 0.0
    throttled_dies: list[int] = field(default_factory=list)
    transient_temps: FloatArray | None = None
    transient_times_s: FloatArray | None = None
    conductance_matrix: FloatArray | None = None


def _build_conductance_matrix(
    topology: ChipletTopology,
    die_state: dict[int, DieThermal],
) -> tuple[FloatArray, FloatArray, list[int]]:
    die_id_order = [die.die_id for die in topology.dies]
    if len(set(die_id_order)) != len(die_id_order):
        raise ValueError("topology die identifiers must be unique")
    index_of = {die_id: index for index, die_id in enumerate(die_id_order)}
    conductance = np.zeros((len(die_id_order), len(die_id_order)), dtype=np.float64)
    ambient = np.zeros(len(die_id_order), dtype=np.float64)
    for die_id in die_id_order:
        ambient[index_of[die_id]] = 1.0 / die_state[die_id].r_to_ambient_k_per_w
    for link in topology.links:
        source_index = index_of.get(link.src_die)
        destination_index = index_of.get(link.dst_die)
        if source_index is None or destination_index is None or source_index == destination_index:
            continue
        link_resistance = (
            link.thermal_resistance_k_per_w
            if link.thermal_resistance_k_per_w is not None
            else _R_THERMAL_K_PER_W[link.technology]
        )
        bond_resistance = (
            link_resistance
            + die_state[link.src_die].r_spread_k_per_w
            + die_state[link.dst_die].r_spread_k_per_w
        )
        link_conductance = 1.0 / bond_resistance
        conductance[source_index, destination_index] += link_conductance
        conductance[destination_index, source_index] += link_conductance
    return conductance, ambient, die_id_order


def _solve_steady_state(
    off_diagonal: FloatArray,
    ambient_conductance: FloatArray,
    power_w: FloatArray,
    ambient_c: float,
) -> FloatArray:
    diagonal = off_diagonal.sum(axis=1) + ambient_conductance
    system = np.diag(diagonal) - off_diagonal
    forcing = power_w + ambient_conductance * ambient_c
    return np.asarray(np.linalg.solve(system, forcing), dtype=np.float64)


def _solve_transient(
    off_diagonal: FloatArray,
    ambient_conductance: FloatArray,
    capacities: FloatArray,
    power_w: FloatArray,
    ambient_c: float,
    initial_temperature_c: FloatArray,
    dt_s: float,
    n_steps: int,
) -> FloatArray:
    diagonal = off_diagonal.sum(axis=1) + ambient_conductance
    system = np.diag(diagonal) - off_diagonal
    forcing = power_w + ambient_conductance * ambient_c
    capacity_over_dt = np.diag(capacities / dt_s)
    implicit_system = capacity_over_dt + system
    temperature = initial_temperature_c.copy()
    trajectory = np.empty((n_steps, off_diagonal.shape[0]), dtype=np.float64)
    for step in range(n_steps):
        right_hand_side = (capacities / dt_s) * temperature + forcing
        temperature = np.asarray(
            np.linalg.solve(implicit_system, right_hand_side), dtype=np.float64
        )
        trajectory[step] = temperature
    return trajectory


def simulate_thermal(
    topology: ChipletTopology,
    power_per_die_mw: dict[int, float] | None = None,
    ambient_c: float = 25.0,
    *,
    die_state: dict[int, DieThermal] | None = None,
    transient_steps: int = 0,
    transient_dt_s: float = 1e-3,
) -> PackageThermalReport:
    """Solve the package thermal network.

    Parameters
    ----------
    topology
        Dies and interposer links forming the thermal network.
    power_per_die_mw
        Optional per-die dissipation overrides in milliwatts.
    ambient_c
        Ambient temperature in degrees Celsius.
    die_state
        Optional per-die material and runtime-state overrides.
    transient_steps
        Number of implicit-Euler transient samples after cold start.
    transient_dt_s
        Positive transient integration step in seconds.

    Returns
    -------
    PackageThermalReport
        Steady-state temperatures and optional transient trajectory.

    Raises
    ------
    ValueError
        If the topology is empty or any numerical contract is invalid.
    """
    if not topology.dies:
        raise ValueError("topology must contain at least one die")
    if not math.isfinite(ambient_c):
        raise ValueError("ambient_c must be finite")
    if transient_steps < 0:
        raise ValueError("transient_steps must be >= 0")
    if not math.isfinite(transient_dt_s) or transient_dt_s <= 0:
        raise ValueError("transient_dt_s must be finite and > 0")
    if power_per_die_mw is not None and any(
        not math.isfinite(power) or power < 0 for power in power_per_die_mw.values()
    ):
        raise ValueError("power_per_die_mw values must be finite and >= 0")

    state: dict[int, DieThermal] = {}
    for die in topology.dies:
        thermal = (
            die_state[die.die_id]
            if die_state is not None and die.die_id in die_state
            else DieThermal(die_id=die.die_id)
        )
        thermal.power_mw = (
            power_per_die_mw.get(die.die_id, 100.0) if power_per_die_mw is not None else 100.0
        )
        state[die.die_id] = thermal

    off_diagonal, ambient_conductance, die_id_order = _build_conductance_matrix(topology, state)
    power_w = np.asarray(
        [state[die_id].power_mw / 1000.0 for die_id in die_id_order], dtype=np.float64
    )
    steady_state = _solve_steady_state(off_diagonal, ambient_conductance, power_w, ambient_c)
    report = PackageThermalReport(conductance_matrix=off_diagonal)
    for index, die_id in enumerate(die_id_order):
        temperature = float(steady_state[index])
        state[die_id].temperature_c = temperature
        report.die_temps[die_id] = temperature
        report.max_temp = max(report.max_temp, temperature)
        if state[die_id].is_throttled:
            report.throttled_dies.append(die_id)

    if transient_steps:
        capacities = np.asarray(
            [state[die_id].heat_capacity_j_per_k for die_id in die_id_order], dtype=np.float64
        )
        initial_temperature = np.full(len(die_id_order), ambient_c, dtype=np.float64)
        report.transient_temps = _solve_transient(
            off_diagonal,
            ambient_conductance,
            capacities,
            power_w,
            ambient_c,
            initial_temperature,
            transient_dt_s,
            transient_steps,
        )
        report.transient_times_s = (
            np.arange(1, transient_steps + 1, dtype=np.float64) * transient_dt_s
        )
    return report


__all__ = ["DieThermal", "PackageThermalReport", "simulate_thermal"]
