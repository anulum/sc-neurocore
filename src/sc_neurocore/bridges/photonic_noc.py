# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic Network-on-Chip Bridge

"""Photonic NoC bridge for SC bitstream networks.

Compiles SC neural networks into photonic network-on-chip interconnect
specifications, modeling:

- **Waveguide routing** — optical paths between processing elements
- **MZI-based gates** — Mach-Zehnder interferometer SC computation
- **Wavelength-division multiplexing** — parallel bitstream channels
- **Power budget analysis** — insertion loss, crosstalk, optical SNR
- **GDSII layout export** — photonic design automation integration

Architecture
------------

::

    SC Network  →  Waveguide Router  →  MZI Compiler  →  Power Budget
         ↓               ↓                  ↓                 ↓
    Populations      Topology          MZI cascade        Loss model
    Projections     Routing table      WDM channels       OSNR check

Dependencies
------------

- ``numpy`` — required
- ``gdstk`` — optional, soft-imported for GDSII export
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────

_C_VACUUM = 2.998e8  # m/s
_SI_REFRACTIVE_INDEX = 3.48  # silicon at 1550 nm
_WAVEGUIDE_LOSS_DB_CM = 2.0  # dB/cm typical Si photonic
_SPLITTER_LOSS_DB = 0.3  # dB per Y-junction
_MZI_INSERTION_LOSS_DB = 0.5  # dB per MZI stage
_CROSSING_LOSS_DB = 0.08  # dB per waveguide crossing
_DETECTOR_SENSITIVITY_DBM = -20.0  # minimum detectable
_LASER_POWER_DBM = 0.0  # on-chip source power

# ── Soft imports ──────────────────────────────────────────────────────

try:
    import gdstk

    _HAS_GDSTK = True
except ImportError:
    gdstk = None
    _HAS_GDSTK = False


# ══════════════════════════════════════════════════════════════════════
# Data Types
# ══════════════════════════════════════════════════════════════════════


class WaveguideType(Enum):
    """Photonic waveguide type."""

    STRIP = "strip"
    RIB = "rib"
    SLOT = "slot"


@dataclass
class WaveguideSegment:
    """A single waveguide path segment.

    Attributes
    ----------
    source : int
        Source node index.
    target : int
        Target node index.
    length_um : float
        Physical length in micrometers.
    wavelength_nm : float
        Operating wavelength (default 1550 nm).
    loss_db : float
        Total insertion loss for this segment.
    n_crossings : int
        Number of waveguide crossings.
    wg_type : WaveguideType
        Waveguide geometry type.
    """

    source: int
    target: int
    length_um: float = 100.0
    wavelength_nm: float = 1550.0
    loss_db: float = 0.0
    n_crossings: int = 0
    wg_type: WaveguideType = WaveguideType.STRIP


@dataclass
class MZIGate:
    """Mach-Zehnder interferometer gate specification.

    Models a single MZI stage implementing an SC computing operation
    via thermo-optic or electro-optic phase shifting.

    Attributes
    ----------
    gate_id : str
        Unique gate identifier.
    operation : str
        Gate operation type (AND, OR, NOT, MUL, ADD).
    input_ports : list[int]
        Input waveguide port indices.
    output_port : int
        Output waveguide port index.
    phase_shift_rad : float
        Applied phase shift in radians.
    arm_length_um : float
        MZI arm length in micrometers.
    insertion_loss_db : float
        Total insertion loss.
    extinction_ratio_db : float
        On/off extinction ratio.
    """

    gate_id: str = ""
    operation: str = "MUL"
    input_ports: list[int] = field(default_factory=list)
    output_port: int = 0
    phase_shift_rad: float = 0.0
    arm_length_um: float = 200.0
    insertion_loss_db: float = _MZI_INSERTION_LOSS_DB
    extinction_ratio_db: float = 20.0


@dataclass
class WDMChannel:
    """Wavelength-division multiplexing channel.

    Attributes
    ----------
    channel_id : int
        Channel index.
    wavelength_nm : float
        Center wavelength.
    bandwidth_nm : float
        Channel bandwidth (default 0.8 nm for DWDM).
    signal_name : str
        Associated SC signal name.
    power_dbm : float
        Launch power.
    """

    channel_id: int = 0
    wavelength_nm: float = 1550.0
    bandwidth_nm: float = 0.8
    signal_name: str = ""
    power_dbm: float = _LASER_POWER_DBM


@dataclass
class PhotonicCircuitDesign:
    """Complete photonic NoC design.

    Attributes
    ----------
    name : str
        Design name.
    waveguides : list[WaveguideSegment]
        All waveguide segments.
    mzi_gates : list[MZIGate]
        All MZI computing stages.
    wdm_channels : list[WDMChannel]
        WDM channel assignments.
    n_nodes : int
        Number of processing element nodes.
    routing_table : dict[tuple[int, int], list[int]]
        Hop-by-hop routing table.
    total_area_um2 : float
        Estimated chip area.
    """

    name: str = ""
    waveguides: list[WaveguideSegment] = field(default_factory=list)
    mzi_gates: list[MZIGate] = field(default_factory=list)
    wdm_channels: list[WDMChannel] = field(default_factory=list)
    n_nodes: int = 0
    routing_table: Dict[tuple[int, int], list[int]] = field(default_factory=dict)
    total_area_um2: float = 0.0


# ══════════════════════════════════════════════════════════════════════
# Waveguide Router
# ══════════════════════════════════════════════════════════════════════


class WaveguideRouter:
    """Route waveguides between SC network nodes.

    Uses a mesh topology with shortest-path (Manhattan) routing.

    Parameters
    ----------
    pitch_um : float
        Node-to-node pitch in micrometers (default 250).
    loss_db_per_cm : float
        Waveguide propagation loss (default 2.0 dB/cm).
    """

    def __init__(
        self,
        pitch_um: float = 250.0,
        loss_db_per_cm: float = _WAVEGUIDE_LOSS_DB_CM,
    ) -> None:
        self._pitch_um = pitch_um
        self._loss_db_per_cm = loss_db_per_cm

    def route(
        self,
        adjacency: np.ndarray[Any, Any],
        node_labels: list[str] | None = None,
    ) -> list[WaveguideSegment]:
        """Route waveguides for an SC network adjacency matrix.

        Parameters
        ----------
        adjacency : np.ndarray
            N×N weight matrix.
        node_labels : list[str] | None
            Optional node labels.

        Returns
        -------
        list[WaveguideSegment]
        """
        n = adjacency.shape[0]
        segments: list[WaveguideSegment] = []

        # Place nodes on a sqrt(N) × sqrt(N) mesh
        grid_size = max(int(math.ceil(math.sqrt(n))), 1)

        for i in range(n):
            for j in range(i + 1, n):
                w = abs(float(adjacency[i, j])) + abs(float(adjacency[j, i]))
                if w < 1e-12:
                    continue

                # Manhattan distance on mesh
                ri, ci_ = divmod(i, grid_size)
                rj, cj = divmod(j, grid_size)
                manhattan = abs(ri - rj) + abs(ci_ - cj)
                length_um = manhattan * self._pitch_um

                # Loss model
                loss = length_um * 1e-4 * self._loss_db_per_cm  # um→cm
                n_crossings = max(0, manhattan - 1)
                loss += n_crossings * _CROSSING_LOSS_DB

                segments.append(
                    WaveguideSegment(
                        source=i,
                        target=j,
                        length_um=length_um,
                        loss_db=loss,
                        n_crossings=n_crossings,
                    )
                )

        return segments


# ══════════════════════════════════════════════════════════════════════
# MZI Compiler
# ══════════════════════════════════════════════════════════════════════


class MZICompiler:
    """Compile SC operations into MZI gate cascades.

    Maps SC gates to photonic MZI operations:
    - AND/MUL → MZI with π/2 phase shift (coherent multiplication)
    - OR/ADD → Y-junction combiner
    - NOT → MZI with π phase shift (bar state)

    Parameters
    ----------
    arm_length_um : float
        Default MZI arm length (default 200 μm).
    """

    _PHASE_MAP = {
        "AND": math.pi / 2,
        "MUL": math.pi / 2,
        "OR": math.pi / 4,
        "ADD": math.pi / 4,
        "NOT": math.pi,
        "THRESHOLD": math.pi / 3,
    }

    def __init__(self, arm_length_um: float = 200.0) -> None:
        self._arm_length = arm_length_um

    def compile_gate(
        self,
        gate_type: str,
        input_ports: list[int],
        output_port: int,
        gate_id: str = "",
    ) -> MZIGate:
        """Compile a single SC gate to an MZI specification.

        Parameters
        ----------
        gate_type : str
            Gate operation (AND, OR, NOT, MUL, ADD, THRESHOLD).
        input_ports : list[int]
            Input waveguide port indices.
        output_port : int
            Output waveguide port index.
        gate_id : str
            Unique identifier.

        Returns
        -------
        MZIGate
        """
        phase = self._PHASE_MAP.get(gate_type.upper(), math.pi / 2)

        return MZIGate(
            gate_id=gate_id or f"mzi_{gate_type}_{output_port}",
            operation=gate_type.upper(),
            input_ports=input_ports,
            output_port=output_port,
            phase_shift_rad=phase,
            arm_length_um=self._arm_length,
            insertion_loss_db=_MZI_INSERTION_LOSS_DB,
        )

    def compile_network(
        self,
        gates: list[Dict[str, Any]],
    ) -> list[MZIGate]:
        """Compile a list of SC gate specs into MZI cascade.

        Parameters
        ----------
        gates : list[dict]
            Each: ``type``, ``inputs`` (list[int]), ``output`` (int).

        Returns
        -------
        list[MZIGate]
        """
        mzi_list: list[MZIGate] = []
        for i, g in enumerate(gates):
            mzi = self.compile_gate(
                gate_type=g["type"],
                input_ports=g["inputs"],
                output_port=g["output"],
                gate_id=f"mzi_{i}",
            )
            mzi_list.append(mzi)
        return mzi_list


# ══════════════════════════════════════════════════════════════════════
# WDM Channel Assigner
# ══════════════════════════════════════════════════════════════════════


class WDMAssigner:
    """Assign WDM channels to SC signal paths.

    Parameters
    ----------
    base_wavelength_nm : float
        Starting wavelength (default 1550.0 nm).
    channel_spacing_nm : float
        Channel spacing (default 0.8 nm for 100 GHz DWDM).
    max_channels : int
        Hard cap on the number of channels the assigner will emit.
        Default 96 follows the ITU-T G.694.1 DWDM C-band grid at
        50 GHz spacing (~0.4 nm). At the default 0.8 nm spacing the
        physical C-band (~1530-1565 nm, ~35 nm) only fits ~44
        channels — the cap protects callers from silently spilling
        into invalid wavelengths. Pass a larger value (or
        ``max_channels=0`` to disable) for multi-band (C+L+S)
        designs.

    Raises
    ------
    ValueError
        From :meth:`assign` when ``len(signal_names)`` exceeds
        ``max_channels`` and ``max_channels > 0``.
    """

    def __init__(
        self,
        base_wavelength_nm: float = 1550.0,
        channel_spacing_nm: float = 0.8,
        max_channels: int = 96,
    ) -> None:
        self._base_wl = base_wavelength_nm
        self._spacing = channel_spacing_nm
        self._max_channels = max_channels

    def assign(
        self,
        signal_names: list[str],
        power_dbm: float = _LASER_POWER_DBM,
    ) -> list[WDMChannel]:
        """Assign a WDM channel to each signal.

        Parameters
        ----------
        signal_names : list[str]
            SC signal names.
        power_dbm : float
            Launch power per channel.

        Returns
        -------
        list[WDMChannel]

        Raises
        ------
        ValueError
            If ``len(signal_names) > self._max_channels`` and the
            cap is non-zero. See class-level ``max_channels``.
        """
        n = len(signal_names)
        if self._max_channels > 0 and n > self._max_channels:
            raise ValueError(
                f"WDMAssigner.assign: {n} signals exceeds the "
                f"max_channels cap of {self._max_channels}. "
                f"Either reduce the signal count, raise max_channels, "
                f"or use multi-band (e.g. C+L+S) by extending the "
                f"assigner."
            )
        channels: list[WDMChannel] = []
        for i, name in enumerate(signal_names):
            channels.append(
                WDMChannel(
                    channel_id=i,
                    wavelength_nm=self._base_wl + i * self._spacing,
                    bandwidth_nm=self._spacing * 0.5,
                    signal_name=name,
                    power_dbm=power_dbm,
                )
            )
        return channels


# ══════════════════════════════════════════════════════════════════════
# Power Budget Analyzer
# ══════════════════════════════════════════════════════════════════════


class PowerBudgetAnalyzer:
    """Optical power budget and OSNR analysis.

    Computes end-to-end power budget for each path in the photonic
    circuit, flagging paths that exceed the detector sensitivity.
    """

    def analyze(
        self,
        design: PhotonicCircuitDesign,
        laser_power_dbm: float = _LASER_POWER_DBM,
        detector_sensitivity_dbm: float = _DETECTOR_SENSITIVITY_DBM,
    ) -> Dict[str, Any]:
        """Run power budget analysis.

        Returns
        -------
        dict
            ``paths``, ``worst_margin_db``, ``n_failed``, ``total_loss_db``.
        """
        paths: list[Dict[str, Any]] = []
        worst_margin = float("inf")
        n_failed = 0

        for wg in design.waveguides:
            # Accumulate losses along path
            mzi_loss = sum(
                m.insertion_loss_db
                for m in design.mzi_gates
                if wg.source in m.input_ports or wg.target == m.output_port
            )
            total_loss = wg.loss_db + mzi_loss
            received_power = laser_power_dbm - total_loss
            margin = received_power - detector_sensitivity_dbm
            failed = margin < 0

            if margin < worst_margin:
                worst_margin = margin

            if failed:
                n_failed += 1

            paths.append(
                {
                    "source": wg.source,
                    "target": wg.target,
                    "waveguide_loss_db": wg.loss_db,
                    "mzi_loss_db": mzi_loss,
                    "total_loss_db": total_loss,
                    "received_power_dbm": received_power,
                    "margin_db": margin,
                    "passed": not failed,
                }
            )

        return {
            "paths": paths,
            "worst_margin_db": worst_margin if paths else 0.0,
            "n_failed": n_failed,
            "n_paths": len(paths),
            "laser_power_dbm": laser_power_dbm,
            "detector_sensitivity_dbm": detector_sensitivity_dbm,
        }


# ══════════════════════════════════════════════════════════════════════
# SC-to-Photonic Compiler (top-level orchestrator)
# ══════════════════════════════════════════════════════════════════════


class SCToPhotonic:
    """Top-level compiler: SC network → photonic NoC design.

    Parameters
    ----------
    pitch_um : float
        Mesh pitch (default 250 μm).
    arm_length_um : float
        MZI arm length (default 200 μm).
    """

    def __init__(
        self,
        pitch_um: float = 250.0,
        arm_length_um: float = 200.0,
    ) -> None:
        self._router = WaveguideRouter(pitch_um=pitch_um)
        self._mzi = MZICompiler(arm_length_um=arm_length_um)
        self._wdm = WDMAssigner()

    def compile(
        self,
        adjacency: np.ndarray[Any, Any],
        node_labels: list[str] | None = None,
        gate_specs: list[Dict[str, Any]] | None = None,
        name: str = "sc_photonic",
    ) -> PhotonicCircuitDesign:
        """Compile SC network into a photonic design.

        Parameters
        ----------
        adjacency : np.ndarray
            N×N weight matrix.
        node_labels : list[str] | None
            Node labels.
        gate_specs : list[dict] | None
            Optional MZI gate specifications.
        name : str
            Design name.

        Returns
        -------
        PhotonicCircuitDesign
        """
        n = adjacency.shape[0]
        labels = node_labels or [f"pe{i}" for i in range(n)]

        # Route waveguides
        waveguides = self._router.route(adjacency)

        # Compile MZI gates
        mzi_gates: list[MZIGate] = []
        if gate_specs:
            mzi_gates = self._mzi.compile_network(gate_specs)
        else:
            # Auto-generate one MZI per output node based on adjacency
            for j in range(n):
                inputs = [i for i in range(n) if abs(adjacency[i, j]) > 1e-12 and i != j]
                if inputs:
                    op = "MUL" if len(inputs) >= 2 else "NOT"
                    mzi_gates.append(self._mzi.compile_gate(op, inputs, j, f"mzi_{labels[j]}"))

        # Assign WDM channels
        wdm_channels = self._wdm.assign(labels)

        # Estimate area
        grid = max(int(math.ceil(math.sqrt(n))), 1)
        pitch = self._router._pitch_um
        area = (grid * pitch) ** 2

        return PhotonicCircuitDesign(
            name=name,
            waveguides=waveguides,
            mzi_gates=mzi_gates,
            wdm_channels=wdm_channels,
            n_nodes=n,
            total_area_um2=area,
        )


# ══════════════════════════════════════════════════════════════════════
# Thermal Phase Shifter Model
# ══════════════════════════════════════════════════════════════════════


class ThermalPhaseShifter:
    """Thermo-optic phase shifter model for MZI tuning.

    Parameters
    ----------
    heater_length_um : float
        Heater length (default 100 μm).
    dn_dt : float
        Thermo-optic coefficient (default 1.86e-4 K⁻¹ for Si).
    thermal_resistance_kw : float
        Heater thermal resistance (default 10 K/mW).
    """

    def __init__(
        self,
        heater_length_um: float = 100.0,
        dn_dt: float = 1.86e-4,
        thermal_resistance_kw: float = 10.0,
    ) -> None:
        self._heater_length = heater_length_um
        self._dn_dt = dn_dt
        self._thermal_r = thermal_resistance_kw

    def power_for_phase(self, phase_rad: float, wavelength_nm: float = 1550.0) -> float:
        """Compute electrical power needed for a given phase shift.

        Returns
        -------
        float
            Required power in milliwatts.
        """
        wl_m = wavelength_nm * 1e-9
        l_m = self._heater_length * 1e-6
        delta_t = (phase_rad * wl_m) / (2 * math.pi * self._dn_dt * l_m)
        return abs(delta_t) / self._thermal_r  # mW

    def analyze_design(self, design: PhotonicCircuitDesign) -> Dict[str, Any]:
        """Compute total power budget for all MZI phase shifters.

        Returns
        -------
        dict
            Per-gate power and total.
        """
        gate_powers: list[Dict[str, Any]] = []
        total_mw = 0.0

        for mzi in design.mzi_gates:
            p = self.power_for_phase(mzi.phase_shift_rad)
            total_mw += p
            gate_powers.append(
                {
                    "gate_id": mzi.gate_id,
                    "phase_rad": mzi.phase_shift_rad,
                    "power_mw": p,
                }
            )

        return {
            "gate_powers": gate_powers,
            "total_power_mw": total_mw,
            "n_gates": len(design.mzi_gates),
        }


# ══════════════════════════════════════════════════════════════════════
# Crosstalk Analyzer
# ══════════════════════════════════════════════════════════════════════


class CrosstalkAnalyzer:
    """Analyze inter-channel crosstalk in WDM systems.

    Parameters
    ----------
    adjacent_xt_db : float
        Adjacent-channel crosstalk (default -25 dB).
    """

    def __init__(self, adjacent_xt_db: float = -25.0) -> None:
        self._adjacent_xt_db = adjacent_xt_db

    def analyze(self, channels: list[WDMChannel]) -> Dict[str, Any]:
        """Analyze crosstalk between WDM channels.

        Returns
        -------
        dict
            ``worst_xt_db``, ``per_channel``, ``osnr_db``.
        """
        per_channel: list[Dict[str, Any]] = []
        worst_xt = -math.inf

        for i, ch in enumerate(channels):
            n_adj = sum(
                1
                for j, other in enumerate(channels)
                if i != j and abs(ch.wavelength_nm - other.wavelength_nm) < ch.bandwidth_nm * 3
            )
            xt = self._adjacent_xt_db + 10.0 * math.log10(max(n_adj, 1))
            osnr = ch.power_dbm - xt

            per_channel.append(
                {
                    "channel_id": ch.channel_id,
                    "signal": ch.signal_name,
                    "wavelength_nm": ch.wavelength_nm,
                    "n_adjacent": n_adj,
                    "crosstalk_db": xt,
                    "osnr_db": osnr,
                }
            )

            if xt > worst_xt:
                worst_xt = xt

        return {
            "per_channel": per_channel,
            "worst_xt_db": worst_xt if per_channel else 0.0,
            "n_channels": len(channels),
        }


# ══════════════════════════════════════════════════════════════════════
# Export Functions
# ══════════════════════════════════════════════════════════════════════


def export_photonic_json(design: PhotonicCircuitDesign, path: str) -> None:
    """Export photonic design to JSON.

    Parameters
    ----------
    design : PhotonicCircuitDesign
        The design to export.
    path : str
        Output file path.
    """
    data = {
        "name": design.name,
        "n_nodes": design.n_nodes,
        "total_area_um2": design.total_area_um2,
        "waveguides": [
            {
                "source": wg.source,
                "target": wg.target,
                "length_um": wg.length_um,
                "loss_db": wg.loss_db,
                "wavelength_nm": wg.wavelength_nm,
                "n_crossings": wg.n_crossings,
            }
            for wg in design.waveguides
        ],
        "mzi_gates": [
            {
                "gate_id": m.gate_id,
                "operation": m.operation,
                "phase_shift_rad": m.phase_shift_rad,
                "insertion_loss_db": m.insertion_loss_db,
            }
            for m in design.mzi_gates
        ],
        "wdm_channels": [
            {
                "channel_id": ch.channel_id,
                "wavelength_nm": ch.wavelength_nm,
                "signal": ch.signal_name,
            }
            for ch in design.wdm_channels
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def visualize_photonic(design: PhotonicCircuitDesign) -> str:
    """Generate ASCII visualization of a photonic design.

    Returns
    -------
    str
        Multi-line ASCII representation.
    """
    lines: list[str] = [
        f"┌{'=' * 56}┐",
        f"│ Photonic NoC: {design.name:<39} │",
        f"│ Nodes: {design.n_nodes:<4}  WGs: {len(design.waveguides):<4}"
        f"  MZIs: {len(design.mzi_gates):<4}  WDM: {len(design.wdm_channels):<3} │",
        f"│ Area: {design.total_area_um2:>10.0f} μm²"
        f"  ({design.total_area_um2 * 1e-6:>6.3f} mm²)           │",
        f"└{'=' * 56}┘",
        "",
    ]

    if design.waveguides:
        lines.append("  Waveguides:")
        for wg in design.waveguides[:10]:
            arrow = f"  [{wg.source}] ──── [{wg.target}]"
            lines.append(f"    {arrow:<20} L={wg.length_um:>6.0f}μm  loss={wg.loss_db:>5.2f}dB")
        if len(design.waveguides) > 10:
            lines.append(f"    ... and {len(design.waveguides) - 10} more")

    if design.mzi_gates:
        lines.append("")
        lines.append("  MZI Gates:")
        for m in design.mzi_gates[:10]:
            lines.append(
                f"    {m.gate_id:<20} op={m.operation:<5}"
                f" φ={m.phase_shift_rad:>5.2f}rad  IL={m.insertion_loss_db:.1f}dB"
            )

    return "\n".join(lines)
