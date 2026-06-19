# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic Photonic Co-Design Loop

"""Stochastic bitstream to photonic co-design orchestration.

This module binds the existing SC bitstream encoder, photonic NoC compiler,
optical pulse compiler, and FDTD smoke simulation into a single evidence
surface. It intentionally emits feasibility blockers instead of silently
claiming tape-out readiness when a design lacks margin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sc_neurocore.edge.bitstream import popcount_slice, probability, scc
from sc_neurocore.edge.lfsr import Lfsr16
from sc_neurocore.optics.photonic_emitter import (
    CompilationResult,
    PhotonicCompiler,
    PhotonicTarget,
)

from .photonic_noc import (
    CrosstalkAnalyzer,
    PhotonicCircuitDesign,
    PowerBudgetAnalyzer,
    SCToPhotonic,
)


@dataclass(frozen=True)
class PhotonicCoDesignConfig:
    """Configuration for a reproducible stochastic photonic design pass."""

    bitstream_length: int = 1024
    seed: int = 0xACE1
    density_alpha: float = 0.01
    min_power_margin_db: float = 3.0
    max_crosstalk_db: float = -18.0
    run_fdtd: bool = True
    fdtd_steps: int = 32
    target: PhotonicTarget = field(default_factory=PhotonicTarget.silicon_photonics)

    def __post_init__(self) -> None:
        if self.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive.")
        if not 0.0 < self.density_alpha < 1.0:
            raise ValueError("density_alpha must be in the open interval (0, 1).")
        if self.fdtd_steps < 0:
            raise ValueError("fdtd_steps must be non-negative.")


@dataclass(frozen=True)
class BitstreamEvidence:
    """One encoded SC channel and its statistical evidence."""

    name: str
    target_probability: float
    packed_words: tuple[int, ...]
    bit_length: int
    popcount: int
    measured_probability: float
    density_error: float
    transitions: int

    def to_json(self) -> dict[str, Any]:
        """Return a compact JSON-ready evidence record."""
        return {
            "name": self.name,
            "target_probability": self.target_probability,
            "bit_length": self.bit_length,
            "popcount": self.popcount,
            "measured_probability": self.measured_probability,
            "density_error": self.density_error,
            "transitions": self.transitions,
            "packed_word_count": len(self.packed_words),
        }


@dataclass(frozen=True)
class PhotonicCoDesignReport:
    """Complete output of a stochastic photonic co-design pass."""

    name: str
    design: PhotonicCircuitDesign
    bitstreams: tuple[BitstreamEvidence, ...]
    optical_results: tuple[CompilationResult, ...]
    power_budget: dict[str, Any]
    crosstalk: dict[str, Any]
    scc_matrix: tuple[tuple[float, ...], ...]
    density_tolerance: float
    fdtd: dict[str, Any]
    layout_manifest: dict[str, Any]
    feasible: bool
    blockers: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready report."""
        return {
            "name": self.name,
            "feasible": self.feasible,
            "blockers": list(self.blockers),
            "density_tolerance": self.density_tolerance,
            "design": {
                "name": self.design.name,
                "n_nodes": self.design.n_nodes,
                "n_waveguides": len(self.design.waveguides),
                "n_mzi_gates": len(self.design.mzi_gates),
                "n_wdm_channels": len(self.design.wdm_channels),
                "total_area_um2": self.design.total_area_um2,
            },
            "bitstreams": [entry.to_json() for entry in self.bitstreams],
            "optical_results": [
                {
                    "target": result.target,
                    "num_modulators": result.num_modulators,
                    "optical_power_mean_mw": result.optical_power_mean_mw,
                    "phase_coverage_rad": result.phase_coverage_rad,
                    "fdtd_energy": result.fdtd_energy,
                }
                for result in self.optical_results
            ],
            "power_budget": self.power_budget,
            "crosstalk": self.crosstalk,
            "scc_matrix": [list(row) for row in self.scc_matrix],
            "fdtd": self.fdtd,
            "layout_manifest": self.layout_manifest,
        }

    def export_json(self, path: str | Path) -> None:
        """Write the report to a JSON file."""
        Path(path).write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8"
        )


def _unpack_words(words: tuple[int, ...], bit_length: int) -> npt.NDArray[np.uint8]:
    bits = np.zeros(bit_length, dtype=np.uint8)
    for i in range(bit_length):
        bits[i] = (words[i // 32] >> (i % 32)) & 1
    return bits


def _transition_count(bits: npt.NDArray[np.uint8]) -> int:
    if bits.size <= 1:
        return 0
    return int(np.count_nonzero(bits[1:] != bits[:-1]))


def _density_tolerance(bitstream_length: int, alpha: float) -> float:
    """Two-sided Hoeffding density tolerance for a Bernoulli bitstream."""
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * bitstream_length))


def derive_probabilities_from_adjacency(
    adjacency: npt.ArrayLike,
    floor: float = 1.0 / 65535.0,
    ceiling: float = 1.0 - 1.0 / 65535.0,
) -> npt.NDArray[np.float64]:
    """Derive per-node SC probabilities from inbound absolute weight mass."""
    matrix = np.asarray(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    if matrix.shape[0] == 0:
        raise ValueError("adjacency must contain at least one node.")
    inbound = np.sum(np.abs(matrix), axis=0)
    scale = float(np.max(inbound))
    if scale <= 0.0:
        return np.full(matrix.shape[0], 0.5, dtype=np.float64)
    scaled: npt.NDArray[np.float64] = np.clip(inbound / scale, floor, ceiling)
    return scaled


def encode_bitstream_bank(
    probabilities: npt.ArrayLike,
    *,
    bitstream_length: int,
    seed: int,
    names: list[str] | None = None,
) -> tuple[BitstreamEvidence, ...]:
    """Encode probabilities into deterministic LFSR-backed SC bitstreams."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional array.")
    if bitstream_length <= 0:
        raise ValueError("bitstream_length must be positive.")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must lie in [0, 1].")
    labels = names or [f"ch{i}" for i in range(probs.shape[0])]
    if len(labels) != probs.shape[0]:
        raise ValueError("names length must match probabilities length.")

    evidence: list[BitstreamEvidence] = []
    for idx, (label, prob) in enumerate(zip(labels, probs)):
        channel_seed = (seed + (idx + 1) * 0x9E37) & 0xFFFF
        words = tuple(Lfsr16(channel_seed).encode_float(float(prob), bitstream_length))
        measured = probability(list(words), bitstream_length)
        bits = _unpack_words(words, bitstream_length)
        evidence.append(
            BitstreamEvidence(
                name=label,
                target_probability=float(prob),
                packed_words=words,
                bit_length=bitstream_length,
                popcount=popcount_slice(list(words)),
                measured_probability=measured,
                density_error=abs(measured - float(prob)),
                transitions=_transition_count(bits),
            )
        )
    return tuple(evidence)


def _scc_matrix(bitstreams: tuple[BitstreamEvidence, ...]) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for left in bitstreams:
        row: list[float] = []
        for right in bitstreams:
            row.append(scc(list(left.packed_words), list(right.packed_words), left.bit_length))
        rows.append(tuple(row))
    return tuple(rows)


def _layout_manifest(design: PhotonicCircuitDesign) -> dict[str, Any]:
    """Create a PDA handoff manifest without claiming foundry DRC/LVS."""
    return {
        "format": "sc-neurocore-photonic-layout-v1",
        "gdsii_status": "handoff_manifest_only",
        "requires_external_pda": ["PDK layer map", "DRC deck", "LVS deck", "FDTD sign-off"],
        "cells": [
            {
                "name": gate.gate_id,
                "type": "MZI",
                "operation": gate.operation,
                "arm_length_um": gate.arm_length_um,
                "phase_shift_rad": gate.phase_shift_rad,
            }
            for gate in design.mzi_gates
        ],
        "routes": [
            {
                "source": segment.source,
                "target": segment.target,
                "length_um": segment.length_um,
                "loss_db": segment.loss_db,
                "n_crossings": segment.n_crossings,
            }
            for segment in design.waveguides
        ],
        "wdm_channels": [
            {
                "channel_id": channel.channel_id,
                "wavelength_nm": channel.wavelength_nm,
                "bandwidth_nm": channel.bandwidth_nm,
                "signal": channel.signal_name,
            }
            for channel in design.wdm_channels
        ],
    }


class StochasticPhotonicCoDesignLoop:
    """End-to-end stochastic bitstream, photonic NoC, and FDTD loop."""

    def __init__(self, config: PhotonicCoDesignConfig | None = None) -> None:
        self.config = config or PhotonicCoDesignConfig()

    def compile(
        self,
        adjacency: npt.ArrayLike,
        *,
        probabilities: npt.ArrayLike | None = None,
        node_labels: list[str] | None = None,
        gate_specs: list[dict[str, Any]] | None = None,
        name: str = "sc_photonic_codesign",
    ) -> PhotonicCoDesignReport:
        """Run the full co-design loop for one SC connectivity matrix."""
        matrix = np.asarray(adjacency, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("adjacency must be a square matrix.")
        if matrix.shape[0] == 0:
            raise ValueError("adjacency must contain at least one node.")
        labels = node_labels or [f"pe{i}" for i in range(matrix.shape[0])]
        if len(labels) != matrix.shape[0]:
            raise ValueError("node_labels length must match adjacency size.")

        probs = (
            derive_probabilities_from_adjacency(matrix)
            if probabilities is None
            else np.asarray(probabilities, dtype=np.float64)
        )
        if probs.shape != (matrix.shape[0],):
            raise ValueError("probabilities must match the adjacency node count.")

        bitstreams = encode_bitstream_bank(
            probs,
            bitstream_length=self.config.bitstream_length,
            seed=self.config.seed,
            names=labels,
        )
        tolerance = _density_tolerance(self.config.bitstream_length, self.config.density_alpha)

        design = SCToPhotonic().compile(
            matrix,
            node_labels=labels,
            gate_specs=gate_specs,
            name=name,
        )
        optical_compiler = PhotonicCompiler(self.config.target)
        optical_results: list[CompilationResult] = []
        representative_idx = max(
            range(len(bitstreams)), key=lambda idx: bitstreams[idx].transitions
        )
        for idx, entry in enumerate(bitstreams):
            bits = _unpack_words(entry.packed_words, entry.bit_length)
            optical_results.append(
                optical_compiler.compile_bitstream(
                    bits,
                    run_fdtd=self.config.run_fdtd and idx == representative_idx,
                    fdtd_steps=self.config.fdtd_steps,
                )
            )

        power_budget = PowerBudgetAnalyzer().analyze(design)
        crosstalk = CrosstalkAnalyzer().analyze(design.wdm_channels)
        scc_rows = _scc_matrix(bitstreams)
        fdtd = {
            "enabled": self.config.run_fdtd,
            "representative_channel": bitstreams[representative_idx].name,
            "steps": self.config.fdtd_steps,
            "energy": optical_results[representative_idx].fdtd_energy,
            "target": self.config.target.name,
            "modulation": self.config.target.modulation.value,
        }
        layout = _layout_manifest(design)

        blockers = self._collect_blockers(
            bitstreams=bitstreams,
            tolerance=tolerance,
            power_budget=power_budget,
            crosstalk=crosstalk,
            fdtd=fdtd,
        )
        return PhotonicCoDesignReport(
            name=name,
            design=design,
            bitstreams=bitstreams,
            optical_results=tuple(optical_results),
            power_budget=power_budget,
            crosstalk=crosstalk,
            scc_matrix=scc_rows,
            density_tolerance=tolerance,
            fdtd=fdtd,
            layout_manifest=layout,
            feasible=not blockers,
            blockers=tuple(blockers),
        )

    def _collect_blockers(
        self,
        *,
        bitstreams: tuple[BitstreamEvidence, ...],
        tolerance: float,
        power_budget: dict[str, Any],
        crosstalk: dict[str, Any],
        fdtd: dict[str, Any],
    ) -> list[str]:
        blockers: list[str] = []
        failing_density = [entry.name for entry in bitstreams if entry.density_error > tolerance]
        if failing_density:
            blockers.append(
                "bitstream density outside Hoeffding tolerance: " + ", ".join(failing_density)
            )
        if power_budget["n_failed"] > 0:
            blockers.append(f"{power_budget['n_failed']} optical paths are below detector margin")
        if power_budget["worst_margin_db"] < self.config.min_power_margin_db:
            blockers.append(
                f"worst optical margin {power_budget['worst_margin_db']:.3f} dB "
                f"is below {self.config.min_power_margin_db:.3f} dB"
            )
        if crosstalk["n_channels"] > 1 and crosstalk["worst_xt_db"] > self.config.max_crosstalk_db:
            blockers.append(
                f"worst crosstalk {crosstalk['worst_xt_db']:.3f} dB "
                f"exceeds {self.config.max_crosstalk_db:.3f} dB"
            )
        if fdtd["enabled"] and fdtd["energy"] <= 0.0:
            blockers.append("FDTD representative pulse produced zero field energy")
        return blockers


__all__ = [
    "BitstreamEvidence",
    "PhotonicCoDesignConfig",
    "PhotonicCoDesignReport",
    "StochasticPhotonicCoDesignLoop",
    "derive_probabilities_from_adjacency",
    "encode_bitstream_bank",
]
