# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary fitness contracts and evaluation

"""Evaluate software and FPGA fitness for evolutionary genomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from sc_neurocore.evo_substrate.genome import Genome


class FitnessType(Enum):
    """Identify the objective exposed by a fitness evaluation."""

    ACCURACY = "accuracy"
    ENERGY = "energy"
    LATENCY = "latency"
    COMPOSITE = "composite"


@dataclass
class FitnessResult:
    """Fitness evaluation result."""

    genome_id: str
    accuracy: float = 0.0
    energy_score: float = 0.0
    latency_score: float = 0.0
    composite: float = 0.0

    def compute_composite(
        self, w_acc: float = 0.5, w_energy: float = 0.3, w_latency: float = 0.2
    ) -> float:
        """Update and return the weighted three-objective fitness.

        Parameters
        ----------
        w_acc, w_energy, w_latency
            Weights for accuracy, energy, and latency scores.
        """
        self.composite = (
            w_acc * self.accuracy + w_energy * self.energy_score + w_latency * self.latency_score
        )
        return self.composite


class FitnessEvaluator:
    """Evaluates organism fitness from simulation metrics."""

    def __init__(self, fitness_type: FitnessType = FitnessType.COMPOSITE) -> None:
        self.fitness_type = fitness_type

    def evaluate(self, genome: Genome, metrics: Dict[str, float]) -> FitnessResult:
        """Evaluate one genome from simulation metrics and hardware proxies.

        Parameters
        ----------
        genome
            Genome whose topology determines energy and latency proxies.
        metrics
            Simulation metrics; the optional accuracy key is in [0, 1].
        """
        result = FitnessResult(genome_id=genome.genome_id)
        result.accuracy = metrics.get("accuracy", 0.0)

        # Energy: fewer neurons + shorter bitstreams = better
        neuron_pen = min(genome.topology.num_neurons / 1024.0, 1.0)
        bs_pen = min(genome.topology.bitstream_length / 1024.0, 1.0)
        result.energy_score = max(0.0, 1.0 - 0.5 * neuron_pen - 0.5 * bs_pen)

        # Latency: fewer layers = faster
        result.latency_score = max(0.0, 1.0 - genome.topology.num_layers / 10.0)

        result.compute_composite()
        return result


@dataclass
class HWFitnessReport:
    """Fitness feedback from actual FPGA execution."""

    genome_id: str
    fpga_cycles: int = 0
    fpga_power_mw: float = 0.0
    fpga_accuracy: float = 0.0
    fmax_mhz: float = 0.0
    timing_met: bool = True

    @property
    def hw_composite(self) -> float:
        """Return the weighted accuracy, clock, and timing-closure score."""
        time_score = min(1.0, 100.0 / max(1.0, self.fmax_mhz)) if self.fmax_mhz > 0 else 0.0
        return (
            0.5 * self.fpga_accuracy
            + 0.3 * (1.0 - time_score)
            + 0.2 * (1.0 if self.timing_met else 0.0)
        )


class HWFitnessCollector:
    """Collects HW fitness from deployed organisms."""

    def __init__(self) -> None:
        self.reports: Dict[str, HWFitnessReport] = {}

    def submit(self, report: HWFitnessReport) -> None:
        """Store the latest FPGA fitness report for one genome."""
        self.reports[report.genome_id] = report

    def get(self, genome_id: str) -> Optional[HWFitnessReport]:
        """Return the latest report for the genome, if submitted."""
        return self.reports.get(genome_id)

    @property
    def total_reports(self) -> int:
        """Return the number of genomes with submitted FPGA reports."""
        return len(self.reports)


__all__ = [
    "FitnessEvaluator",
    "FitnessResult",
    "FitnessType",
    "HWFitnessCollector",
    "HWFitnessReport",
]
