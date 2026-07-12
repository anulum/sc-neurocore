# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary runtime and deployment safety contracts

"""Bound mutation resources and record seeded runtime-fault decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism
from sc_neurocore.fault_injection import DegradationPlan, FaultModel


@dataclass(frozen=True)
class RuntimeFaultConfig:
    """Runtime fault-check settings for evolved SC organisms."""

    fault_model: FaultModel = FaultModel.BIT_FLIP
    ber: float = 0.0
    seed_offset: int = 0
    sample_neurons: int = 8
    fitness_penalty_on_extend: float = 0.95
    fitness_penalty_on_replay: float = 0.85


@dataclass(frozen=True)
class RuntimeFaultCheck:
    """Recorded runtime fault/degradation decision for one organism."""

    organism_id: str
    generation: int
    action: str
    recommended_bitstream_length: int
    replay_seed: int
    affected_ratio: float
    audit_status: str
    reason: str

    @classmethod
    def from_plan(cls, organism: Organism, plan: DegradationPlan) -> RuntimeFaultCheck:
        """Capture a degradation plan against the organism's current identity."""
        return cls(
            organism_id=organism.genome.genome_id,
            generation=organism.genome.generation,
            action=plan.action.value,
            recommended_bitstream_length=plan.recommended_bitstream_length,
            replay_seed=plan.replay_seed,
            affected_ratio=plan.observation.affected_ratio,
            audit_status=plan.observation.audit.status.value,
            reason=plan.reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready fault-check summary."""
        return {
            "organism_id": self.organism_id,
            "generation": self.generation,
            "action": self.action,
            "recommended_bitstream_length": self.recommended_bitstream_length,
            "replay_seed": self.replay_seed,
            "affected_ratio": self.affected_ratio,
            "audit_status": self.audit_status,
            "reason": self.reason,
        }


@dataclass
class SafetyBounds:
    """Constrains the mutation space to prevent runaway replication.

    Enforces hard limits on genome parameters that could cause
    resource exhaustion or unsafe behaviour on FPGA tiles.
    """

    max_neurons: int = 1024
    min_neurons: int = 4
    max_layers: int = 16
    max_bitstream: int = 4096
    min_bitstream: int = 32
    max_connectivity: float = 1.0
    max_tau_deep: float = 100000.0
    max_replications_per_gen: int = 64

    def clamp(self, genome: Genome) -> Genome:
        """Clamp mutable genome fields to configured deployment bounds."""
        genome.topology.num_neurons = int(
            np.clip(genome.topology.num_neurons, self.min_neurons, self.max_neurons)
        )
        genome.topology.num_layers = min(self.max_layers, max(1, genome.topology.num_layers))
        genome.topology.bitstream_length = int(
            np.clip(genome.topology.bitstream_length, self.min_bitstream, self.max_bitstream)
        )
        genome.topology.connectivity = float(
            np.clip(genome.topology.connectivity, 0.01, self.max_connectivity)
        )
        genome.neuron.tau_deep = min(self.max_tau_deep, genome.neuron.tau_deep)
        return genome

    def is_within_bounds(self, genome: Genome) -> bool:
        """Return whether topology dimensions fit the configured bounds."""
        return (
            self.min_neurons <= genome.topology.num_neurons <= self.max_neurons
            and 1 <= genome.topology.num_layers <= self.max_layers
            and self.min_bitstream <= genome.topology.bitstream_length <= self.max_bitstream
        )


@dataclass
class ResourceBudget:
    """Per-organism resource constraints."""

    max_area_um2: float = 1e6
    max_power_mw: float = 100.0
    max_neurons: int = 1024

    def check(self, genome: Genome) -> Tuple[bool, List[str]]:
        """Return budget compliance and human-readable resource violations."""
        violations = []
        if genome.topology.num_neurons > self.max_neurons:
            violations.append(f"neurons={genome.topology.num_neurons}>{self.max_neurons}")
        est_area = genome.topology.num_neurons * genome.topology.bitstream_length * 0.1
        if est_area > self.max_area_um2:
            violations.append(f"area={est_area:.0f}>{self.max_area_um2:.0f}")
        return (len(violations) == 0, violations)


@dataclass
class SafetyCheckResult:
    """Result of a formal safety check on emitted Verilog/NIR."""

    genome_id: str
    passed: bool
    violations: List[str] = field(default_factory=list)
    neuron_count_ok: bool = True
    connectivity_ok: bool = True
    bitstream_ok: bool = True


class FormalSafetyGuard:
    """Validates emitted organisms against safety constraints before deployment.

    Links to the safety_cert module for IEC 61508 compliance.
    """

    def __init__(self, bounds: Optional[SafetyBounds] = None) -> None:
        self.bounds = bounds or SafetyBounds()
        self.checked: int = 0
        self.rejected: int = 0

    def check(self, genome: Genome) -> SafetyCheckResult:
        """Evaluate topology limits and update cumulative rejection counters."""
        self.checked += 1
        violations = []
        n_ok = genome.topology.num_neurons <= self.bounds.max_neurons
        c_ok = genome.topology.connectivity <= self.bounds.max_connectivity
        b_ok = genome.topology.bitstream_length <= self.bounds.max_bitstream

        if not n_ok:
            violations.append(f"neurons={genome.topology.num_neurons}>{self.bounds.max_neurons}")
        if not c_ok:
            violations.append(
                f"connectivity={genome.topology.connectivity}>{self.bounds.max_connectivity}"
            )
        if not b_ok:
            violations.append(
                f"bitstream={genome.topology.bitstream_length}>{self.bounds.max_bitstream}"
            )

        passed = len(violations) == 0
        if not passed:
            self.rejected += 1

        return SafetyCheckResult(
            genome_id=genome.genome_id,
            passed=passed,
            violations=violations,
            neuron_count_ok=n_ok,
            connectivity_ok=c_ok,
            bitstream_ok=b_ok,
        )

    @property
    def rejection_rate(self) -> float:
        """Return rejected checks divided by all completed checks."""
        return self.rejected / self.checked if self.checked > 0 else 0.0


__all__ = [
    "FormalSafetyGuard",
    "ResourceBudget",
    "RuntimeFaultCheck",
    "RuntimeFaultConfig",
    "SafetyBounds",
    "SafetyCheckResult",
]
