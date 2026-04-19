# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared Core Types

"""Unified type definitions shared across Optimizer, NAS, and Runtime.

Eliminates duplicate LayerConfig / HardwareBudget definitions and provides
a single source of truth for resource estimation and constraint checking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class DecorrelationStrategy(Enum):
    """Decorrelation strategies for stochastic bitstreams."""

    NONE = "None"
    LFSR = "LFSR"
    SOBOL = "Sobol"
    HALTON = "Halton"
    SCC_DECORRELATOR = "SCC_Decorrelator"


class ComputeMode(Enum):
    """Compute mode for a layer."""

    SC = "SC"
    DETERMINISTIC = "Deterministic"
    HYBRID = "Hybrid"


class NeuronType(Enum):
    """Supported neuron models."""

    LIF = "LIF"
    IZHIKEVICH = "Izhikevich"
    ADEX = "AdEx"
    HH = "Hodgkin-Huxley"


@dataclass
class HardwareBudget:
    """Unified FPGA resource budget (shared by Optimizer, NAS, Runtime)."""

    max_luts: int = 500_000
    max_ffs: int = 500_000
    max_bram_kb: int = 2048
    max_dsp: int = 256
    max_power_mw: float = 5_000.0
    max_latency_cycles: int = 0

    def utilisation(self, luts: int, ffs: int = 0, bram: int = 0, dsp: int = 0) -> Dict[str, float]:
        return {
            "luts": luts / self.max_luts if self.max_luts else 0,
            "ffs": ffs / self.max_ffs if self.max_ffs else 0,
            "bram": bram / self.max_bram_kb if self.max_bram_kb else 0,
            "dsp": dsp / self.max_dsp if self.max_dsp else 0,
        }


@dataclass
class ResourceReport:
    """Unified resource consumption report."""

    total_luts: int = 0
    total_ffs: int = 0
    total_dsp: int = 0
    total_bram_kb: float = 0.0
    total_power_mw: float = 0.0
    total_latency_cycles: int = 0
    mean_accuracy: float = 0.0

    def meets_budget(self, budget: HardwareBudget) -> bool:
        if self.total_luts > budget.max_luts:
            return False
        if self.total_power_mw > budget.max_power_mw:
            return False
        if budget.max_latency_cycles > 0 and self.total_latency_cycles > budget.max_latency_cycles:
            return False
        if self.total_ffs > budget.max_ffs:
            return False
        if self.total_dsp > budget.max_dsp:
            return False
        return not self.total_bram_kb > budget.max_bram_kb

    def summary(self) -> str:
        return (
            f"LUTs: {self.total_luts}, FFs: {self.total_ffs}, "
            f"DSP: {self.total_dsp}, BRAM: {self.total_bram_kb:.1f} KB, "
            f"Power: {self.total_power_mw:.2f} mW, "
            f"Latency: {self.total_latency_cycles} cycles, "
            f"Accuracy: {self.mean_accuracy:.4f}"
        )


@dataclass
class LayerSpec:
    """Layer specification usable by Optimizer, NAS, and Runtime.

    This is the shared representation — each subsystem can extend it
    but all core resource estimation uses this base.
    """

    layer_id: str
    neurons: int = 64
    mac_count: int = 0
    bitstream_length: int = 256
    decorrelator: DecorrelationStrategy = DecorrelationStrategy.LFSR
    mode: ComputeMode = ComputeMode.SC
    neuron_type: NeuronType = NeuronType.LIF
    is_critical_path: bool = False

    def estimate_luts(self) -> int:
        """Unified LUT estimation shared across all subsystems."""
        if self.mode == ComputeMode.DETERMINISTIC:
            return max(self.mac_count, self.neurons) * 120

        base_macs = max(self.mac_count, self.neurons * 2)
        luts = base_macs * 2 + int(math.log2(max(1, self.bitstream_length))) * 5

        decorr_cost = {
            DecorrelationStrategy.SOBOL: base_macs * 15,
            DecorrelationStrategy.HALTON: base_macs * 12,
            DecorrelationStrategy.SCC_DECORRELATOR: base_macs * 8,
            DecorrelationStrategy.LFSR: 16,
        }.get(self.decorrelator, 0)
        luts += decorr_cost

        neuron_mult = {
            NeuronType.LIF: 1.0,
            NeuronType.IZHIKEVICH: 1.8,
            NeuronType.ADEX: 2.2,
            NeuronType.HH: 4.5,
        }.get(self.neuron_type, 1.0)

        return int(luts * neuron_mult)

    def estimate_power_mw(self) -> float:
        if self.mode == ComputeMode.DETERMINISTIC:
            return max(self.mac_count, self.neurons) * 0.5
        base = max(self.mac_count, self.neurons)
        return base * 0.01 * (self.bitstream_length / 256.0)

    def estimate_accuracy(self) -> float:
        if self.mode == ComputeMode.DETERMINISTIC:
            return 1.0
        length = max(1, self.bitstream_length)
        base = {
            DecorrelationStrategy.SOBOL: 1.0 - 1.0 / length,
            DecorrelationStrategy.HALTON: 1.0 - 1.2 / length,
            DecorrelationStrategy.SCC_DECORRELATOR: 1.0 - 1.5 / length,
            DecorrelationStrategy.LFSR: 1.0 - 1.0 / math.sqrt(length),
        }.get(self.decorrelator, 1.0 - 2.0 / math.sqrt(length))
        return max(0.1, min(1.0, base))


def estimate_network(layers: List[LayerSpec]) -> ResourceReport:
    """Estimate total resources for a network of layers."""
    return ResourceReport(
        total_luts=sum(l.estimate_luts() for l in layers),
        total_power_mw=sum(l.estimate_power_mw() for l in layers),
        total_latency_cycles=max((l.bitstream_length for l in layers), default=0),
        mean_accuracy=sum(l.estimate_accuracy() for l in layers) / max(len(layers), 1),
    )
