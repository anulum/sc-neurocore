# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated Safety & Regulatory Certification Generator

"""Static worst-case execution-time path analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class WCETPath:
    """Worst-case execution time for one SC computation path."""

    path_id: str
    description: str
    stages: List[str]
    cycles_per_stage: List[int]

    def __post_init__(self) -> None:
        """Validate one formula-derived timing path."""
        if not isinstance(self.path_id, str) or not self.path_id.strip():
            raise ValueError("path_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.stages, list):
            raise ValueError("stages must be a list")
        if not isinstance(self.cycles_per_stage, list):
            raise ValueError("cycles_per_stage must be a list")
        if len(self.stages) != len(self.cycles_per_stage):
            raise ValueError("stages and cycles_per_stage must have the same length")
        if not self.stages:
            raise ValueError("stages must not be empty")
        for stage in self.stages:
            if not isinstance(stage, str) or not stage.strip():
                raise ValueError("stages must contain non-empty strings")
        for cycles in self.cycles_per_stage:
            if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 0:
                raise ValueError("cycles_per_stage must contain non-negative integers")

    @property
    def total_cycles(self) -> int:
        """Return the sum of validated stage-cycle counts."""
        for cycles in self.cycles_per_stage:
            if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 0:
                raise ValueError("cycles_per_stage must contain non-negative integers")
        return sum(self.cycles_per_stage)

    def wcet_ns(self, clock_mhz: float) -> float:
        """Convert cycles to nanoseconds at a caller-supplied clock."""
        if isinstance(clock_mhz, bool) or not isinstance(clock_mhz, int | float):
            raise ValueError("clock_mhz must be a finite positive number")
        if not math.isfinite(float(clock_mhz)) or float(clock_mhz) <= 0.0:
            raise ValueError("clock_mhz must be a finite positive number")
        return self.total_cycles * 1000.0 / clock_mhz


class WCETAnalyzer:
    """Formula model for SC pipeline cycle counts.

    This is not a synthesis timing report or a measured hardware bound. It uses
    the following caller-reviewable stage assumptions:
    - LFSR encoding: bitstream_length cycles
    - Dot product: num_inputs cycles
    - LIF evaluation: fixed (3 cycles)
    - AER encoding: num_neurons worst-case
    - STP update: 1 cycle
    """

    LFSR_OVERHEAD = 1
    DOT_PRODUCT_PER_INPUT = 1
    LIF_FIXED = 3
    AER_PER_NEURON = 1
    STP_FIXED = 1

    @classmethod
    def analyze(
        cls,
        bitstream_length: int,
        num_inputs: int,
        num_neurons: int,
        has_stp: bool = False,
    ) -> WCETPath:
        """Build a single-layer path from explicit dimension assumptions."""
        for value, field_name, minimum in (
            (bitstream_length, "bitstream_length", 1),
            (num_inputs, "num_inputs", 1),
            (num_neurons, "num_neurons", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field_name} must be an integer >= {minimum}")
        if not isinstance(has_stp, bool):
            raise ValueError("has_stp must be a boolean")
        stages = ["LFSR_Encode", "DotProduct", "LIF_Eval", "AER_Encode"]
        cycles = [
            bitstream_length * cls.LFSR_OVERHEAD,
            num_inputs * cls.DOT_PRODUCT_PER_INPUT,
            cls.LIF_FIXED,
            num_neurons * cls.AER_PER_NEURON,
        ]
        if has_stp:
            stages.append("STP_Update")
            cycles.append(cls.STP_FIXED)
        return WCETPath(
            path_id="sc_inference",
            description="Full SC inference pipeline",
            stages=stages,
            cycles_per_stage=cycles,
        )

    @classmethod
    def analyze_multistage(
        cls,
        layers: List[Dict[str, int]],
    ) -> WCETPath:
        """Analyze a multi-layer SC network."""
        if not isinstance(layers, list) or not layers:
            raise ValueError("layers must be a non-empty list")
        stages = []
        cycles = []
        for i, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError("each layer must be a dictionary")
            bs = layer.get("bitstream_length", 256)
            ni = layer.get("num_inputs", 8)
            nn = layer.get("num_neurons", 16)
            for value, field_name, minimum in (
                (bs, f"layers[{i}].bitstream_length", 1),
                (ni, f"layers[{i}].num_inputs", 1),
                (nn, f"layers[{i}].num_neurons", 1),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"{field_name} must be an integer >= {minimum}")
            stages.extend([f"L{i}_LFSR", f"L{i}_Dot", f"L{i}_LIF", f"L{i}_AER"])
            cycles.extend(
                [
                    bs * cls.LFSR_OVERHEAD,
                    ni * cls.DOT_PRODUCT_PER_INPUT,
                    cls.LIF_FIXED,
                    nn * cls.AER_PER_NEURON,
                ]
            )
        return WCETPath("sc_network", "Multi-layer SC network", stages, cycles)


__all__ = [
    "WCETPath",
    "WCETAnalyzer",
]
