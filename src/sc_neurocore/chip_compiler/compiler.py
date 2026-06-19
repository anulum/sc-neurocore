# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-chip SNN compiler

"""Compile an SNN to a target neuromorphic chip.

Partitions the network into cores, checks constraints (neurons/core,
fan-out, weight precision, supported neuron types), quantizes weights,
and produces a deployment map. Reports constraint violations with
specific fix suggestions.

This is the foundation for the "GCC of neuromorphic computing" — one
compiler that targets all chips via pluggable chip specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chip_spec import BUILTIN_CHIPS, ChipSpec


@dataclass
class CoreMapping:
    """Mapping of neurons to one chip core."""

    core_id: int
    layer_index: int
    neuron_start: int
    neuron_end: int
    n_neurons: int


@dataclass
class CompilationResult:
    """Result of compiling an SNN to a chip target."""

    chip: str
    success: bool = False
    core_mappings: list[CoreMapping] = field(default_factory=list)
    total_cores_used: int = 0
    total_neurons_mapped: int = 0
    weight_bits: int = 0
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quantized_weights: list[np.ndarray[Any, Any]] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Compilation [{self.chip}]: {status}",
            f"  Cores: {self.total_cores_used}",
            f"  Neurons: {self.total_neurons_mapped}",
            f"  Weight precision: {self.weight_bits}-bit",
        ]
        for v in self.violations:  # pragma: no cover
            lines.append(f"  [VIOLATION] {v}")
        for w in self.warnings:  # pragma: no cover
            lines.append(f"  [WARNING] {w}")
        return "\n".join(lines)


def compile_for_chip(
    layer_sizes: list[tuple[int, int]],
    weights: list[np.ndarray[Any, Any]] | None = None,
    neuron_types: list[str] | None = None,
    target: str | ChipSpec = "loihi2",
) -> CompilationResult:
    """Compile an SNN to a target neuromorphic chip.

    Parameters
    ----------
    layer_sizes : list of (n_inputs, n_neurons)
    weights : list of ndarray, optional
        Weight matrices per layer. If provided, will be quantized.
    neuron_types : list of str, optional
        Neuron type per layer (e.g., 'LIF', 'Izhikevich').
    target : str or ChipSpec
        Target chip name or spec.

    Returns
    -------
    CompilationResult
    """
    if isinstance(target, str):
        chip = BUILTIN_CHIPS.get(target)
        if chip is None:
            return CompilationResult(
                chip=target,
                success=False,
                violations=[
                    f"Unknown chip target '{target}'. Available: {list(BUILTIN_CHIPS.keys())}"
                ],
            )
    else:
        chip = target

    result = CompilationResult(chip=chip.name, weight_bits=chip.core.weight_bits)
    violations = []
    warnings = []
    mappings = []

    total_neurons = sum(n for _, n in layer_sizes)
    result.total_neurons_mapped = total_neurons

    # Check total capacity
    if not chip.fits(total_neurons):
        violations.append(
            f"Network has {total_neurons} neurons but {chip.name} supports "
            f"max {chip.total_neurons} ({chip.total_cores} cores x "
            f"{chip.core.max_neurons} neurons/core)"
        )

    # Check neuron type compatibility
    if neuron_types is not None:
        for i, nt in enumerate(neuron_types):
            if nt not in chip.core.supported_neuron_types:
                violations.append(
                    f"Layer {i}: neuron type '{nt}' not supported on {chip.name}. "
                    f"Supported: {chip.core.supported_neuron_types}"
                )

    # Check fan-out per layer
    for i, (n_in, n_out) in enumerate(layer_sizes):
        if n_out > chip.max_fan_out:
            violations.append(
                f"Layer {i}: fan-out {n_out} exceeds {chip.name} max {chip.max_fan_out}"
            )

    # Partition into cores
    core_id = 0
    for i, (n_in, n_out) in enumerate(layer_sizes):
        neurons_remaining = n_out
        offset = 0
        while neurons_remaining > 0:
            batch = min(neurons_remaining, chip.core.max_neurons)
            mappings.append(
                CoreMapping(
                    core_id=core_id,
                    layer_index=i,
                    neuron_start=offset,
                    neuron_end=offset + batch,
                    n_neurons=batch,
                )
            )
            core_id += 1
            offset += batch
            neurons_remaining -= batch

    result.core_mappings = mappings
    result.total_cores_used = core_id

    if core_id > chip.total_cores:
        violations.append(f"Needs {core_id} cores but {chip.name} has {chip.total_cores}")

    # Quantize weights
    if weights is not None:
        bits = chip.core.weight_bits
        n_levels = 2**bits
        quantized = []
        for w in weights:
            abs_max = max(np.abs(w).max(), 1e-8)
            scale = abs_max / (n_levels // 2 - 1)
            q = np.round(w / scale) * scale
            q = np.clip(q, -abs_max, abs_max)
            quantized.append(q)

            # Warn about precision loss
            mse = float(np.mean((w - q) ** 2))
            if mse > 0.01 * float(np.mean(w**2)):  # pragma: no cover
                warnings.append(
                    f"Weight quantization to {bits}-bit introduces "
                    f"{mse / max(float(np.mean(w**2)), 1e-12) * 100:.1f}% relative error"
                )
        result.quantized_weights = quantized

    # Analog noise warning
    if chip.analog_noise_cv > 0.05:
        warnings.append(
            f"{chip.name} has {chip.analog_noise_cv:.0%} analog noise CV. "
            f"Use variation-aware training (digital_twin.FPGAMismatchModel)"
        )

    result.violations = violations
    result.warnings = warnings
    result.success = len(violations) == 0

    return result
