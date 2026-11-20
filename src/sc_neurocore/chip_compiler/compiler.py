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
        """Return a plain-text report: the status, the mapping totals, then each violation and warning.

        Returns
        -------
        str
            One line per item, the violations and warnings marked ``[VIOLATION]`` and ``[WARNING]``.
        """
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Compilation [{self.chip}]: {status}",
            f"  Cores: {self.total_cores_used}",
            f"  Neurons: {self.total_neurons_mapped}",
            f"  Weight precision: {self.weight_bits}-bit",
        ]
        for v in self.violations:
            lines.append(f"  [VIOLATION] {v}")
        for w in self.warnings:
            lines.append(f"  [WARNING] {w}")
        return "\n".join(lines)


def _quantise(w: np.ndarray[Any, Any], bits: int) -> np.ndarray[Any, Any]:
    """Return ``w`` on the signed grid of ``bits`` bits; ``w`` is finite and not empty, and ``bits`` is at least 1.

    With two or more bits the grid is symmetric: ``2**(bits-1) - 1`` levels on each side of zero, a step of
    ``max|w| / (2**(bits-1) - 1)``, and values clipped to ``max|w|``. One bit codes only a sign, so each weight
    becomes ``+a`` or ``-a`` (zero takes ``+a``), where ``a`` is the mean magnitude: for a fixed sign pattern this
    level minimises the squared error, the binary-weight scale of Rastegari et al., ECCV 2016 (XNOR-Net).

    The compiler assumes signed weights with one level per matrix. A chip whose 1-bit synapse means connectivity
    (``{0, +w}``) needs its own rule; ``+a`` and ``-a`` say nothing about such a chip.
    """
    if bits == 1:
        level = float(np.mean(np.abs(w)))
        signed: np.ndarray[Any, Any] = np.where(w >= 0, level, -level)
        return signed
    abs_max = max(float(np.abs(w).max()), 1e-8)
    scale = abs_max / (2 ** (bits - 1) - 1)
    gridded: np.ndarray[Any, Any] = np.clip(np.round(w / scale) * scale, -abs_max, abs_max)
    return gridded


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
        Input and neuron count of each layer.
    weights : list of ndarray, optional
        Weight matrices per layer. If provided, they are quantised to the core's weight precision: a symmetric grid
        for two or more bits, and plus or minus the mean magnitude for one bit. A precision below one bit, or a
        matrix that is empty or holds NaN or infinity, is a violation, and then no matrix is quantised.
    neuron_types : list of str, optional
        Neuron type per layer (e.g., 'LIF', 'Izhikevich').
    target : str or ChipSpec
        Target chip name or spec.

    Returns
    -------
    CompilationResult
        The mapping, the quantised weights, and every violation and warning; ``success`` is true only without
        violations.
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

    # Quantize weights: nothing is quantised when the precision or any weight matrix cannot be.
    if weights is not None:
        bits = chip.core.weight_bits
        arrays = [np.asarray(w, dtype=float) for w in weights]
        refusals = []
        if bits < 1:
            refusals.append(
                f"{chip.name} weight precision is {bits} bits; a weight needs at least 1 bit"
            )
        for i, w in enumerate(arrays):
            if w.size == 0:
                refusals.append(f"Weights of layer {i} are empty")
            elif not np.all(np.isfinite(w)):
                refusals.append(f"Weights of layer {i} are not finite (NaN or infinity)")
        violations.extend(refusals)
        quantized = []
        for w in arrays if not refusals else []:
            q = _quantise(w, bits)
            quantized.append(q)

            # Warn about precision loss
            mse = float(np.mean((w - q) ** 2))
            if mse > 0.01 * float(np.mean(w**2)):
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
