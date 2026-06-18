# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automatic FPGA resource optimizer

"""Automatically compress an SNN to fit a target FPGA.

Takes a network that exceeds the FPGA resource budget and iteratively
applies pruning, quantization, and bitstream length reduction until
the energy estimator says it fits.

Connects compression + adaptive precision + energy estimator into
a single command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sc_neurocore.compression.pruning import prune_weights
from sc_neurocore.compression.quantization import quantize_weights
from sc_neurocore.energy.estimator import estimate


@dataclass
class OptimizationStep:
    """One step in the optimization process."""

    action: str
    luts_before: int
    luts_after: int
    sparsity: float
    bitstream_length: int


@dataclass
class OptimizationResult:
    """Result of the resource optimization process."""

    fits: bool
    target: str
    final_luts: int
    target_luts: int
    utilization_pct: float
    final_bitstream_length: int
    final_sparsity: float
    steps: list[OptimizationStep] = field(default_factory=list)
    optimized_weights: list[np.ndarray[Any, Any]] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        """Render a multi-line human-readable report of the optimization outcome.

        Returns
        -------
        str
            One line for the target verdict, LUT utilisation, bitstream length,
            sparsity and step count, followed by one line per compression step.
        """
        lines = [
            f"Resource Optimization: {self.target}",
            f"  Fits: {'YES' if self.fits else 'NO'}",
            f"  LUTs: {self.final_luts:,} / {self.target_luts:,} ({self.utilization_pct:.1f}%)",
            f"  Bitstream length: {self.final_bitstream_length}",
            f"  Sparsity: {self.final_sparsity:.1%}",
            f"  Steps taken: {len(self.steps)}",
        ]
        for s in self.steps:
            lines.append(f"    {s.action}: {s.luts_before:,} -> {s.luts_after:,} LUTs")
        return "\n".join(lines)


def fit_to_target(
    layer_sizes: list[tuple[int, int]],
    weights: list[np.ndarray[Any, Any]],
    target: str = "ice40",
    max_iterations: int = 10,
    min_bitstream_length: int = 32,
    initial_bitstream_length: int = 256,
) -> OptimizationResult:
    """Automatically compress an SNN to fit a target FPGA.

    Iteratively applies:
    1. Bitstream length reduction (halving L)
    2. Weight pruning (increasing threshold)
    3. Weight quantization (reducing bit width)

    Stops when the energy estimator says the network fits on the target.

    Parameters
    ----------
    layer_sizes : list of (n_inputs, n_neurons)
    weights : list of ndarray
    target : str
        FPGA target ('ice40', 'ecp5', 'artix7', 'zynq').
    max_iterations : int
        Maximum optimization steps.
    min_bitstream_length : int
        Minimum allowed L.
    initial_bitstream_length : int
        Starting bitstream length.

    Returns
    -------
    OptimizationResult
    """
    from sc_neurocore.energy.fpga_models import TARGETS

    target_info = TARGETS.get(target)
    if target_info is None:
        raise ValueError(f"Unknown target '{target}'")

    current_weights = [w.copy() for w in weights]
    current_L = initial_bitstream_length
    steps = []
    prune_threshold = 0.001
    quant_bits = 16

    for iteration in range(max_iterations):
        report = estimate(layer_sizes, target=target, bitstream_length=current_L)

        if report.fits_on_target:
            break

        luts_before = report.total_luts

        # Strategy 1: Halve bitstream length
        if current_L > min_bitstream_length:
            current_L = max(current_L // 2, min_bitstream_length)
            report_after = estimate(layer_sizes, target=target, bitstream_length=current_L)
            steps.append(
                OptimizationStep(
                    action=f"Reduce L to {current_L}",
                    luts_before=luts_before,
                    luts_after=report_after.total_luts,
                    sparsity=0.0,
                    bitstream_length=current_L,
                )
            )
            if report_after.fits_on_target:  # pragma: no cover
                break
            continue

        # Strategy 2: Prune weights
        prune_threshold *= 3
        current_weights, prune_report = prune_weights(current_weights, threshold=prune_threshold)
        report_after = estimate(layer_sizes, target=target, bitstream_length=current_L)
        steps.append(
            OptimizationStep(
                action=f"Prune threshold={prune_threshold:.3f}",
                luts_before=luts_before,
                luts_after=report_after.total_luts,
                sparsity=prune_report.sparsity,
                bitstream_length=current_L,
            )
        )

        # Strategy 3: Reduce quantization
        if quant_bits > 4:
            quant_bits = max(quant_bits - 2, 4)
            current_weights = quantize_weights(current_weights, bits=quant_bits)
            steps.append(
                OptimizationStep(
                    action=f"Quantize to {quant_bits}-bit",
                    luts_before=report_after.total_luts,
                    luts_after=report_after.total_luts,
                    sparsity=prune_report.sparsity,
                    bitstream_length=current_L,
                )
            )

    # Final estimate
    final_report = estimate(layer_sizes, target=target, bitstream_length=current_L)

    total_params = sum(w.size for w in current_weights)
    nonzero = sum(np.count_nonzero(w) for w in current_weights)
    sparsity = 1.0 - nonzero / max(total_params, 1)

    return OptimizationResult(
        fits=final_report.fits_on_target,
        target=target,
        final_luts=final_report.total_luts,
        target_luts=target_info.total_luts,
        utilization_pct=final_report.utilization_pct,
        final_bitstream_length=current_L,
        final_sparsity=sparsity,
        steps=steps,
        optimized_weights=current_weights,
    )
