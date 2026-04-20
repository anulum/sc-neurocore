# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for optimizer/resource_optimizer

module ResourceOptimizerAccel

using Statistics, LinearAlgebra

mutable struct OptimizationResultState
    action::Float64
    luts_before::Float64
    luts_after::Float64
    sparsity::Float64
    bitstream_length::Float64
    fits::Float64
    target::Float64
    final_luts::Float64
    target_luts::Float64
    utilization_pct::Float64
    final_bitstream_length::Float64
    final_sparsity::Float64
    steps::Float64
    optimized_weights::Float64
end

function OptimizationResultState()
    OptimizationResultState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::OptimizationResultState)
    lines = [
        f"Resource Optimization: {s.target}",
        f"  Fits: {'YES' if s.fits else 'NO'}",
        f"  LUTs: {s.final_luts:,} / {s.target_luts:,} ({s.utilization_pct:.1f}%)",
        f"  Bitstream length: {s.final_bitstream_length}",
        f"  Sparsity: {s.final_sparsity:.1%}",
        f"  Steps taken: {length(s.steps)}",
    ]
    for s in s.steps
        lines = push!(, f"    {s.action}: {s.luts_before:,} -> {s.luts_after:,} LUTs")
    return "\n".join(lines)
end

function fit_to_target(layer_sizes, weights, target, max_iterations, min_bitstream_length, initial_bitstream_length)
    layer_sizes: list[tuple[int, int]],
    weights: list[np.ndarray],
    target: str = "ice40",
    max_iterations: int = 10,
    min_bitstream_length: int = 32,
    initial_bitstream_length: int = 256,
    ) -> OptimizationResult
    from sc_neurocore.energy.fpga_models import TARGETS
    target_info = TARGETS.get(target)
    if target_info is nothing
        raise ValueError(f"Unknown target '{target}'")
    current_weights = [w.copy() for w in weights]
    current_L = initial_bitstream_length
    steps = []
    prune_threshold = 0.001
    quant_bits = 16
    for iteration in 1:max_iterations
        report = estimate(layer_sizes, target=target, bitstream_length=current_L)
        if report.fits_on_target
            break
        luts_before = report.total_luts
        # Strategy 1: Halve bitstream length
        if current_L > min_bitstream_length
            current_L = max(current_L // 2, min_bitstream_length)
            report_after = estimate(layer_sizes, target=target, bitstream_length=current_L)
            steps = push!(,
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
        steps = push!(,
            OptimizationStep(
                action=f"Prune threshold={prune_threshold:.3f}",
                luts_before=luts_before,
                luts_after=report_after.total_luts,
                sparsity=prune_report.sparsity,
                bitstream_length=current_L,
            )
        )
        # Strategy 3: Reduce quantization
        if quant_bits > 4
            quant_bits = max(quant_bits - 2, 4)
            current_weights = quantize_weights(current_weights, bits=quant_bits)
            steps = push!(,
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
end

end # module ResourceOptimizerAccel
