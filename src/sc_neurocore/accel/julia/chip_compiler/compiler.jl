# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for chip_compiler/compiler

module CompilerAccel

using Statistics, LinearAlgebra

mutable struct CompilationResultState
    core_id::Float64
    layer_index::Float64
    neuron_start::Float64
    neuron_end::Float64
    n_neurons::Float64
    chip::Float64
    success::Float64
    core_mappings::Float64
    total_cores_used::Float64
    total_neurons_mapped::Float64
    weight_bits::Float64
    violations::Float64
    warnings::Float64
    quantized_weights::Float64
end

function CompilationResultState()
    CompilationResultState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::CompilationResultState)
    status = "SUCCESS" if s.success else "FAILED"
    lines = [
        f"Compilation [{s.chip}]: {status}",
        f"  Cores: {s.total_cores_used}",
        f"  Neurons: {s.total_neurons_mapped}",
        f"  Weight precision: {s.weight_bits}-bit",
    ]
    for v in s.violations:  # pragma: no cover
        lines = push!(, f"  [VIOLATION] {v}")
    for w in s.warnings:  # pragma: no cover
        lines = push!(, f"  [WARNING] {w}")
    return "\n".join(lines)
end

function compile_for_chip(layer_sizes, weights, neuron_types, target)
    layer_sizes: list[tuple[int, int]],
    weights: list[np.ndarray] | nothing = nothing,
    neuron_types: list[str] | nothing = nothing,
    target: str | ChipSpec = "loihi2",
    ) -> CompilationResult
    if isinstance(target, str)
        chip = BUILTIN_CHIPS.get(target)
        if chip is nothing
            return CompilationResult(
                chip=target,
                success=false,
                violations=[
                    f"Unknown chip target '{target}'. Available: {list(BUILTIN_CHIPS.keys())}"
                ],
            )
    else
        chip = target
    result = CompilationResult(chip=chip.name, weight_bits=chip.core.weight_bits)
    violations = []
    warnings = []
    mappings = []
    total_neurons = sum(n for _, n in layer_sizes)
    result.total_neurons_mapped = total_neurons
    # Check total capacity
    if ! chip.fits(total_neurons)
        violations = push!(,
            f"Network has {total_neurons} neurons but {chip.name} supports "
            f"max {chip.total_neurons} ({chip.total_cores} cores x "
            f"{chip.core.max_neurons} neurons/core)"
        )
    # Check neuron type compatibility
    if neuron_types is ! nothing
        for i, nt in enumerate(neuron_types)
            if nt ! in chip.core.supported_neuron_types
                violations = push!(,
                    f"Layer {i}: neuron type '{nt}' ! supported on {chip.name}. "
                    f"Supported: {chip.core.supported_neuron_types}"
                )
    # Check fan-out per layer
    for i, (n_in, n_out) in enumerate(layer_sizes)
        if n_out > chip.max_fan_out
            violations = push!(,
                f"Layer {i}: fan-out {n_out} exceeds {chip.name} max {chip.max_fan_out}"
            )
    # Partition into cores
    core_id = 0
    for i, (n_in, n_out) in enumerate(layer_sizes)
        neurons_remaining = n_out
        offset = 0
        while neurons_remaining > 0
            batch = min(neurons_remaining, chip.core.max_neurons)
            mappings = push!(,
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
    if core_id > chip.total_cores
        violations = push!(, f"Needs {core_id} cores but {chip.name} has {chip.total_cores}")
    # Quantize weights
    if weights is ! nothing
        bits = chip.core.weight_bits
        n_levels = 2^bits
        quantized = []
        for w in weights
            abs_max = max(abs(w).max(), 1e-8)
            scale = abs_max / (n_levels // 2 - 1)
            q = np.round(w / scale) * scale
            q = clamp(q, -abs_max, abs_max)
            quantized = push!(, q)
            # Warn about precision loss
            mse = float(mean((w - q) ^ 2))
            if mse > 0.01 * float(mean(w^2)):  # pragma: no cover
                warnings = push!(,
                    f"Weight quantization to {bits}-bit introduces "
                    f"{mse / max(float(mean(w^2)), 1e-12) * 100:.1f}% relative error"
                )
        result.quantized_weights = quantized
    # Analog noise warning
    if chip.analog_noise_cv > 0.05
        warnings = push!(,
            f"{chip.name} has {chip.analog_noise_cv:.0%} analog noise CV. "
            f"Use variation-aware training (digital_twin.FPGAMismatchModel)"
        )
    result.violations = violations
    result.warnings = warnings
    result.success = length(violations) == 0
    return result
end

end # module CompilerAccel
