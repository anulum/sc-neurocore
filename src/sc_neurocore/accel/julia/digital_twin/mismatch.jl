# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for digital_twin/mismatch

module MismatchAccel

using Statistics, LinearAlgebra

mutable struct FPGAMismatchModelState
    quantization_bits::Float64
    weight_cv::Float64
    threshold_cv::Float64
    clock_jitter_pct::Float64
    seed::Float64
end

function FPGAMismatchModelState()
    FPGAMismatchModelState(16.0, 0.02, 0.05, 0.01, 42.0)
end

function quantize(s::FPGAMismatchModelState, values)
    fraction = s.quantization_bits // 2
    scale = 1 << fraction
    quantized = np.round(values * scale) / scale
    return quantized
end

function perturb_weights(s::FPGAMismatchModelState, weights)
    noise = s._rng.normal(0, s.weight_cv, weights.shape)
    return s.quantize(weights * (1.0 + noise))
end

function perturb_thresholds(s::FPGAMismatchModelState, thresholds)
    noise = s._rng.normal(0, s.threshold_cv, thresholds.shape)
    return s.quantize(thresholds * (1.0 + noise))
end

function jitter_timing(s::FPGAMismatchModelState, n_steps)
    jitter = s._rng.normal(1.0, s.clock_jitter_pct, n_steps)
    return clamp(jitter, 0.9, 1.1)
end

function apply_to_network_weights(s::FPGAMismatchModelState, weights)
    return [s.perturb_weights(w) for w in weights]
end

function mismatch_report(s::FPGAMismatchModelState, weights)
    perturbed = s.apply_to_network_weights(weights)
    total_params = sum(w.size for w in weights)
    total_error = sum(abs(w - p).sum() for w, p in zip(weights, perturbed))
    max_error = max(abs(w - p).max() for w, p in zip(weights, perturbed))
    return {
        "total_parameters": total_params,
        "mean_absolute_error": float(total_error / max(total_params, 1)),
        "max_absolute_error": float(max_error),
        "weight_cv": s.weight_cv,
        "threshold_cv": s.threshold_cv,
        "quantization_bits": s.quantization_bits,
    }
end

end # module MismatchAccel
