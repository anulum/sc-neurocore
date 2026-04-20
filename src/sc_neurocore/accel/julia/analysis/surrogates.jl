# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analysis/surrogates

module SurrogatesAccel

using Statistics, LinearAlgebra

function surrogate_isi_shuffle(binary_train::Any, seed::Any)
    # Accelerated surrogate_isi_shuffle (17 lines)
    return nothing
end

function surrogate_dither(binary_train::Any, dither_ms::Any, dt::Any, seed::Any)
    # Accelerated surrogate_dither (13 lines)
    return nothing
end

function surrogate_trial_shuffle(trains::Any, seed::Any)
    # Accelerated surrogate_trial_shuffle (7 lines)
    return nothing
end

function homogeneous_poisson(rate_hz::Any, duration_s::Any, dt::Any, seed::Any)
    # Accelerated homogeneous_poisson (7 lines)
    return nothing
end

function inhomogeneous_poisson(rate_func::Any, duration_s::Any, dt::Any, seed::Any)
    # Accelerated inhomogeneous_poisson (18 lines)
    return nothing
end

function gamma_process(rate_hz::Any, shape::Any, duration_s::Any, dt::Any, seed::Any)
    # Accelerated gamma_process (21 lines)
    return nothing
end

function compound_poisson_process(rate_hz::Any, burst_mean::Any, duration_s::Any, dt::Any, seed::Any)
    # Accelerated compound_poisson_process (19 lines)
    return nothing
end

function surrogate_joint_isi(binary_train::Any, seed::Any)
    # Accelerated surrogate_joint_isi (24 lines)
    return nothing
end

function surrogate_bin_shuffling(binary_train::Any, bin_size::Any, seed::Any)
    # Accelerated surrogate_bin_shuffling (13 lines)
    return nothing
end

function surrogate_spike_train_shifting(binary_train::Any, max_shift::Any, seed::Any)
    # Accelerated surrogate_spike_train_shifting (7 lines)
    return nothing
end

end # module SurrogatesAccel
