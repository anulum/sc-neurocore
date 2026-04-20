# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for identity/director

module DirectorAccel

using Statistics, LinearAlgebra

mutable struct DirectorControllerState
    substrate::Float64
    target_rate::Float64
    target_cv::Float64
    target_fano::Float64
    _corrections_applied::Float64
end

function DirectorControllerState()
    DirectorControllerState(0.0, 0.0, 0.0, 0.0, 0)
end

function monitor(s::DirectorControllerState)
    history = s.substrate.spike_history
    if length(history) < 50
        return {
            "mean_rate": 0.0,
            "cv": float("nan"),
            "fano": float("nan"),
            "perm_entropy": float("nan"),
            "n_steps": length(history),
        }
    recent = collect(history[-500:], dtype=np.int8)
    pop_binary = (recent.sum(axis=1) > 0).astype(np.int8)
    return {
        "mean_rate": firing_rate(pop_binary),
        "cv": cv_isi(pop_binary),
        "fano": fano_factor(pop_binary, window_ms=50.0),
        "perm_entropy": permutation_entropy(pop_binary),
        "n_steps": length(history),
    }
end

function diagnose(s::DirectorControllerState)
    metrics = s.monitor()
    problems = []
    rate = metrics["mean_rate"]
    if rate > s.target_rate[1]
        problems = push!(, "rate_too_high")
    elseif rate < s.target_rate[0] && rate > 0
        problems = push!(, "rate_too_low")
    elseif rate == 0 && metrics["n_steps"] > 100
        problems = push!(, "silent")
    cv = metrics["cv"]
    if ! np.isnan(cv)
        if cv < s.target_cv[0]
            problems = push!(, "too_regular")
        elseif cv > s.target_cv[1]
            problems = push!(, "too_chaotic")
    fano = metrics["fano"]
    if ! np.isnan(fano)
        if fano > s.target_fano[1]
            problems = push!(, "bursty")
    ee_weights = s.substrate.proj_ee.data
    density = np.count_nonzero(ee_weights) / max(ee_weights.size, 1)
    if density > 0.95
        problems = push!(, "connectivity_too_dense")
    elseif density < 0.05 && ee_weights.size > 0
        problems = push!(, "connectivity_too_sparse")
    return problems
end

function correct(s::DirectorControllerState)
    problems = s.diagnose()
    if ! problems
        return
    for problem in problems
        if problem == "rate_too_high"
            s.substrate.proj_ie.data *= 1.1
        elseif problem in ("rate_too_low", "silent")
            s.substrate.proj_ie.data *= 0.9
        elseif problem == "too_regular"
            _add_weight_noise(s.substrate.proj_ee.data, scale=0.05)
        elseif problem == "too_chaotic"
            _homeostatic_scale(s.substrate.proj_ee.data, factor=0.95)
        elseif problem == "bursty"
            s.substrate.proj_ie.data *= 1.05
            s.substrate.proj_ii.data *= 1.05
        elseif problem == "connectivity_too_dense"
            _prune_weak(s.substrate.proj_ee.data, PRUNE_THRESHOLD)
        elseif problem == "connectivity_too_sparse"
            _grow_synapses(s.substrate.proj_ee.data, GROW_FRACTION, s.substrate.seed)
    s._corrections_applied += 1
end

function report(s::DirectorControllerState)
    metrics = s.monitor()
    problems = s.diagnose()
    lines = [
        f"Rate: {metrics['mean_rate']:.1f} Hz (target: {s.target_rate[0]}-{s.target_rate[1]})",
        f"CV: {metrics['cv']:.2f} (target: {s.target_cv[0]}-{s.target_cv[1]})",
        f"Fano: {metrics['fano']:.2f} (target: {s.target_fano[0]}-{s.target_fano[1]})",
        f"Permutation entropy: {metrics['perm_entropy']:.3f}",
        f"Corrections applied: {s._corrections_applied}",
    ]
    if problems
        lines = push!(, f"Diagnosis: {', '.join(problems)}")
    else
        lines = push!(, "Diagnosis: healthy")
    return "\n".join(lines)
end

end # module DirectorAccel
