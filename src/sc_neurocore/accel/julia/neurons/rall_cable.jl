# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for rall_cable

module RallCableAccel

export step!, simulate, RallCableNeuronState, validate

mutable struct RallCableNeuronState
    n_comp::Int
    tau_m::Float64
    v_rest::Float64
    g_ratio::Float64
    v_threshold::Float64
    v_reset::Float64
    dt::Float64
    v::Vector{Float64}
end

function RallCableNeuronState(n_comp::Int=5)
    n = max(n_comp, 1)
    RallCableNeuronState(n, 20.0, -65.0, 0.5, -50.0, -65.0, 0.1, fill(-65.0, n))
end

finite_rall(x::Float64) = isfinite(x)

function validate(s::RallCableNeuronState)
    return s.n_comp >= 1 &&
        length(s.v) == s.n_comp &&
        finite_rall(s.tau_m) &&
        s.tau_m > 0.0 &&
        finite_rall(s.v_rest) &&
        finite_rall(s.g_ratio) &&
        s.g_ratio >= 0.0 &&
        finite_rall(s.v_threshold) &&
        finite_rall(s.v_reset) &&
        finite_rall(s.dt) &&
        s.dt > 0.0 &&
        all(isfinite, s.v)
end

function solve_tridiagonal(lower::Vector{Float64}, diagonal::Vector{Float64}, upper::Vector{Float64}, rhs::Vector{Float64})
    n = length(diagonal)
    if n == 0 || length(rhs) != n || length(lower) != n - 1 || length(upper) != n - 1
        return nothing
    end
    c_prime = zeros(max(n - 1, 0))
    d_prime = zeros(n)
    pivot = diagonal[1]
    if !(finite_rall(pivot) && pivot != 0.0)
        return nothing
    end
    if n > 1
        c_prime[1] = upper[1] / pivot
    end
    d_prime[1] = rhs[1] / pivot
    for i in 2:n
        pivot = diagonal[i] - lower[i - 1] * c_prime[i - 1]
        if !(finite_rall(pivot) && pivot != 0.0)
            return nothing
        end
        if i < n
            c_prime[i] = upper[i] / pivot
        end
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / pivot
    end
    solution = zeros(n)
    solution[n] = d_prime[n]
    for i in (n - 1):-1:1
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    end
    all(isfinite, solution) ? solution : nothing
end

function candidate(s::RallCableNeuronState, I_ext::Float64)
    if !(validate(s) && finite_rall(I_ext))
        return nothing
    end
    alpha = s.dt / s.tau_m
    offdiag = -alpha * s.g_ratio
    diagonal = fill(1.0 + alpha + 2.0 * alpha * s.g_ratio, s.n_comp)
    if s.n_comp == 1
        diagonal[1] = 1.0 + alpha
    else
        diagonal[1] = 1.0 + alpha + alpha * s.g_ratio
        diagonal[end] = 1.0 + alpha + alpha * s.g_ratio
    end
    lower = fill(offdiag, max(s.n_comp - 1, 0))
    upper = fill(offdiag, max(s.n_comp - 1, 0))
    rhs = s.v .- s.v_rest
    rhs[end] += alpha * I_ext
    solved = solve_tridiagonal(lower, diagonal, upper, rhs)
    solved === nothing && return nothing
    return solved .+ s.v_rest
end

function step!(s::RallCableNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if dt != s.dt
        s.dt = dt
    end
    next_v = candidate(s, I_ext)
    next_v === nothing && return -1
    previous_soma = s.v[1]
    if next_v[1] >= s.v_threshold && previous_soma < s.v_threshold
        next_v[1] = s.v_reset
        s.v = next_v
        return 1
    end
    s.v = next_v
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = RallCableNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v[1]
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module RallCableAccel
