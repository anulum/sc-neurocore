# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for sigmoid_rate

module SigmoidRateAccel

export step!, simulate, simulate_trace, SigmoidRateNeuronState, valid, reset!

mutable struct SigmoidRateNeuronState
    r::Float64
    tau::Float64
    beta::Float64
    theta::Float64
    dt::Float64
end

function SigmoidRateNeuronState()
    SigmoidRateNeuronState(0.0, 10.0, 1.0, 0.0, 0.1)
end

function valid(s::SigmoidRateNeuronState)::Bool
    return all(isfinite, (s.r, s.tau, s.beta, s.theta, s.dt)) &&
        0.0 <= s.r <= 1.0 &&
        s.tau > 0.0 &&
        s.dt > 0.0
end

function sigmoid_transfer(beta::Float64, I_ext::Float64, theta::Float64)::Float64
    z = beta * (I_ext - theta)
    if isinf(z)
        return z > 0.0 ? 1.0 : 0.0
    end
    if !isfinite(z)
        throw(DomainError(z, "SigmoidRate transfer argument must be finite or saturating"))
    end
    if z >= 0.0
        return 1.0 / (1.0 + exp(-z))
    end
    exp_z = exp(z)
    return exp_z / (1.0 + exp_z)
end

function step!(s::SigmoidRateNeuronState, I_ext::Float64=0.0)::Float64
    if !isfinite(I_ext) || !valid(s)
        throw(DomainError((s.r, I_ext), "SigmoidRate state/current must be finite and well-formed"))
    end
    sigma = sigmoid_transfer(s.beta, I_ext, s.theta)
    next_r = exact_relaxation(s.r, sigma, s.dt, s.tau)
    if !isfinite(next_r) || next_r < 0.0 || next_r > 1.0
        throw(DomainError(next_r, "SigmoidRate exact relaxation update must remain finite and in [0,1]"))
    end
    s.r = next_r
    return next_r
end

function exact_relaxation(r::Float64, sigma::Float64, dt::Float64, tau::Float64)::Float64
    decay = exp(-dt / tau)
    return decay * r + (1.0 - decay) * sigma
end

function reset!(s::SigmoidRateNeuronState)::Nothing
    s.r = 0.0
    return nothing
end

function simulate_trace(
    r::Float64,
    tau::Float64,
    beta::Float64,
    theta::Float64,
    dt::Float64,
    n_steps::Int,
    I_ext::Float64,
)
    if n_steps < 0
        throw(DomainError(n_steps, "SigmoidRate step count must be non-negative"))
    end
    s = SigmoidRateNeuronState(r, tau, beta, theta, dt)
    if !valid(s) || !isfinite(I_ext)
        throw(DomainError((r, I_ext), "SigmoidRate batch contract is invalid"))
    end
    trace = Vector{Float64}(undef, n_steps)
    for index in eachindex(trace)
        trace[index] = step!(s, I_ext)
    end
    return (trace=trace, rf=s.r)
end

function simulate(
    n_steps::Int=1000;
    I_ext::Float64=10.0,
    r::Float64=0.0,
    tau::Float64=10.0,
    beta::Float64=1.0,
    theta::Float64=0.0,
    dt::Float64=0.1,
)
    result = simulate_trace(r, tau, beta, theta, dt, n_steps, I_ext)
    return result.trace, result.rf
end

end # module SigmoidRateAccel
