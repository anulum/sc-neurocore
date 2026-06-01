# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for sigmoid_rate

module SigmoidRateAccel

export step!, simulate, SigmoidRateNeuronState, valid, reset!

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

function step!(s::SigmoidRateNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
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

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = SigmoidRateNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.r
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module SigmoidRateAccel
