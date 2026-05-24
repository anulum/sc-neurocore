# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for benda_herz

module BendaHerzAccel

export step!, simulate, BendaHerzNeuronState, valid, reset!, f_onset

mutable struct BendaHerzNeuronState
    a::Float64
    f_max::Float64
    beta::Float64
    i_half::Float64
    tau_a::Float64
    delta_a::Float64
    dt::Float64
    rng_threshold::Float64
end

function BendaHerzNeuronState()
    BendaHerzNeuronState(0.0, 200.0, 0.1, 5.0, 100.0, 0.5, 1.0, 0.0)
end

function valid(s::BendaHerzNeuronState)::Bool
    return isfinite(s.a) && s.a >= 0.0 &&
        isfinite(s.f_max) && s.f_max > 0.0 &&
        isfinite(s.beta) && s.beta > 0.0 &&
        isfinite(s.i_half) &&
        isfinite(s.tau_a) && s.tau_a > 0.0 &&
        isfinite(s.delta_a) && s.delta_a >= 0.0 &&
        isfinite(s.dt) && s.dt > 0.0 &&
        isfinite(s.rng_threshold) && 0.0 <= s.rng_threshold < 1.0
end

function f_onset(s::BendaHerzNeuronState, x::Float64)::Float64
    z = s.beta * (x - s.i_half)
    if z >= 0.0
        return s.f_max / (1.0 + exp(-z))
    end
    exp_z = exp(z)
    return s.f_max * exp_z / (1.0 + exp_z)
end

function step!(s::BendaHerzNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    if !isfinite(I_ext) || !isfinite(dt) || dt <= 0.0 || !valid(s)
        return 0
    end

    rate = f_onset(s, I_ext - s.a)
    p = rate * dt / 1000.0
    if !isfinite(rate) || !isfinite(p) || p > 1.0
        return 0
    end
    next_a = s.a + (-s.a / s.tau_a + s.delta_a * rate) * dt
    if !isfinite(next_a) || next_a < 0.0
        return 0
    end

    s.dt = dt
    s.a = next_a
    return s.rng_threshold < p ? 1 : 0
end

function reset!(s::BendaHerzNeuronState)::Nothing
    s.a = 0.0
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=1.0)
    s = BendaHerzNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.a
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module BendaHerzAccel
