# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for perfect_integrator

module PerfectIntegratorAccel

export step!, simulate, PerfectIntegratorNeuronState, valid, reset!

mutable struct PerfectIntegratorNeuronState
    v::Float64
    c_m::Float64
    v_threshold::Float64
    v_reset::Float64
    dt::Float64
end

function PerfectIntegratorNeuronState()
    PerfectIntegratorNeuronState(0.0, 1.0, 1.0, 0.0, 0.1)
end

function valid(s::PerfectIntegratorNeuronState)::Bool
    return isfinite(s.v) &&
        isfinite(s.c_m) && s.c_m > 0.0 &&
        isfinite(s.v_threshold) &&
        isfinite(s.v_reset) && s.v_threshold > s.v_reset &&
        s.v < s.v_threshold &&
        isfinite(s.dt) && s.dt > 0.0
end

function step!(s::PerfectIntegratorNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    s.dt = dt
    if !isfinite(I_ext) || !valid(s)
        throw(DomainError((s.v, s.c_m, s.v_threshold, s.v_reset, s.dt, I_ext), "PerfectIntegrator state/current must be finite and physically ordered"))
    end

    voltage_increment = I_ext / s.c_m * s.dt
    next_v = s.v + voltage_increment
    if !isfinite(voltage_increment) || !isfinite(next_v)
        throw(DomainError((voltage_increment, next_v), "PerfectIntegrator voltage increment must remain finite"))
    end

    s.v = next_v
    if s.v >= s.v_threshold
        s.v = s.v_reset
        return 1
    end
    return 0
end

function reset!(s::PerfectIntegratorNeuronState)::Nothing
    s.v = s.v_reset
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PerfectIntegratorNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module PerfectIntegratorAccel
