# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for non_resetting_lif

module NonResettingLifAccel

export step!, simulate, NonResettingLIFNeuronState, valid, reset!

mutable struct NonResettingLIFNeuronState
    v::Float64
    theta::Float64
    v_rest::Float64
    theta_rest::Float64
    delta_theta::Float64
    tau_m::Float64
    tau_theta::Float64
    r_m::Float64
    dt::Float64
end

function NonResettingLIFNeuronState()
    NonResettingLIFNeuronState(-65.0, -50.0, -65.0, -50.0, 5.0, 10.0, 50.0, 1.0, 0.1)
end

function valid(s::NonResettingLIFNeuronState)::Bool
    return all(isfinite, (s.v, s.theta, s.v_rest, s.theta_rest, s.delta_theta, s.tau_m, s.tau_theta, s.r_m, s.dt)) &&
        s.delta_theta >= 0.0 &&
        s.r_m >= 0.0 &&
        s.tau_m > 0.0 &&
        s.tau_theta > 0.0 &&
        s.dt > 0.0
end

function step!(s::NonResettingLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    s.dt = dt
    if !isfinite(I_ext) || !valid(s)
        throw(DomainError((s.v, s.theta, I_ext), "NonResettingLIF state/current must be finite and physically valid"))
    end
    membrane_update = (-(s.v - s.v_rest) + s.r_m * I_ext) / s.tau_m * s.dt
    next_v = s.v + membrane_update
    if !isfinite(membrane_update) || !isfinite(next_v)
        throw(DomainError((membrane_update, next_v), "NonResettingLIF membrane update must remain finite"))
    end
    threshold_update = -(s.theta - s.theta_rest) / s.tau_theta * s.dt
    next_theta = s.theta + threshold_update
    if !isfinite(threshold_update) || !isfinite(next_theta)
        throw(DomainError((threshold_update, next_theta), "NonResettingLIF threshold update must remain finite"))
    end
    spike = next_v >= next_theta
    if spike
        next_theta += s.delta_theta
        if !isfinite(next_theta)
            throw(DomainError(next_theta, "NonResettingLIF threshold update must remain finite"))
        end
    end
    s.v = next_v
    s.theta = next_theta
    return spike ? 1 : 0
end

function reset!(s::NonResettingLIFNeuronState)::Nothing
    s.v = s.v_rest
    s.theta = s.theta_rest
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = NonResettingLIFNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module NonResettingLifAccel
