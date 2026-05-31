# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for adaptive_threshold_if

module AdaptiveThresholdIfAccel

export step!, simulate, valid, AdaptiveThresholdIFNeuronState

mutable struct AdaptiveThresholdIFNeuronState
    v::Float64
    theta::Float64
    v_rest::Float64
    v_reset::Float64
    theta_rest::Float64
    delta_theta::Float64
    tau_m::Float64
    tau_theta::Float64
    dt::Float64
end

function AdaptiveThresholdIFNeuronState()
    AdaptiveThresholdIFNeuronState(-65.0, -50.0, -65.0, -65.0, -50.0, 5.0, 10.0, 50.0, 0.1)
end

function step!(s::AdaptiveThresholdIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    s.dt = dt
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "AdaptiveThresholdIF input current must be finite"))
    end
    if !valid(s)
        throw(DomainError(s.v, "AdaptiveThresholdIF state parameters must be finite and physically ordered"))
    end
    next_v = _exact_relaxation(s, s.v, s.v_rest + I_ext, s.tau_m)
    next_theta = _exact_relaxation(s, s.theta, s.theta_rest, s.tau_theta)
    if !all(isfinite, (next_v, next_theta))
        throw(DomainError(next_v, "AdaptiveThresholdIF exact relaxation update must remain finite"))
    end
    if next_v >= next_theta
        spike_theta = next_theta + s.delta_theta
        if !isfinite(spike_theta)
            throw(DomainError(spike_theta, "AdaptiveThresholdIF threshold jump update must remain finite"))
        end
        s.v = s.v_reset
        s.theta = spike_theta
        return 1
    end
    s.v = next_v
    s.theta = next_theta
    return 0
end

function valid(s::AdaptiveThresholdIFNeuronState)
    return isfinite(s.v) &&
        isfinite(s.theta) &&
        isfinite(s.v_rest) &&
        isfinite(s.v_reset) &&
        isfinite(s.theta_rest) &&
        isfinite(s.delta_theta) &&
        s.delta_theta >= 0.0 &&
        isfinite(s.tau_m) &&
        s.tau_m > 0.0 &&
        isfinite(s.tau_theta) &&
        s.tau_theta > 0.0 &&
        isfinite(s.dt) &&
        s.dt > 0.0 &&
        s.theta_rest > s.v_rest &&
        s.theta_rest > s.v_reset
end

function _exact_relaxation(s::AdaptiveThresholdIFNeuronState, state::Float64, steady_state::Float64, tau::Float64)::Float64
    return steady_state + (state - steady_state) * exp(-s.dt / tau)
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AdaptiveThresholdIFNeuronState()
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

end # module AdaptiveThresholdIfAccel
