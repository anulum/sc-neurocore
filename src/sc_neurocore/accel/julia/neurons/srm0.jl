# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for srm0

module Srm0Accel

export get_state, step!, simulate, SRM0NeuronState

mutable struct SRM0NeuronState
    v::Float64
    v_rest::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_eta::Float64
    eta_reset::Float64
    resistance::Float64
    dt::Float64
    eta::Float64
    t::Float64
    last_spike_time::Float64
end

function SRM0NeuronState()
    SRM0NeuronState(0.0, 0.0, 1.0, 20.0, 50.0, 5.0, 1.0, 1.0, 0.0, 0.0, -1000.0)
end

finite_srm0(x::Float64) = isfinite(x)

function validate(s::SRM0NeuronState)
    return finite_srm0(s.v) &&
        finite_srm0(s.v_rest) &&
        finite_srm0(s.v_threshold) &&
        finite_srm0(s.tau_m) &&
        s.tau_m > 0.0 &&
        finite_srm0(s.tau_eta) &&
        s.tau_eta > 0.0 &&
        finite_srm0(s.eta_reset) &&
        s.eta_reset >= 0.0 &&
        finite_srm0(s.resistance) &&
        finite_srm0(s.dt) &&
        s.dt > 0.0 &&
        finite_srm0(s.eta) &&
        finite_srm0(s.t) &&
        finite_srm0(s.last_spike_time)
end

function get_state(s::SRM0NeuronState)
    return Dict("v" => s.v, "eta" => s.eta, "t" => s.t)
end

function eta_coupling_integral(s::SRM0NeuronState)
    membrane_decay = exp(-s.dt / s.tau_m)
    eta_decay = exp(-s.dt / s.tau_eta)
    rate_delta = (1.0 / s.tau_m) - (1.0 / s.tau_eta)
    if abs(rate_delta) < 1.0e-14
        return s.dt * membrane_decay / s.tau_m
    end
    return (eta_decay - membrane_decay) / (s.tau_m * rate_delta)
end

function exact_candidate(s::SRM0NeuronState, I_ext::Float64)
    membrane_decay = exp(-s.dt / s.tau_m)
    eta_decay = exp(-s.dt / s.tau_eta)
    steady = s.v_rest + s.resistance * I_ext
    next_eta = s.eta * eta_decay
    next_v = steady + (s.v - steady) * membrane_decay + s.eta * eta_coupling_integral(s)
    if !(finite_srm0(next_v) && finite_srm0(next_eta))
        return nothing
    end
    return next_v, next_eta
end

function step!(s::SRM0NeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if dt != s.dt
        s.dt = dt
    end
    if !(validate(s) && finite_srm0(I_ext))
        return -1
    end
    candidate = exact_candidate(s, I_ext)
    candidate === nothing && return -1
    next_v, next_eta = candidate
    next_t = s.t + s.dt
    if next_v >= s.v_threshold
        s.v = s.v_rest
        s.eta = -s.eta_reset
        s.t = next_t
        s.last_spike_time = next_t
        return 1
    end
    s.v = next_v
    s.eta = next_eta
    s.t = next_t
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=1.0)
    s = SRM0NeuronState()
    s.dt = dt
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module Srm0Accel
