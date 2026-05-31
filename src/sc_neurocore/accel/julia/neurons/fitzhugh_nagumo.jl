# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for fitzhugh_nagumo

module FitzhughNagumoAccel

export step!, simulate, validate_state, FitzHughNagumoNeuronState

mutable struct FitzHughNagumoNeuronState
    v::Float64
    w::Float64
    a::Float64
    b::Float64
    epsilon::Float64
    dt::Float64
    v_threshold::Float64
end

function FitzHughNagumoNeuronState()
    FitzHughNagumoNeuronState(-1.0, -0.5, 0.7, 0.8, 0.08, 0.1, 1.0)
end

function validate_state(s::FitzHughNagumoNeuronState)
    return all(isfinite, (s.v, s.w, s.a, s.b, s.epsilon, s.dt, s.v_threshold)) &&
        s.b > 0.0 &&
        s.epsilon > 0.0 &&
        s.dt > 0.0
end

function _rhs(s::FitzHughNagumoNeuronState, v::Float64, w::Float64, i_ext::Float64)
    if !(isfinite(v) && isfinite(w) && isfinite(i_ext))
        throw(DomainError((v, w, i_ext), "FitzHugh-Nagumo derivative input must be finite"))
    end
    dv = v - v^3 / 3.0 - w + i_ext
    dw = s.epsilon * (v + s.a - s.b * w)
    if !(isfinite(dv) && isfinite(dw))
        throw(DomainError((dv, dw), "FitzHugh-Nagumo derivative became non-finite"))
    end
    return dv, dw
end

function _rk4_candidate(s::FitzHughNagumoNeuronState, i_ext::Float64)
    dt = s.dt
    v0 = s.v
    w0 = s.w
    k1v, k1w = _rhs(s, v0, w0, i_ext)
    k2v, k2w = _rhs(s, v0 + 0.5 * dt * k1v, w0 + 0.5 * dt * k1w, i_ext)
    k3v, k3w = _rhs(s, v0 + 0.5 * dt * k2v, w0 + 0.5 * dt * k2w, i_ext)
    k4v, k4w = _rhs(s, v0 + dt * k3v, w0 + dt * k3w, i_ext)
    return (
        v0 + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
        w0 + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
    )
end

function step!(s::FitzHughNagumoNeuronState, i_ext::Float64=0.0; dt::Float64=s.dt)
    if !isfinite(dt) || dt <= 0.0
        throw(ArgumentError("dt must be finite and positive"))
    end
    old_dt = s.dt
    s.dt = dt
    if !(isfinite(i_ext) && validate_state(s))
        s.dt = old_dt
        throw(DomainError((s.v, s.w, i_ext), "FitzHugh-Nagumo state/current must be finite"))
    end
    v_prev = s.v
    new_v, new_w = _rk4_candidate(s, i_ext)
    if !(isfinite(new_v) && isfinite(new_w))
        s.dt = old_dt
        throw(DomainError((new_v, new_w), "FitzHugh-Nagumo candidate became non-finite"))
    end
    s.v = new_v
    s.w = new_w
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = FitzHughNagumoNeuronState()
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

end # module FitzhughNagumoAccel
