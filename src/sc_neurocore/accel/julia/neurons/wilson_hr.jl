# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for wilson_hr

module WilsonHrAccel

export step!, simulate, validate, WilsonHRNeuronState

mutable struct WilsonHRNeuronState
    v::Float64
    r::Float64
    tau_r::Float64
    v_peak::Float64
    dt::Float64
end

function WilsonHRNeuronState()
    WilsonHRNeuronState(-0.7, 0.1, 1.9, 0.4, 0.05)
end

function validate(s::WilsonHRNeuronState)::Bool
    return isfinite(s.v) &&
        isfinite(s.r) &&
        isfinite(s.tau_r) && s.tau_r > 0.0 &&
        isfinite(s.v_peak) &&
        isfinite(s.dt) && s.dt > 0.0
end

poly(v::Float64)::Float64 = -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55)

function derivatives(s::WilsonHRNeuronState, v::Float64, r::Float64, I_ext::Float64)
    if !all(isfinite, (v, r, I_ext))
        return (0.0, 0.0, false)
    end
    p = poly(v)
    syn = -26.0 * r * (v + 0.92)
    dv = p + syn + I_ext
    dr = (-r + 1.35 * v + 1.03) / s.tau_r
    if !all(isfinite, (p, syn, dv, dr))
        return (0.0, 0.0, false)
    end
    return (dv, dr, true)
end

function rk4_candidate(s::WilsonHRNeuronState, I_ext::Float64)
    v0, r0, dt = s.v, s.r, s.dt
    k1v, k1r, ok = derivatives(s, v0, r0, I_ext)
    ok || return (0.0, 0.0, false)
    k2v, k2r, ok = derivatives(s, v0 + 0.5 * dt * k1v, r0 + 0.5 * dt * k1r, I_ext)
    ok || return (0.0, 0.0, false)
    k3v, k3r, ok = derivatives(s, v0 + 0.5 * dt * k2v, r0 + 0.5 * dt * k2r, I_ext)
    ok || return (0.0, 0.0, false)
    k4v, k4r, ok = derivatives(s, v0 + dt * k3v, r0 + dt * k3r, I_ext)
    ok || return (0.0, 0.0, false)
    next_v = v0 + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
    next_r = r0 + dt * (k1r + 2.0 * k2r + 2.0 * k3r + k4r) / 6.0
    return (next_v, next_r, isfinite(next_v) && isfinite(next_r))
end

function step!(s::WilsonHRNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate(s) || !isfinite(I_ext)
        return -1
    end
    next_v, next_r, ok = rk4_candidate(s, I_ext)
    if !ok
        return -1
    end
    s.v = next_v
    s.r = next_r
    if s.v >= s.v_peak
        s.v = -0.7
        return 1
    end
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = WilsonHRNeuronState()
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

end # module WilsonHrAccel
