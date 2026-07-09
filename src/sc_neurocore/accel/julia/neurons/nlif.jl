# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

module NlifAccel

export NonlinearLIFNeuronState, valid, step!, reset!

mutable struct NonlinearLIFNeuronState
    v::Float64
    w::Float64
    v_rest::Float64
    v_crit::Float64
    v_threshold::Float64
    v_reset::Float64
    a::Float64
    b::Float64
    tau_w::Float64
    c_m::Float64
    dt::Float64
end

function NonlinearLIFNeuronState(; v=-65.0, w=0.0, v_rest=-65.0, v_crit=-40.0,
    v_threshold=-20.0, v_reset=-65.0, a=0.04, b=0.5, tau_w=100.0, c_m=1.0, dt=0.1)
    return NonlinearLIFNeuronState(v, w, v_rest, v_crit, v_threshold, v_reset, a, b, tau_w, c_m, dt)
end

function valid(s::NonlinearLIFNeuronState)::Bool
    return all(isfinite, (s.v, s.w, s.v_rest, s.v_crit, s.v_threshold, s.v_reset,
        s.a, s.b, s.tau_w, s.c_m, s.dt)) &&
        s.v_rest < s.v_crit &&
        s.v_crit < s.v_threshold &&
        s.v_reset < s.v_threshold &&
        s.a >= 0.0 &&
        s.b >= 0.0 &&
        s.tau_w > 0.0 &&
        s.c_m > 0.0 &&
        s.dt > 0.0 &&
        s.dt <= s.tau_w
end

function derivatives(s::NonlinearLIFNeuronState, v::Float64, w::Float64, current::Float64)
    nonlinear = s.a * (v - s.v_rest) * (v - s.v_crit)
    dv = (nonlinear - w + current) / s.c_m
    dw = (s.b * (v - s.v_rest) - w) / s.tau_w
    return dv, dw
end

function rk4_candidate(s::NonlinearLIFNeuronState, current::Float64)
    k1v, k1w = derivatives(s, s.v, s.w, current)
    k2v, k2w = derivatives(s, s.v + 0.5 * s.dt * k1v, s.w + 0.5 * s.dt * k1w, current)
    k3v, k3w = derivatives(s, s.v + 0.5 * s.dt * k2v, s.w + 0.5 * s.dt * k2w, current)
    k4v, k4w = derivatives(s, s.v + s.dt * k3v, s.w + s.dt * k3w, current)
    next_v = s.v + (s.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    next_w = s.w + (s.dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
    return next_v, next_w
end

function step!(s::NonlinearLIFNeuronState, current::Float64)::Int
    if !isfinite(current) || !valid(s)
        return -1
    end

    next_v, next_w = rk4_candidate(s, current)
    if !(isfinite(next_v) && isfinite(next_w))
        return -1
    end
    s.v = next_v
    s.w = next_w

    if next_v >= s.v_threshold
        s.v = s.v_reset
        return 1
    end
    return 0
end

function reset!(s::NonlinearLIFNeuronState)::Nothing
    s.v = s.v_rest
    s.w = 0.0
    return nothing
end

end
