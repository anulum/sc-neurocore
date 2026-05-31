# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for direction_selective_rgc

module DirectionSelectiveRgcAccel

export step!, step_rf!, simulate, validate_direction_selective_rgc, DirectionSelectiveRGCState

mutable struct DirectionSelectiveRGCState
    tau::Float64
    theta::Float64
    is_on_centre::Float64
    w_centre::Float64
    w_surround::Float64
    direction_pref::Float64
    dt::Float64
    v::Float64
    prev_intensity::Float64
    surround::Float64
end

function DirectionSelectiveRGCState()
    DirectionSelectiveRGCState(10.0, 0.5, 1.0, 1.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0)
end

function new_on!(s::DirectionSelectiveRGCState)
    s.is_on_centre = 1.0
    return s
end

function new_off!(s::DirectionSelectiveRGCState)
    s.is_on_centre = 0.0
    return s
end

@inline _finite(xs...) = all(isfinite, xs)

function validate_direction_selective_rgc(s::DirectionSelectiveRGCState)::Bool
    return _finite(s.tau, s.theta, s.is_on_centre, s.w_centre, s.w_surround, s.direction_pref, s.dt, s.v, s.prev_intensity, s.surround) &&
        s.tau > 0.0 && s.theta > 0.0 && s.dt > 0.0 &&
        s.w_centre >= 0.0 && s.w_surround >= 0.0 && s.prev_intensity >= 0.0 && s.surround >= 0.0 &&
        (s.is_on_centre == 0.0 || s.is_on_centre == 1.0)
end

function step_rf!(s::DirectionSelectiveRGCState, intensity::Float64, surround_mean::Float64)
    (!isfinite(intensity) || !isfinite(surround_mean) || intensity < 0.0 || surround_mean < 0.0 || !validate_direction_selective_rgc(s)) && return -1
    temporal_diff = intensity - s.prev_intensity
    centre_response = (s.is_on_centre == 1.0) ? s.w_centre * temporal_diff : -s.w_centre * temporal_diff
    next_surround = 0.9 * s.surround + 0.1 * surround_mean
    drive = centre_response - s.w_surround * next_surround
    decay = exp(-s.dt / s.tau)
    next_v = drive + (s.v - drive) * decay
    (!_finite(next_surround, drive, decay, next_v) || next_surround < 0.0) && return -1
    s.prev_intensity = intensity
    s.surround = next_surround
    if next_v >= s.theta
        s.v = 0.0
        return 1
    end
    s.v = next_v
    return 0
end

function step!(s::DirectionSelectiveRGCState, I_ext::Float64=0.0; dt::Union{Nothing,Float64}=nothing)
    if dt !== nothing
        if !isfinite(dt) || dt <= 0.0
            return -1
        end
        s.dt = dt
    end
    return step_rf!(s, I_ext, 0.0)
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=1.0)
    s = DirectionSelectiveRGCState()
    s.dt = dt
    trace = zeros(Float64, n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module DirectionSelectiveRgcAccel
