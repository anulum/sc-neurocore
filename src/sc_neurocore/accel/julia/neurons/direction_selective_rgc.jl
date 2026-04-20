# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for direction_selective_rgc

module DirectionSelectiveRgcAccel

export step!, simulate, DirectionSelectiveRGCState

mutable struct DirectionSelectiveRGCState
    tau::Float64
    theta::Float64
    is_on_centre::Float64
    w_centre::Float64
    w_surround::Float64
    direction_pref::Float64
    dt::Float64
    v::Float64
    _prev_intensity::Float64
    _surround::Float64
end

function DirectionSelectiveRGCState()
    DirectionSelectiveRGCState(10.0, 0.5, 1.0, 1.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0)
end

function new_on(s::DirectionSelectiveRGCState)
    return cls(is_on_centre=true)
end

function new_off(s::DirectionSelectiveRGCState)
    return cls(is_on_centre=false)
end

function step_rf(s::DirectionSelectiveRGCState, intensity, surround_mean)
    temporal_diff = intensity - s._prev_intensity
    s._prev_intensity = intensity
    if s.is_on_centre
        centre_response = s.w_centre * temporal_diff
    else
        centre_response = -s.w_centre * temporal_diff
    end
    s._surround = 0.9 * s._surround + 0.1 * surround_mean
    surround_inhib = s.w_surround * s._surround
    drive = centre_response - surround_inhib
    s.v += (-s.v + drive) / s.tau * s.dt
    if s.v >= s.theta
        s.v = 0.0
        return 1
    end
    return 0
end

function step!(s::DirectionSelectiveRGCState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        return s.step_rf(I_ext, 0.0)
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = DirectionSelectiveRGCState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.tau
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module DirectionSelectiveRgcAccel
