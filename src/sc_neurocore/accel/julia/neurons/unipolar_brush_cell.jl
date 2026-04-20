# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for unipolar_brush_cell

module UnipolarBrushCellAccel

export step!, simulate, UnipolarBrushCellState

mutable struct UnipolarBrushCellState
    v::Float64
    persistent::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_persistent::Float64
    persistent_gain::Float64
    gain::Float64
    dt::Float64
end

function UnipolarBrushCellState()
    UnipolarBrushCellState(-65.0, 0.0, -65.0, -70.0, -50.0, 8.0, 200.0, 0.5, 2.5, 0.5)
end

function step!(s::UnipolarBrushCellState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        inp = s.gain * max(0.0, I_ext)
        dp = (s.persistent_gain * inp - s.persistent) / s.tau_persistent
        s.persistent += s.dt * dp
        s.persistent = max(0.0, s.persistent)
        dv = (-(s.v - s.v_rest) + inp + s.persistent) / s.tau_m
        s.v += s.dt * dv
        if s.v >= s.v_threshold
            s.v = s.v_reset
            return 1
        end
        s.v = max(-100.0, min(60.0, s.v))
        if ! isfinite(s.v)
            s.v = s.v_reset
        end
        if ! isfinite(s.persistent)
            s.persistent = 0.0
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = UnipolarBrushCellState()
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

end # module UnipolarBrushCellAccel
