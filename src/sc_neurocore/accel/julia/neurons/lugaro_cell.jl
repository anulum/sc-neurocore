# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for lugaro_cell

module LugaroCellAccel

export step!, simulate, LugaroCellState

mutable struct LugaroCellState
    v::Float64
    adapt::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_adapt::Float64
    a_adapt::Float64
    gain::Float64
    serotonin::Float64
    dt::Float64
end

function LugaroCellState()
    LugaroCellState(-55.0, 0.0, -55.0, -65.0, -48.0, 10.0, 150.0, 0.05, 2.0, 0.0, 0.5)
end

function with_serotonin(s::LugaroCellState, level)
    return cls(serotonin=max(0.0, min(1.0, level)))
end

function step!(s::LugaroCellState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        effective_gain = s.gain * (1.0 + 0.5 * s.serotonin)
        inp = effective_gain * I_ext
        dv = (-(s.v - s.v_rest) - s.adapt + inp) / s.tau_m
        s.v += s.dt * dv
        da = (s.a_adapt * (s.v - s.v_rest) - s.adapt) / s.tau_adapt
        s.adapt += s.dt * da
        if s.v >= s.v_threshold
            s.v = s.v_reset
            s.adapt += 1.0
            return 1
        end
        s.v = max(-100.0, min(60.0, s.v))
        if ! isfinite(s.v)
            s.v = s.v_reset
        end
        if ! isfinite(s.adapt)
            s.adapt = 0.0
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = LugaroCellState()
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

end # module LugaroCellAccel
