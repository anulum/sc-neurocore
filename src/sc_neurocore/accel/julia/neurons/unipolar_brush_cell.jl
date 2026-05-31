# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for unipolar_brush_cell

module UnipolarBrushCellAccel

export step!, simulate, UnipolarBrushCellState

const UBC_V_MIN = -100.0
const UBC_V_MAX = 60.0

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

function _all_finite(values...)
    all(isfinite, values)
end

function _validate_configuration(s::UnipolarBrushCellState, dt::Float64)
    if !_all_finite(
        s.v_rest,
        s.v_reset,
        s.v_threshold,
        s.tau_m,
        s.tau_persistent,
        s.persistent_gain,
        s.gain,
        s.dt,
        dt,
    )
        throw(ArgumentError("configuration values must be finite"))
    end
    if s.tau_m <= 0.0
        throw(ArgumentError("tau_m must be positive"))
    end
    if s.tau_persistent <= 0.0
        throw(ArgumentError("tau_persistent must be positive"))
    end
    if dt <= 0.0
        throw(ArgumentError("dt must be positive"))
    end
    if s.persistent_gain < 0.0
        throw(ArgumentError("persistent_gain must be non-negative"))
    end
    if s.gain < 0.0
        throw(ArgumentError("gain must be non-negative"))
    end
    if s.v_reset >= s.v_threshold
        throw(ArgumentError("v_reset must be below v_threshold"))
    end
end

function _validate_state(s::UnipolarBrushCellState)
    if !_all_finite(s.v, s.persistent)
        throw(ArgumentError("state values must be finite"))
    end
    if s.v < UBC_V_MIN || s.v > UBC_V_MAX
        throw(ArgumentError("v state is outside the bounded membrane range"))
    end
    if s.persistent < 0.0
        throw(ArgumentError("persistent state must be non-negative"))
    end
end

function _first_order_relaxation(previous::Float64, steady_state::Float64, dt::Float64, tau::Float64)
    previous + (steady_state - previous) * (-expm1(-dt / tau))
end

function step!(s::UnipolarBrushCellState, I_ext::Float64=0.0; dt::Float64=s.dt)
    _validate_configuration(s, dt)
    _validate_state(s)
    if !isfinite(I_ext)
        throw(ArgumentError("current must be finite"))
    end
    inp = s.gain * max(0.0, I_ext)
    if !isfinite(inp)
        throw(ArgumentError("input drive must be finite"))
    end
    next_persistent = _first_order_relaxation(s.persistent, s.persistent_gain * inp, dt, s.tau_persistent)
    next_persistent = max(0.0, next_persistent)
    next_v = _first_order_relaxation(s.v, s.v_rest + inp + next_persistent, dt, s.tau_m)
    if !_all_finite(next_persistent, next_v)
        throw(ArgumentError("candidate state must be finite"))
    end
    s.persistent = next_persistent
    if next_v >= s.v_threshold
        s.v = s.v_reset
        return 1
    end
    s.v = max(UBC_V_MIN, min(UBC_V_MAX, next_v))
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.5)
    if n_steps < 0
        throw(ArgumentError("n_steps must be non-negative"))
    end
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
