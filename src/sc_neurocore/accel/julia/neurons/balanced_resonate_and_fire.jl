# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for balanced_resonate_and_fire

module BalancedResonateAndFireAccel

export BalancedResonateAndFireNeuronState, damping, dynamic_threshold, simulate, step!, sustain_oscillation_boundary

mutable struct BalancedResonateAndFireNeuronState
    x::Float64
    y::Float64
    q::Float64
    omega::Float64
    b_offset::Float64
    threshold::Float64
    gamma::Float64
    dt::Float64
end

function BalancedResonateAndFireNeuronState()
    BalancedResonateAndFireNeuronState(0.0, 0.0, 0.0, 10.0, 1.0, 1.0, 0.9, 0.01)
end

function sustain_oscillation_boundary(omega::Float64, dt::Float64)::Float64
    if dt <= 0.0
        throw(ArgumentError("dt must be positive"))
    end
    if omega <= 0.0
        throw(ArgumentError("omega must be positive"))
    end
    scaled = dt * omega
    if scaled > 1.0
        throw(ArgumentError("dt * omega must be <= 1"))
    end
    return (-1.0 + sqrt(max(0.0, 1.0 - scaled * scaled))) / dt
end

function _validate(s::BalancedResonateAndFireNeuronState)::Nothing
    if s.dt <= 0.0 || !isfinite(s.dt)
        throw(ArgumentError("dt must be finite and positive"))
    end
    if s.omega <= 0.0 || !isfinite(s.omega)
        throw(ArgumentError("omega must be finite and positive"))
    end
    if s.dt * s.omega > 1.0
        throw(ArgumentError("dt * omega must be <= 1"))
    end
    if s.b_offset <= 0.0 || !isfinite(s.b_offset)
        throw(ArgumentError("b_offset must be finite and positive"))
    end
    if s.threshold <= 0.0 || !isfinite(s.threshold)
        throw(ArgumentError("threshold must be finite and positive"))
    end
    if s.gamma < 0.0 || s.gamma >= 1.0 || !isfinite(s.gamma)
        throw(ArgumentError("gamma must satisfy 0 <= gamma < 1"))
    end
    if !(isfinite(s.x) && isfinite(s.y) && isfinite(s.q))
        throw(ArgumentError("state variables must be finite"))
    end
    return nothing
end

function damping(s::BalancedResonateAndFireNeuronState)::Float64
    _validate(s)
    return sustain_oscillation_boundary(s.omega, s.dt) - s.b_offset - s.q
end

function dynamic_threshold(s::BalancedResonateAndFireNeuronState)::Float64
    _validate(s)
    return s.threshold + s.q
end

function step!(s::BalancedResonateAndFireNeuronState, current::Float64=0.0)::Int
    _validate(s)
    b_t = damping(s)
    theta_t = dynamic_threshold(s)
    x_prev = s.x
    y_prev = s.y
    s.x = x_prev + s.dt * (b_t * x_prev - s.omega * y_prev + current)
    s.y = y_prev + s.dt * (s.omega * x_prev + b_t * y_prev)
    spike = s.x >= theta_t ? 1 : 0
    s.q = s.gamma * s.q + spike
    return spike
end

function simulate(n_steps::Int=1000; current::Float64=2.0, I_ext=nothing, dt=nothing)
    s = BalancedResonateAndFireNeuronState()
    if I_ext !== nothing
        current = Float64(I_ext)
    end
    if dt !== nothing
        s.dt = Float64(dt)
    end
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        spike = step!(s, current)
        trace[t] = s.x
        spikes += spike
    end
    return trace, spikes
end

end # module BalancedResonateAndFireAccel
