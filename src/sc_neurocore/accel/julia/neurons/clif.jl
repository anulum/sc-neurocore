# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for clif

module ClifAccel

export step!, simulate, ComplementaryLIFNeuronState, validate

const V_MAX = 1.0e12

mutable struct ComplementaryLIFNeuronState
    v_pos::Float64
    v_neg::Float64
    tau::Float64
    v_threshold::Float64
    dt::Float64
    alpha::Float64
end

function ComplementaryLIFNeuronState()
    tau = 10.0
    dt = 1.0
    ComplementaryLIFNeuronState(0.0, 0.0, tau, 1.0, dt, exp(-dt / tau))
end

_finite(value::Float64) = isfinite(value)

function _validated_alpha(s::ComplementaryLIFNeuronState)
    if !_finite(s.tau) || s.tau <= 0.0 || !_finite(s.dt) || s.dt <= 0.0
        error("tau and dt must be positive")
    end
    ratio = -s.dt / s.tau
    alpha = ratio < -700.0 ? 0.0 : exp(ratio)
    if !_finite(alpha) || alpha < 0.0 || alpha >= 1.0
        error("alpha must be in [0, 1)")
    end
    return alpha
end

function validate(s::ComplementaryLIFNeuronState)
    if !_finite(s.v_pos) || !_finite(s.v_neg) || abs(s.v_pos) > V_MAX || abs(s.v_neg) > V_MAX
        return false
    end
    if !_finite(s.v_threshold) || s.v_threshold <= 0.0
        return false
    end
    try
        _validated_alpha(s)
    catch _e
        return false
    end
    return true
end

function step!(s::ComplementaryLIFNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !isfinite(I_ext) || !validate(s)
        return -2
    end
    alpha = _validated_alpha(s)
    inp_pos = max(I_ext, 0.0)
    inp_neg = max(-I_ext, 0.0)
    v_pos_next = alpha * s.v_pos + inp_pos
    v_neg_next = alpha * s.v_neg + inp_neg
    diff = v_pos_next - v_neg_next
    if !isfinite(v_pos_next) || !isfinite(v_neg_next) || !isfinite(diff) || abs(v_pos_next) > V_MAX || abs(v_neg_next) > V_MAX
        return -2
    end
    s.alpha = alpha
    if diff >= s.v_threshold
        s.v_pos = 0.0
        s.v_neg = 0.0
        return 1
    end
    if diff <= -s.v_threshold
        s.v_pos = 0.0
        s.v_neg = 0.0
        return -1
    end
    s.v_pos = v_pos_next
    s.v_neg = v_neg_next
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ComplementaryLIFNeuronState()
    s.dt = dt
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = result == -2 ? NaN : s.v_pos - s.v_neg
        if result == 1 || result == -1
            spikes += 1
        end
    end
    return trace, spikes
end

end # module ClifAccel
