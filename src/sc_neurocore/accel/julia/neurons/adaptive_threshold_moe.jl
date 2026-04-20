# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for adaptive_threshold_moe

module AdaptiveThresholdMoeAccel

export step!, simulate, AdaptiveThresholdMoENeuronState

mutable struct AdaptiveThresholdMoENeuronState
    k::Float64
    ema_alpha::Float64
    v::Float64
    v_th::Float64
    _mean_abs_x::Float64
end

function AdaptiveThresholdMoENeuronState()
    AdaptiveThresholdMoENeuronState(4.0, 0.1, 0.0, 0.0, 0.0)
end

function step_collapsed(s::AdaptiveThresholdMoENeuronState, activation)
    s._mean_abs_x = (1.0 - s.ema_alpha) * s._mean_abs_x + s.ema_alpha * abs(activation)
    s.v_th = (s._mean_abs_x > 1e-12) ? s._mean_abs_x / s.k : 1.0
    return max(round(activation / s.v_th), 0)
end

function sparsity(s::AdaptiveThresholdMoENeuronState)
    return (abs(s.v) < s.v_th) ? 1.0 : 0.0
end

function step!(s::AdaptiveThresholdMoENeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s._mean_abs_x = (1.0 - s.ema_alpha) * s._mean_abs_x + s.ema_alpha * abs(I_ext)
        s.v_th = (s._mean_abs_x > 1e-12) ? s._mean_abs_x / s.k : 1.0
        s.v += I_ext
        s_int = (s.v_th > 1e-12) ? round(s.v / s.v_th) : 0
        if s_int != 0
            s.v -= s.v_th * s_int
        end
        return max(s_int, 0)
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AdaptiveThresholdMoENeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.k
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module AdaptiveThresholdMoeAccel
