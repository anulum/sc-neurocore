# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for adaptive_threshold_moe

module AdaptiveThresholdMoeAccel

export step!, step_collapsed, simulate, sparsity, valid, AdaptiveThresholdMoENeuronState

mutable struct AdaptiveThresholdMoENeuronState
    k::Float64
    ema_alpha::Float64
    v::Float64
    v_th::Float64
    _mean_abs_x::Float64
end

function AdaptiveThresholdMoENeuronState()
    AdaptiveThresholdMoENeuronState(4.0, 0.1, 0.0, 1.0, 0.0)
end

function valid(s::AdaptiveThresholdMoENeuronState)
    return all(isfinite, (s.k, s.ema_alpha, s.v, s.v_th, s._mean_abs_x)) &&
        s.k > 0.0 &&
        s.ema_alpha > 0.0 &&
        s.ema_alpha <= 1.0 &&
        s.v_th > 0.0 &&
        s._mean_abs_x >= 0.0
end

function threshold_from_mean(mean_abs_x::Float64, k::Float64)
    if !isfinite(mean_abs_x) || mean_abs_x < 0.0 || !isfinite(k) || k <= 0.0
        throw(DomainError(mean_abs_x, "AdaptiveThresholdMoE adaptive threshold mean must remain finite and non-negative"))
    end
    v_th = (mean_abs_x > 1e-12) ? mean_abs_x / k : 1.0
    if !isfinite(v_th) || v_th <= 0.0
        throw(DomainError(v_th, "AdaptiveThresholdMoE adaptive threshold must remain finite and positive"))
    end
    return v_th
end

function step_collapsed(s::AdaptiveThresholdMoENeuronState, activation)
    if !isfinite(activation)
        throw(DomainError(activation, "AdaptiveThresholdMoE activation must be finite"))
    end
    if !valid(s)
        throw(DomainError(s.v, "AdaptiveThresholdMoE runtime state must be finite and physically valid"))
    end
    next_mean_abs_x = (1.0 - s.ema_alpha) * s._mean_abs_x + s.ema_alpha * abs(activation)
    next_v_th = threshold_from_mean(next_mean_abs_x, s.k)
    ratio = activation / next_v_th
    if !isfinite(ratio)
        throw(DomainError(ratio, "AdaptiveThresholdMoE threshold ratio must remain finite"))
    end
    s_int = max(round(ratio), 0)
    s._mean_abs_x = next_mean_abs_x
    s.v_th = next_v_th
    return s_int
end

function sparsity(s::AdaptiveThresholdMoENeuronState)
    if !valid(s)
        throw(DomainError(s.v, "AdaptiveThresholdMoE runtime state must be finite and physically valid"))
    end
    return (abs(s.v) < s.v_th) ? 1.0 : 0.0
end

function step!(s::AdaptiveThresholdMoENeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "AdaptiveThresholdMoE input current must be finite"))
    end
    if !valid(s)
        throw(DomainError(s.v, "AdaptiveThresholdMoE runtime state must be finite and physically valid"))
    end
    next_mean_abs_x = (1.0 - s.ema_alpha) * s._mean_abs_x + s.ema_alpha * abs(I_ext)
    next_v_th = threshold_from_mean(next_mean_abs_x, s.k)
    next_v = s.v + I_ext
    if !isfinite(next_v)
        throw(DomainError(next_v, "AdaptiveThresholdMoE soft reset residual must remain finite"))
    end
    ratio = next_v / next_v_th
    if !isfinite(ratio)
        throw(DomainError(ratio, "AdaptiveThresholdMoE threshold ratio must remain finite"))
    end
    s_int = max(round(ratio), 0)
    residual = (s_int != 0) ? next_v - next_v_th * s_int : next_v
    if !isfinite(residual)
        throw(DomainError(residual, "AdaptiveThresholdMoE soft reset residual must remain finite"))
    end
    s._mean_abs_x = next_mean_abs_x
    s.v_th = next_v_th
    s.v = residual
    return s_int
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AdaptiveThresholdMoENeuronState()
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

end # module AdaptiveThresholdMoeAccel
