# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for ai_optimized

module AiOptimizedAccel

export step!, simulate, MetaPlasticNeuronState

mutable struct MetaPlasticNeuronState
    v_fast::Float64
    v_medium::Float64
    v_slow::Float64
    tau_fast::Float64
    tau_medium::Float64
    tau_slow::Float64
    alpha::Float64
    beta::Float64
    gamma::Float64
    theta_base::Float64
    dt::Float64
    v::Float64
    w_key::Float64
    w_query::Float64
    tau::Float64
    theta::Float64
    pred::Float64
    tau_pred::Float64
    target_rate::Float64
    window::Float64
    _history::Float64
    _step_count::Float64
    phi::Float64
    amplitude::Float64
    omega::Float64
    coupling::Float64
    n_units::Float64
    sigma_e::Float64
    excitation::Float64
    inhibition::Float64
end

function MetaPlasticNeuronState()
    MetaPlasticNeuronState(0.0, 0.0, 0.0, 5.0, 200.0, 10000.0, 0.9, 5.0, 0.3, 1.0, 1.0, 0.0, 1.0, 0.5, 10.0, 1.0, 0.0, 50.0, 0.1, 50.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 16.0, 1.0, 4.0, 0.5)
end

function surrogate_grad(s::MetaPlasticNeuronState)
    return 1.0 / (1.0 + s.beta * abs(s.v - s.theta)) ^ 2
end

function _build_weights(s::MetaPlasticNeuronState)
    n = s.n_units
    s._weights = [[0.0] * n for _ in range(n)]
    for i in 1:n
        for j in 1:n
            d = min(abs(i - j), n - abs(i - j))
            s._weights[i][j] = s.excitation * exp(-d * d / (2.0 * s.sigma_e ^ 2)) - s.inhibition
        end
    end
end

function _activation(s::MetaPlasticNeuronState, x)
    r = max(0.0, x)
    return r * r / (1.0 + r * r)
end

function bump_position(s::MetaPlasticNeuronState)
    return s.u.index(max(s.u))
end

function update_meta(s::MetaPlasticNeuronState, reward)
    error = abs(reward - s.expected_reward)
    s.error_trace += (-s.error_trace + error) / s.tau_meta * s.dt
    meta_lr = s.lr0 / (1.0 + exp(-s.kappa * (s.error_trace - s.target_error)))
    s.expected_reward += meta_lr * (reward - s.expected_reward)
end

function meta_lr(s::MetaPlasticNeuronState)
    return s.lr0 / (1.0 + exp(-s.kappa * (s.error_trace - s.target_error)))
end

function step!(s::MetaPlasticNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v += (-s.v + I_ext) / s.tau * s.dt
        if s.v >= s.theta
            s.v = 0.0
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = MetaPlasticNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v_fast
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module AiOptimizedAccel
