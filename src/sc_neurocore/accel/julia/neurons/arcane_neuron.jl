# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for arcane_neuron

module ArcaneNeuronAccel

export step!, simulate, ArcaneNeuronState, valid, reset!, identity_state, confidence, novelty, meta_learning_rate, get_state

mutable struct ArcaneNeuronState
    v_fast::Float64
    tau_fast::Float64
    v_work::Float64
    tau_work::Float64
    alpha_w::Float64
    v_deep::Float64
    tau_deep::Float64
    alpha_d::Float64
    theta::Float64
    gamma::Float64
    delta_conf::Float64
    w_gate::Vector{Float64}
    w_pred::Vector{Float64}
    kappa::Float64
    surprise_baseline::Float64
    lr_base::Float64
    eta::Float64
    prediction::Float64
    surprise::Float64
    current_novelty::Float64
    current_confidence::Float64
    spike_history::Vector{Float64}
    novelty_history::Vector{Float64}
    hist_idx::Int
    nov_idx::Int
    total_steps::Int
    identity_drift::Float64
    w_inh::Float64
    dt::Float64
end

function ArcaneNeuronState()
    ArcaneNeuronState(
        0.0,
        5.0,
        0.0,
        200.0,
        0.3,
        0.0,
        10000.0,
        0.05,
        1.0,
        0.2,
        0.3,
        [0.8, 0.1, 0.05, 0.05],
        [0.6, 0.3, 0.1],
        5.0,
        0.1,
        0.01,
        2.0,
        0.0,
        0.0,
        0.0,
        0.5,
        fill(0.0, 50),
        fill(0.5, 20),
        0,
        0,
        0,
        0.0,
        0.3,
        1.0,
    )
end

function valid(s::ArcaneNeuronState)::Bool
    scalars = (
        s.v_fast,
        s.tau_fast,
        s.v_work,
        s.tau_work,
        s.alpha_w,
        s.v_deep,
        s.tau_deep,
        s.alpha_d,
        s.theta,
        s.gamma,
        s.delta_conf,
        s.kappa,
        s.surprise_baseline,
        s.lr_base,
        s.eta,
        s.prediction,
        s.surprise,
        s.current_novelty,
        s.current_confidence,
        s.identity_drift,
        s.w_inh,
        s.dt,
    )
    return all(isfinite, scalars) &&
        s.tau_fast > 0.0 &&
        s.tau_work > 0.0 &&
        s.tau_deep > 0.0 &&
        s.theta > 0.0 &&
        s.alpha_w >= 0.0 &&
        s.alpha_d >= 0.0 &&
        s.lr_base >= 0.0 &&
        s.w_inh >= 0.0 &&
        s.dt > 0.0 &&
        length(s.w_gate) == 4 &&
        all(isfinite, s.w_gate) &&
        length(s.w_pred) == 3 &&
        all(isfinite, s.w_pred) &&
        length(s.spike_history) > 0 &&
        all(x -> x == 0.0 || x == 1.0, s.spike_history) &&
        length(s.novelty_history) > 0 &&
        all(isfinite, s.novelty_history) &&
        s.hist_idx >= 0 &&
        s.nov_idx >= 0 &&
        s.total_steps >= 0
end

function identity_state(s::ArcaneNeuronState)::Float64
    return s.v_deep
end

function confidence(s::ArcaneNeuronState)::Float64
    return s.current_confidence
end

function novelty(s::ArcaneNeuronState)::Float64
    return s.current_novelty
end

function meta_learning_rate(s::ArcaneNeuronState)::Float64
    return s.lr_base * (1.0 + s.eta * s.current_novelty)
end

function get_state(s::ArcaneNeuronState)::Dict{String, Float64}
    return Dict(
        "v_fast" => s.v_fast,
        "v_work" => s.v_work,
        "v_deep" => s.v_deep,
        "confidence" => s.current_confidence,
        "novelty" => s.current_novelty,
        "surprise" => s.surprise,
        "prediction" => s.prediction,
        "identity_drift" => s.identity_drift,
        "meta_lr" => meta_learning_rate(s),
        "total_steps" => Float64(s.total_steps),
    )
end

function step!(s::ArcaneNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    if !isfinite(I_ext) || !isfinite(dt) || dt <= 0.0 || !valid(s)
        throw(DomainError((s.v_fast, s.v_work, s.v_deep, I_ext), "ArcaneNeuron state/current must be finite and physically valid"))
    end

    spike_rate = sum(s.spike_history) / length(s.spike_history)
    current_confidence = 1.0 - sum(s.novelty_history) / length(s.novelty_history)
    gate_input = s.w_gate[1] * I_ext + s.w_gate[2] * s.v_fast + s.w_gate[3] * s.v_work + s.w_gate[4] * current_confidence
    gate = _stable_sigmoid(gate_input)
    i_eff = gate * I_ext
    fast_drive = i_eff - s.w_inh * spike_rate
    next_v_fast_continuous = _exact_relaxation(s.v_fast, fast_drive, dt, s.tau_fast)
    _require_finite(next_v_fast_continuous)

    prediction = s.w_pred[1] * next_v_fast_continuous + s.w_pred[2] * s.v_work + s.w_pred[3] * s.v_deep
    _require_finite(prediction)
    surprise = abs(next_v_fast_continuous - prediction)
    current_novelty = _stable_sigmoid(s.kappa * (surprise - s.surprise_baseline))

    eff_threshold = s.theta * (1.0 + s.gamma * s.v_deep) * (1.0 - s.delta_conf * current_confidence)
    _require_finite(eff_threshold)
    eff_threshold = max(eff_threshold, 0.1)
    spike = next_v_fast_continuous >= eff_threshold ? 1 : 0
    accepted_v_fast = spike == 1 ? 0.0 : next_v_fast_continuous

    work_drive = spike == 1 ? s.alpha_w * next_v_fast_continuous : 0.0
    next_v_work = _exact_relaxation(s.v_work, work_drive, dt, s.tau_work)
    _require_finite(next_v_work)

    deep_drive = s.alpha_d * next_v_work * current_novelty
    next_v_deep = _exact_relaxation(s.v_deep, deep_drive, dt, s.tau_deep)
    _require_finite(next_v_deep)

    meta_lr = s.lr_base * (1.0 + s.eta * current_novelty)
    error = accepted_v_fast - prediction
    next_w_pred = copy(s.w_pred)
    next_w_pred[1] += meta_lr * error * accepted_v_fast
    next_w_pred[2] += meta_lr * error * next_v_work
    next_w_pred[3] += meta_lr * error * next_v_deep
    norm_value = sqrt(sum(abs2, next_w_pred))
    _require_finite(norm_value)
    if norm_value > 0.0
        next_w_pred ./= norm_value
    end
    if !all(isfinite, next_w_pred)
        throw(DomainError(next_w_pred, "ArcaneNeuron predictor candidate must remain finite"))
    end

    next_novelty_history = copy(s.novelty_history)
    next_novelty_history[(s.nov_idx % length(next_novelty_history)) + 1] = current_novelty
    next_spike_history = copy(s.spike_history)
    next_spike_history[(s.hist_idx % length(next_spike_history)) + 1] = Float64(spike)

    old_v_deep = s.v_deep
    s.v_fast = accepted_v_fast
    s.v_work = next_v_work
    s.v_deep = next_v_deep
    s.prediction = prediction
    s.surprise = surprise
    s.current_novelty = current_novelty
    s.current_confidence = current_confidence
    s.novelty_history = next_novelty_history
    s.nov_idx += 1
    s.identity_drift += abs(next_v_deep - old_v_deep)
    s.w_pred = next_w_pred
    s.spike_history = next_spike_history
    s.hist_idx += 1
    s.total_steps += 1
    s.dt = dt
    return spike
end

function _exact_relaxation(state::Float64, steady_state::Float64, dt::Float64, tau::Float64)::Float64
    decay = exp(-dt / tau)
    return decay * state + (1.0 - decay) * steady_state
end

function _stable_sigmoid(x::Float64)::Float64
    if x == Inf
        return 1.0
    elseif x == -Inf
        return 0.0
    elseif x >= 0.0
        z = exp(-x)
        return 1.0 / (1.0 + z)
    end
    z = exp(x)
    return z / (1.0 + z)
end

function _require_finite(value::Float64)::Nothing
    if !isfinite(value)
        throw(DomainError(value, "ArcaneNeuron exact relaxation candidate must remain finite"))
    end
    return nothing
end

function reset!(s::ArcaneNeuronState)::Nothing
    s.v_fast = 0.0
    s.v_work = 0.0
    s.prediction = 0.0
    s.surprise = 0.0
    s.current_novelty = 0.0
    s.spike_history = fill(0.0, length(s.spike_history))
    s.hist_idx = 0
    s.identity_drift = 0.0
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=1.0)
    s = ArcaneNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v_fast
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module ArcaneNeuronAccel
