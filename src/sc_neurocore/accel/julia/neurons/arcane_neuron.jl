# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for arcane_neuron

module ArcaneNeuronAccel

export step!, simulate, ArcaneNeuronState

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
    w_gate::Float64
    w_pred::Float64
    kappa::Float64
    surprise_baseline::Float64
    lr_base::Float64
    eta::Float64
    _prediction::Float64
    _surprise::Float64
    _novelty::Float64
    _confidence::Float64
    _spike_history::Float64
    _novelty_history::Float64
    _hist_idx::Float64
    _nov_idx::Float64
    _total_steps::Float64
    w_inh::Float64
    dt::Float64
end

function ArcaneNeuronState()
    ArcaneNeuronState(0.0, 5.0, 0.0, 200.0, 0.3, 0.0, 10000.0, 0.05, 1.0, 0.2, 0.3, 0.0, 0.0, 5.0, 0.1, 0.01, 2.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0)
end

function identity_state(s::ArcaneNeuronState)
    return s.v_deep
end

function confidence(s::ArcaneNeuronState)
    return s._confidence
end

function novelty(s::ArcaneNeuronState)
    return s._novelty
end

function meta_learning_rate(s::ArcaneNeuronState)
    return s.lr_base * (1.0 + s.eta * s._novelty)
end

function get_state(s::ArcaneNeuronState)
    return {"v_fast": s.v_fast, "v_work": s.v_work, "v_deep": s.v_deep, "confidence": s._confidence, "novelty": s._novelty, "surprise": s._surprise, "prediction": s._prediction, "meta_lr": s.meta_learning_rate, "total_steps": s._total_steps}
end

function step!(s::ArcaneNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        spike_rate = sum(s._spike_history) / length(s._spike_history)
        s._confidence = 1.0 - mean(s._novelty_history)
        gate_input = s.w_gate[0] * I_ext + s.w_gate[1] * s.v_fast + s.w_gate[2] * s.v_work + s.w_gate[3] * s._confidence
        gate = 1.0 / (1.0 + exp(-gate_input))
        i_eff = gate * I_ext
        s.v_fast += (-s.v_fast + i_eff - s.w_inh * spike_rate) / s.tau_fast * s.dt
        s._prediction = s.w_pred[0] * s.v_fast + s.w_pred[1] * s.v_work + s.w_pred[2] * s.v_deep
        s._surprise = abs(s.v_fast - s._prediction)
        s._novelty = 1.0 / (1.0 + exp(-s.kappa * (s._surprise - s.surprise_baseline)))
        s._novelty_history[s._nov_idx % length(s._novelty_history)] = s._novelty
        s._nov_idx += 1
        eff_threshold = s.theta * (1.0 + s.gamma * s.v_deep) * (1.0 - s.delta_conf * s._confidence)
        eff_threshold = max(eff_threshold, 0.1)
        spike = (s.v_fast >= eff_threshold) ? 1 : 0
        if spike
            s.v_work += s.alpha_w * s.v_fast / s.tau_work * s.dt
            s.v_fast = 0.0
        end
        s.v_work += -s.v_work / s.tau_work * s.dt
        s.v_deep += (-s.v_deep + s.alpha_d * s.v_work * s._novelty) / s.tau_deep * s.dt
        meta_lr = s.lr_base * (1.0 + s.eta * s._novelty)
        error = s.v_fast - s._prediction
        s.w_pred[0] += meta_lr * error * s.v_fast
        s.w_pred[1] += meta_lr * error * s.v_work
        s.w_pred[2] += meta_lr * error * s.v_deep
        norm = np.linalg.norm(s.w_pred)
        if norm > 0
            s.w_pred /= norm
        end
        s._spike_history[s._hist_idx % length(s._spike_history)] = spike
        s._hist_idx += 1
        s._total_steps += 1
        return spike
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ArcaneNeuronState()
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

end # module ArcaneNeuronAccel
