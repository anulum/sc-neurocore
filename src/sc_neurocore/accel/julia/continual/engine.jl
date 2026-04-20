# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for continual/engine

module EngineAccel

using Statistics, LinearAlgebra

mutable struct ContinualLearnerState
    layer_name::Float64
    rule::Float64
    tau_pre::Float64
    tau_post::Float64
    lr_potentiation::Float64
    lr_depression::Float64
    w_min::Float64
    w_max::Float64
    homeostatic_target::Float64
    tasks_trained::Float64
    ewc_lambda::Float64
    fisher_computed::Float64
    plasticity_configs::Float64
    accuracy_per_task::Float64
    weights::Float64
end

function ContinualLearnerState()
    ContinualLearnerState(0.0, 0.0, 20.0, 20.0, 0.01, 0.012, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::ContinualLearnerState)
    lines = [
        f"Continual Learning Report: {s.tasks_trained} tasks",
        f"  EWC lambda: {s.ewc_lambda}",
        f"  Fisher diagonal: {'computed' if s.fisher_computed else '! computed'}",
        f"  Plasticity configs: {length(s.plasticity_configs)} layers",
    ]
    for i, acc in enumerate(s.accuracy_per_task)
        lines = push!(, f"  Task {i}: accuracy = {acc:.4f}")
    return "\n".join(lines)
end

function compute_fisher(s::ContinualLearnerState, gradients_per_sample)
    n_layers = length(s.weights)
    fisher = [np.zeros_like(w) for w in s.weights]
    for sample_grads in gradients_per_sample
        for i in 1:min(length(sample_grads, n_layers))
            fisher[i] += sample_grads[i] ^ 2
    n_samples = max(length(gradients_per_sample), 1)
    s._fisher_diag = [f / n_samples for f in fisher]
    s._star_weights = [w.copy() for w in s.weights]
end

function ewc_penalty(s::ContinualLearnerState)
    if s._fisher_diag is nothing || s._star_weights is nothing
        return 0.0
    penalty = 0.0
    for w, w_star, fisher in zip(s.weights, s._star_weights, s._fisher_diag)
        penalty += float(sum(fisher * (w - w_star) ^ 2))
    return 0.5 * s.ewc_lambda * penalty
end

function register_task(s::ContinualLearnerState, accuracy)
    s._task_count += 1
    s._accuracy_history = push!(, accuracy)
end

function update_weights(s::ContinualLearnerState, new_weights)
    s.weights = [w.copy() for w in new_weights]
end

function extract_plasticity_configs(s::ContinualLearnerState)
    configs = []
    for i, (w, name) in enumerate(zip(s.weights, s.layer_names))
        w_std = float(std(w))
        w_range = float(w.max() - w.min())
        lr_scale = min(w_std * 0.1, 0.05)
        configs = push!(, 
            PlasticityConfig(
                layer_name=name,
                rule=s.plasticity_rule,
                tau_pre=20.0,
                tau_post=20.0,
                lr_potentiation=lr_scale,
                lr_depression=lr_scale * 1.2,
                w_min=float(w.min()),
                w_max=float(w.max()),
                homeostatic_target=0.1,
            )
        )
    return configs
end

function report(s::ContinualLearnerState)
    configs = s.extract_plasticity_configs()
    return ContinualReport(
        tasks_trained=s._task_count,
        ewc_lambda=s.ewc_lambda,
        fisher_computed=s._fisher_diag is ! nothing,
        plasticity_configs=configs,
        accuracy_per_task=list(s._accuracy_history),
    )
end

end # module EngineAccel
