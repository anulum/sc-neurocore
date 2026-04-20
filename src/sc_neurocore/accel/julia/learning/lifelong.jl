# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for learning/lifelong

module LifelongAccel

using Statistics, LinearAlgebra

mutable struct EWC_SCLayerState
    ewc_lambda::Float64
end

function EWC_SCLayerState()
    EWC_SCLayerState(10.0)
end

function consolidate_task(s::EWC_SCLayerState)
    # In SC, Fisher Info approx ~ Activity * Plasticity
    # Weights that changed a lot || are high are often important.
    # Simplified: Importance = Current Weight Magnitude (Hebbian)
    current_w = s.get_weights()
    s.star_weights = current_w.copy()
    # Assume all non-zero weights are somewhat important
    s.fisher_info = current_w.copy()
end

function apply_ewc_penalty(s::EWC_SCLayerState, step_size)
    current_w = s.get_weights()
    delta = current_w - s.star_weights
    penalty_grad = s.fisher_info * delta
    correction = s.ewc_lambda * step_size * penalty_grad
    new_w = clamp(current_w - correction, s.w_min, s.w_max)
    for i in 1:s.n_neurons
        for j in 1:s.n_inputs
            s.synapses[i][j].w = float(new_w[i, j])
    return float(sum(abs(penalty_grad)))
end

end # module LifelongAccel
