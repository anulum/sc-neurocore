# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for contrastive/ssl

module SslAccel

using Statistics, LinearAlgebra

mutable struct CSDPRuleState
    temperature::Float64
    lr::Float64
    decay::Float64
end

function CSDPRuleState()
    CSDPRuleState(0.0, 0.01, 0.001)
end

function compute(s::CSDPRuleState)
    self,
    view_a: np.ndarray[Any, Any],
    view_b: np.ndarray[Any, Any],
    ) -> float
    batch = view_a.shape[0]
    if batch < 2
        return 0.0
    # Normalize
    a_norm = view_a / clamp(norm(view_a, axis=1, keepdims=true), 1e-8, nothing)
    b_norm = view_b / clamp(norm(view_b, axis=1, keepdims=true), 1e-8, nothing)
    # Similarity matrix
    sim = a_norm @ b_norm.T / s.temperature
    # InfoNCE: positive = diagonal, negatives = off-diagonal
    # log softmax along rows
    exp_sim = exp(sim - sim.max(axis=1, keepdims=true))
    log_prob = log(
        np.clip(
            np.diag(exp_sim) / exp_sim.sum(axis=1),
            1e-10,
            nothing,
        )
    )
    return -float(log_prob.mean())
end

function positive_update(s::CSDPRuleState)
    self,
    weights: np.ndarray[Any, Any],
    pre_spikes: np.ndarray[Any, Any],
    post_spikes: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]
    dW = s.lr * np.outer(post_spikes, pre_spikes) - s.decay * weights
    return weights + dW
end

function negative_update(s::CSDPRuleState)
    self,
    weights: np.ndarray[Any, Any],
    pre_spikes: np.ndarray[Any, Any],
    post_spikes: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]
    dW = -s.lr * np.outer(post_spikes, pre_spikes)
    return weights + dW
end

function contrastive_step(s::CSDPRuleState)
    self,
    weights: np.ndarray[Any, Any],
    pos_pre: np.ndarray[Any, Any],
    pos_post: np.ndarray[Any, Any],
    neg_pre: np.ndarray[Any, Any],
    neg_post: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]
    w = s.positive_update(weights, pos_pre, pos_post)
    w = s.negative_update(w, neg_pre, neg_post)
    return w
end

function goodness(s::CSDPRuleState, activations, Any])
    return float(sum(activations^2))
end

end # module SslAccel
