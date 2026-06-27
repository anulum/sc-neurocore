# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for contrastive/ssl

module SslAccel

using LinearAlgebra

export SpikeContrastiveLossState,
    CSDPRuleState,
    compute,
    positive_update,
    negative_update,
    contrastive_step,
    goodness,
    validate_ssl

struct SpikeContrastiveLossState
    temperature::Float64
end

function SpikeContrastiveLossState(temperature::Real=0.5)
    scalar = Float64(temperature)
    isfinite(scalar) && scalar > 0.0 || throw(ArgumentError("temperature must be finite and positive"))
    SpikeContrastiveLossState(scalar)
end

struct CSDPRuleState
    lr::Float64
    decay::Float64
end

function CSDPRuleState(lr::Real=0.01, decay::Real=0.001)
    lr_scalar = Float64(lr)
    decay_scalar = Float64(decay)
    isfinite(lr_scalar) && lr_scalar >= 0.0 || throw(ArgumentError("lr must be finite and non-negative"))
    isfinite(decay_scalar) && decay_scalar >= 0.0 || throw(ArgumentError("decay must be finite and non-negative"))
    CSDPRuleState(lr_scalar, decay_scalar)
end

function compute(state::SpikeContrastiveLossState, view_a::AbstractMatrix{<:Real}, view_b::AbstractMatrix{<:Real})
    a = _validated_matrix(view_a, "view_a")
    b = _validated_matrix(view_b, "view_b")
    size(a) == size(b) || throw(ArgumentError("view_a and view_b must have the same shape"))
    batch = size(a, 1)
    batch < 2 && return 0.0

    a_norm = _normalise_rows(a)
    b_norm = _normalise_rows(b)
    logits = (a_norm * transpose(b_norm)) ./ state.temperature
    total = 0.0

    for row in 1:batch
        shifted = logits[row, :] .- maximum(logits[row, :])
        exp_logits = exp.(shifted)
        prob = max(exp_logits[row] / sum(exp_logits), 1e-10)
        total += log(prob)
    end

    -total / Float64(batch)
end

function positive_update(
    state::CSDPRuleState,
    weights::AbstractMatrix{<:Real},
    pre_spikes::AbstractVector{<:Real},
    post_spikes::AbstractVector{<:Real},
)
    w, pre, post = _validated_update_inputs(weights, pre_spikes, post_spikes)
    w .+ state.lr .* (post * transpose(pre)) .- state.decay .* w
end

function negative_update(
    state::CSDPRuleState,
    weights::AbstractMatrix{<:Real},
    pre_spikes::AbstractVector{<:Real},
    post_spikes::AbstractVector{<:Real},
)
    w, pre, post = _validated_update_inputs(weights, pre_spikes, post_spikes)
    w .- state.lr .* (post * transpose(pre))
end

function contrastive_step(
    state::CSDPRuleState,
    weights::AbstractMatrix{<:Real},
    pos_pre::AbstractVector{<:Real},
    pos_post::AbstractVector{<:Real},
    neg_pre::AbstractVector{<:Real},
    neg_post::AbstractVector{<:Real},
)
    after_positive = positive_update(state, weights, pos_pre, pos_post)
    negative_update(state, after_positive, neg_pre, neg_post)
end

function goodness(::CSDPRuleState, activations::AbstractArray{<:Real})
    values = Float64.(activations)
    all(isfinite, values) || throw(ArgumentError("activations must contain only finite values"))
    sum(values .^ 2)
end

function validate_ssl()
    loss = SpikeContrastiveLossState(0.5)
    view = Matrix{Float64}(I, 3, 3)
    rule = CSDPRuleState(0.1, 0.01)
    weights = [0.2 0.4; 0.1 0.3]
    updated = contrastive_step(rule, weights, [1.0, 0.5], [0.25, 1.0], [0.0, 1.0], [0.5, 0.5])
    compute(loss, view, view) >= 0.0 && size(updated) == size(weights) && goodness(rule, [1.0, -2.0, 0.5]) ≈ 5.25
end

function _validated_matrix(values::AbstractMatrix{<:Real}, name::String)
    matrix = Float64.(values)
    size(matrix, 2) > 0 || throw(ArgumentError("$name must contain at least one feature"))
    all(isfinite, matrix) || throw(ArgumentError("$name must contain only finite values"))
    matrix
end

function _validated_vector(values::AbstractVector{<:Real}, name::String)
    vector = Float64.(values)
    all(isfinite, vector) || throw(ArgumentError("$name must contain only finite values"))
    vector
end

function _validated_update_inputs(
    weights::AbstractMatrix{<:Real},
    pre_spikes::AbstractVector{<:Real},
    post_spikes::AbstractVector{<:Real},
)
    w = _validated_matrix(weights, "weights")
    pre = _validated_vector(pre_spikes, "pre_spikes")
    post = _validated_vector(post_spikes, "post_spikes")
    size(w) == (length(post), length(pre)) ||
        throw(ArgumentError("weights must have shape (length(post_spikes), length(pre_spikes))"))
    w, pre, post
end

function _normalise_rows(values::Matrix{Float64})
    out = copy(values)
    for row in axes(out, 1)
        denom = max(norm(@view out[row, :]), 1e-8)
        out[row, :] ./= denom
    end
    out
end

end # module SslAccel
