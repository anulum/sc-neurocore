# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for few-shot HAAM

module HaamAccel

using LinearAlgebra
using Statistics

export HebbianFewShotState,
    SpikePrototypeNetState,
    store!,
    query_scores,
    query,
    reset!,
    export_weights,
    classify!,
    export_prototypes,
    validate_haam

const VALID_METRICS = Set(["cosine", "euclidean", "hamming"])

mutable struct HebbianFewShotState
    n_features::Int
    n_classes::Int
    lr_hebbian::Float64
    memory::Matrix{Float64}
    counts::Vector{Int}
end

function HebbianFewShotState(n_features::Int, n_classes::Int, lr_hebbian::Real=0.1)
    n_features > 0 || throw(ArgumentError("n_features must be positive"))
    n_classes > 0 || throw(ArgumentError("n_classes must be positive"))
    lr = Float64(lr_hebbian)
    isfinite(lr) && lr >= 0.0 || throw(ArgumentError("lr_hebbian must be finite and non-negative"))
    HebbianFewShotState(
        n_features,
        n_classes,
        lr,
        zeros(Float64, n_classes, n_features),
        zeros(Int, n_classes),
    )
end

mutable struct SpikePrototypeNetState
    n_features::Int
    metric::String
    prototypes::Dict{Int, Vector{Float64}}
end

function SpikePrototypeNetState(n_features::Int, metric::String="cosine")
    n_features > 0 || throw(ArgumentError("n_features must be positive"))
    metric in VALID_METRICS || throw(ArgumentError("metric must be one of: cosine, euclidean, hamming"))
    SpikePrototypeNetState(n_features, metric, Dict{Int, Vector{Float64}}())
end

function _feature_vector(pattern::AbstractVector{<:Real}, n_features::Int, name::String)
    length(pattern) == n_features || throw(ArgumentError("$name must resolve to $n_features features"))
    vector = Float64.(pattern)
    all(isfinite, vector) || throw(ArgumentError("$name must contain only finite values"))
    vector
end

function _validate_label(label::Int, n_classes::Int)
    0 <= label < n_classes || throw(ArgumentError("label must be in [0, $n_classes)"))
    label
end

function _cosine_score(lhs::Vector{Float64}, rhs::Vector{Float64})
    denom = norm(lhs) * norm(rhs)
    denom <= 1e-12 ? 0.0 : dot(lhs, rhs) / denom
end

function _metric_score(metric::String, query_vec::Vector{Float64}, prototype::Vector{Float64})
    if metric == "cosine"
        return _cosine_score(query_vec, prototype)
    elseif metric == "euclidean"
        return -norm(query_vec .- prototype)
    end
    disagreements = count((query_vec .> 0.0) .!= (prototype .> 0.0))
    -Float64(disagreements) / Float64(length(query_vec))
end

function store!(state::HebbianFewShotState, pattern::AbstractVector{<:Real}, label::Int)
    class_index = _validate_label(label, state.n_classes) + 1
    vector = _feature_vector(pattern, state.n_features, "spike_pattern")
    state.memory[class_index, :] .+= state.lr_hebbian .* vector
    state.counts[class_index] += 1
    nothing
end

function query_scores(state::HebbianFewShotState, pattern::AbstractVector{<:Real})
    vector = _feature_vector(pattern, state.n_features, "spike_pattern")
    scores = zeros(Float64, state.n_classes)
    for class_index in 1:state.n_classes
        if state.counts[class_index] > 0
            scores[class_index] = _cosine_score(vec(state.memory[class_index, :]), vector)
        end
    end
    scores
end

function query(state::HebbianFewShotState, pattern::AbstractVector{<:Real})
    any(!=(0), state.counts) || throw(ArgumentError("at least one support example must be stored before query"))
    argmax(query_scores(state, pattern)) - 1
end

function reset!(state::HebbianFewShotState)
    fill!(state.memory, 0.0)
    fill!(state.counts, 0)
    nothing
end

function export_weights(state::HebbianFewShotState)
    copy(state.memory)
end

function _build_prototypes(
    support_x::Vector{<:AbstractVector{<:Real}},
    support_y::Vector{Int},
    n_features::Int,
)
    isempty(support_x) && throw(ArgumentError("support_x must contain at least one support pattern"))
    length(support_x) == length(support_y) || throw(ArgumentError("support_x and support_y must have the same length"))
    prototypes = Dict{Int, Vector{Float64}}()
    for label in sort(unique(support_y))
        rows = [_feature_vector(pattern, n_features, "support pattern") for (pattern, y) in zip(support_x, support_y) if y == label]
        prototypes[label] = vec(mean(reduce(hcat, rows), dims=2))
    end
    prototypes
end

function classify!(
    state::SpikePrototypeNetState,
    support_x::Vector{<:AbstractVector{<:Real}},
    support_y::Vector{Int},
    query_x::Vector{<:AbstractVector{<:Real}},
)
    state.prototypes = _build_prototypes(support_x, support_y, state.n_features)
    labels = sort(collect(keys(state.prototypes)))
    predictions = Int[]
    for query_pattern in query_x
        query_vec = _feature_vector(query_pattern, state.n_features, "query")
        best_label = labels[1]
        best_score = -Inf
        for label in labels
            score = _metric_score(state.metric, query_vec, state.prototypes[label])
            if score > best_score
                best_score = score
                best_label = label
            end
        end
        push!(predictions, best_label)
    end
    predictions
end

function export_prototypes(state::SpikePrototypeNetState)
    Dict(label => copy(prototype) for (label, prototype) in state.prototypes)
end

function validate_haam()
    learner = HebbianFewShotState(4, 2, 0.1)
    store!(learner, [1.0, 0.0, 0.0, 0.0], 0)
    query(learner, [0.9, 0.0, 0.0, 0.0]) == 0
end

end # module HaamAccel
